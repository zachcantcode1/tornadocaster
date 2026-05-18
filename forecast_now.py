"""
forecast_now.py — Multi-hour RRFS tornado composite forecast map.

Fetches fstart–fend forecast hours from the latest RRFS (or HRRR) run,
computes the TC composite for each hour, then aggregates via element-wise
max + 40 km neighborhood max.  This mirrors Nadocast's "chance of tornado
within 25 miles at any point during the outlook period" structure.

Usage:
    python forecast_now.py [--output forecast.png] [--fstart 1] [--fend 18] [--model rrfs|hrrr]
"""
import argparse
import asyncio
import logging
import os
import re
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import maximum_filter

from src.ingestion.noaa_fetcher import NOAAIndexFetcher
from src.features.derived import build_first_pass_derived_fields
from src.visualization.plot_forecast import compute_tornado_composite, plot_conus_forecast
from main import (
    resolve_latest_rrfs_url,
    resolve_latest_hrrr_url,
    RRFS_FIELD_CATALOG,
    FIRST_PASS_FIELD_CATALOG,
)

# Use calibrated ML probability model when available; fall back to TC composite
try:
    from src.models.calibrated_predictor import get_predictor as _get_predictor
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# 40 km radius neighborhood — matches Nadocast's "within 25 miles of a point"
_NEIGHBORHOOD_KM  = 40.0
_RRFS_GRID_KM     = 3.0
_NEIGHBORHOOD_PX  = int(round(_NEIGHBORHOOD_KM / _RRFS_GRID_KM))  # ~13 px → 27-px kernel

# RRFS fields needed for tornado composite
_RRFS_NEEDED = {
    "cape_ml", "cin_ml", "hlcy_3km", "hlcy_1km",
    "hgt_lcl", "mxuphl_03km", "relv_1km", "efhl_surface",
    "tmp_2m", "dpt_2m", "cape_surface", "cape_mu", "cin_surface",
    "ugrd_10m", "vgrd_10m", "ugrd_pbl", "vgrd_pbl",
}

# HRRR fallback fields
_HRRR_NEEDED = {
    "cape_ml", "cin_ml", "hlcy_3km", "vwsh_0_6km", "tmp_2m", "dpt_2m",
    "cape_surface", "cape_mu", "cin_surface", "cin_mu",
    "ugrd_500", "vgrd_500", "ugrd_850", "vgrd_850",
    "ugrd_925", "vgrd_925", "tmp_500", "tmp_700",
}

_RRFS_FETCH_CATALOG = [row for row in RRFS_FIELD_CATALOG if row[0] in _RRFS_NEEDED]
_HRRR_FETCH_CATALOG = [(n, v, l, c) for (n, v, l, c) in FIRST_PASS_FIELD_CATALOG
                       if n in _HRRR_NEEDED]


async def fetch_fields(base_url: str, catalog: list) -> tuple[dict, np.ndarray, np.ndarray]:
    fetcher = NOAAIndexFetcher()
    try:
        idx_text = await fetcher.fetch_idx_file(f"{base_url}.idx")
        records  = fetcher.parse_idx_text(idx_text)
        specs    = [(row[0], row[1], row[2]) for row in catalog]
        fields   = await fetcher.fetch_named_fields(base_url, records, specs)
        if not fields:
            raise RuntimeError("No fields fetched.")

        lat2d = lon2d = None
        for da in fields.values():
            if "latitude" in da.coords and "longitude" in da.coords:
                lat2d = da.coords["latitude"].values.astype(np.float32)
                lon2d = da.coords["longitude"].values.astype(np.float32)
                lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
                break
        if lat2d is None:
            raise RuntimeError("Could not extract lat/lon coords.")
        return fields, lat2d, lon2d
    finally:
        await fetcher.close()


async def fetch_one_hour(
    url: str, catalog: list, sem: asyncio.Semaphore,
    use_ml: bool = False, valid_dt: datetime | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray] | None:
    async with sem:
        try:
            fields, lat2d, lon2d = await fetch_fields(url, catalog)
            derived    = build_first_pass_derived_fields(fields)
            all_fields = {**fields, **derived}

            if use_ml and _ML_AVAILABLE:
                try:
                    predictor = _get_predictor()
                    composite = predictor.predict(all_fields, valid_dt=valid_dt)
                    if composite is not None:
                        logger.info("  ✓ %s  max prob=%.3f (ML)", url.split("/")[-1], float(np.nanmax(composite)))
                    else:
                        composite = compute_tornado_composite(all_fields)
                except Exception as ml_exc:
                    logger.warning("  ML predictor failed (%s), falling back to TC composite", ml_exc)
                    composite = compute_tornado_composite(all_fields)
            else:
                composite = compute_tornado_composite(all_fields)

            if composite is None:
                return None
            mxuphl = all_fields.get("mxuphl_03km")
            uh_arr = np.asarray(mxuphl.values, dtype=np.float32) if mxuphl is not None else None
            if not use_ml:
                logger.info("  ✓ %s  max TC=%.2f", url.split("/")[-1], float(np.nanmax(composite)))
            return composite, uh_arr, lat2d, lon2d
        except Exception as exc:
            logger.warning("  ✗ %s — %s", url.split("/")[-1], exc)
            return None


def _swap_fhour_rrfs(url: str, fhour: int) -> str:
    return re.sub(r"\.f\d{3}\.", f".f{fhour:03d}.", url)

def _swap_fhour_hrrr(url: str, fhour: int) -> str:
    return re.sub(r"f\d{2}\.grib2", f"f{fhour:02d}.grib2", url)


async def main(output_path: str, fstart: int, fend: int, model: str, use_ml: bool = False) -> None:
    # Resolve the run anchor (using fstart so we know this hour exists)
    fetcher = NOAAIndexFetcher()
    try:
        if model == "rrfs":
            anchor_url = await resolve_latest_rrfs_url(fetcher, fhour=fstart)
            catalog    = _RRFS_FETCH_CATALOG
            model_tag  = "RRFS"
            swap_fn    = _swap_fhour_rrfs
        else:
            anchor_url = await resolve_latest_hrrr_url(fetcher, fhour=fstart)
            catalog    = _HRRR_FETCH_CATALOG
            model_tag  = "HRRR"
            swap_fn    = _swap_fhour_hrrr
    finally:
        await fetcher.close()

    # Parse run date/cycle for title
    parts = anchor_url.split("/")
    if model == "rrfs":
        date_part = next((p for p in parts if p.startswith("rrfs.202")), "rrfs.unknown")
        run_date  = date_part.replace("rrfs.", "")
        run_cycle = next((p for p in parts if len(p) == 2 and p.isdigit()), "??")
    else:
        date_part = next((p for p in parts if p.startswith("hrrr.202")), "hrrr.unknown")
        run_date  = date_part.replace("hrrr.", "")
        run_cycle = next((p[6:8] for p in parts if p.startswith("hrrr.t") and "z.wrf" in p), "??")

    run_label = (
        f"{model_tag} {run_date[:4]}-{run_date[4:6]}-{run_date[6:]} "
        f"{run_cycle}Z  F{fstart:03d}–F{fend:03d}"
    )
    logger.info("Multi-hour forecast: %s", run_label)

    # Build URL list and fetch all hours concurrently (max 4 in flight)
    fhours = list(range(fstart, fend + 1))
    urls   = [swap_fn(anchor_url, fh) for fh in fhours]

    # Compute valid datetime for each forecast hour (for temporal features)
    try:
        run_cycle_int = int(run_cycle)
        run_dt = datetime.strptime(run_date, "%Y%m%d").replace(
            hour=run_cycle_int, tzinfo=timezone.utc
        )
        valid_dts = [run_dt.replace(hour=0, minute=0, second=0) +
                     __import__("datetime").timedelta(hours=run_cycle_int + fh)
                     for fh in fhours]
    except Exception:
        valid_dts = [None] * len(fhours)

    sem    = asyncio.Semaphore(4)
    results = await asyncio.gather(*[
        fetch_one_hour(u, catalog, sem, use_ml=use_ml, valid_dt=vdt)
        for u, vdt in zip(urls, valid_dts)
    ])

    # Accumulate across hours.
    # ML mode: P(tornado in any hour) = 1 - ∏(1-p_i)  — Nadocast combination formula.
    # TC composite mode: element-wise max (physics index, not a probability).
    prob_no_event: np.ndarray | None = None   # ML only: running product of (1-p)
    max_composite: np.ndarray | None = None   # TC mode or final ML result
    max_mxuphl:    np.ndarray | None = None
    lat2d = lon2d = None
    n_fetched = 0

    for result in results:
        if result is None:
            continue
        comp, uh, lat, lon = result
        n_fetched += 1
        if lat2d is None:
            lat2d, lon2d = lat, lon

        if use_ml:
            if prob_no_event is None:
                prob_no_event = (1.0 - comp.astype(np.float64))
            else:
                prob_no_event *= (1.0 - comp.astype(np.float64))
        else:
            if max_composite is None:
                max_composite = comp.copy()
            else:
                np.maximum(max_composite, comp, out=max_composite)

        if uh is not None:
            if max_mxuphl is None:
                max_mxuphl = uh.copy()
            else:
                np.maximum(max_mxuphl, uh, out=max_mxuphl)

    if use_ml:
        if prob_no_event is None:
            raise RuntimeError("No forecast hours could be computed — check model availability.")
        max_composite = np.clip(1.0 - prob_no_event, 0.0, 1.0).astype(np.float32)
    else:
        if max_composite is None:
            raise RuntimeError("No forecast hours could be computed — check model availability.")

    logger.info("Aggregated %d/%d forecast hours.", n_fetched, len(fhours))

    # 40 km neighborhood max for TC composite mode only.
    # ML model was trained with 40 km label radius — spatial spread already encoded.
    if not use_ml:
        kernel = 2 * _NEIGHBORHOOD_PX + 1
        max_composite = maximum_filter(max_composite, size=kernel).astype(np.float32)
        if max_mxuphl is not None:
            max_mxuphl = maximum_filter(max_mxuphl, size=kernel).astype(np.float32)

    grid_h, grid_w = max_composite.shape
    if use_ml:
        lo, mid, hi = 0.02, 0.10, 0.30
        n_nonzero  = int(np.sum(max_composite >= lo))
        n_moderate = int(np.sum(max_composite >= mid))
        n_sig      = int(np.sum(max_composite >= hi))
        logger.info("Grid: %dx%d  |  P>=2%%: %d  |  P>=10%%: %d  |  P>=30%%: %d",
                    grid_h, grid_w, n_nonzero, n_moderate, n_sig)
        logger.info("Probability stats — max: %.1f%%  mean(>=2%%): %.1f%%",
                    float(np.nanmax(max_composite)) * 100,
                    float(np.mean(max_composite[max_composite >= lo])) * 100 if n_nonzero else 0.0)
    else:
        lo, mid, hi = 0.1, 1.0, 2.0
        n_nonzero  = int(np.sum(max_composite > lo))
        n_moderate = int(np.sum(max_composite >= mid))
        n_sig      = int(np.sum(max_composite >= hi))
        logger.info("Grid: %dx%d  |  TC>0.1: %d  |  TC>=1: %d  |  TC>=2: %d",
                    grid_h, grid_w, n_nonzero, n_moderate, n_sig)
        logger.info("Composite stats — max: %.2f  mean(>0.1): %.3f",
                    float(np.nanmax(max_composite)),
                    float(np.mean(max_composite[max_composite > lo])) if n_nonzero else 0.0)

    generated_by = (
        f"Tornadocaster  |  generated "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    map_title = (
        f"Tornado Probability  —  {run_label}"
        if use_ml else
        f"Tornado Composite  —  {run_label}"
    )
    plot_conus_forecast(
        lat=lat2d,
        lon=lon2d,
        stp=max_composite,
        mxuphl=max_mxuphl,
        title=map_title,
        subtitle=generated_by,
        output_path=output_path,
        dpi=150,
        prob_mode=use_ml,
    )

    abs_path = os.path.abspath(output_path)
    print(f"\nForecast map saved to:\n  {abs_path}\n")
    print(f"  Run            : {run_label}")
    print(f"  Hours fetched  : {n_fetched}/{len(fhours)}")
    print(f"  Grid size      : {grid_h} x {grid_w}")
    if use_ml:
        print(f"  Max probability: {np.nanmax(max_composite)*100:.1f}%")
        print(f"  P >= 10%       : {n_moderate} grid points")
        print(f"  P >= 30%       : {n_sig} grid points")
    else:
        print(f"  Composite max  : {np.nanmax(max_composite):.2f}")
        print(f"  TC >= 1.0      : {n_moderate} grid points")
        print(f"  TC >= 2.0      : {n_sig} grid points")
    if max_mxuphl is not None:
        print(f"  MXUPHL >= 25   : {int(np.sum(max_mxuphl >= 25.0))} grid points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a real-time multi-hour RRFS tornado composite forecast map."
    )
    parser.add_argument("--output", default="forecast.png", help="Output PNG path")
    parser.add_argument("--fstart", type=int, default=1,
                        help="First forecast hour (default: 1)")
    parser.add_argument("--fend",   type=int, default=18,
                        help="Last forecast hour (default: 18)")
    parser.add_argument("--model",  default="rrfs", choices=["rrfs", "hrrr"],
                        help="Data source: rrfs (default) or hrrr")
    parser.add_argument("--ml", action="store_true",
                        help="Use trained ML probability model (requires prior training run)")
    parser.add_argument("--ml-model", default=None,
                        help="Path to a specific .pkl model file (default: models/tornado_lgbm.pkl)")
    args = parser.parse_args()

    if args.ml_model:
        import src.models.calibrated_predictor as _cp
        _cp._DEFAULT_MODEL_PATH = __import__("pathlib").Path(args.ml_model)

    asyncio.run(main(args.output, args.fstart, args.fend, args.model, use_ml=args.ml))
