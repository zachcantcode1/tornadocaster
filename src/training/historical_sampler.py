"""
historical_sampler.py — Fetch historical HRRR runs and compute TC composite
features for training data construction.

For each target date/cycle, downloads only the fields needed for the TC
composite (byte-range fetches, same mechanism as the live forecast).
Returns a flat feature array per grid point plus labels derived from
SPC storm report proximity.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from src.ingestion.noaa_fetcher import NOAAIndexFetcher
from src.features.derived import build_first_pass_derived_fields
from src.visualization.plot_forecast import compute_tornado_composite

logger = logging.getLogger(__name__)

# HRRR S3 base
_HRRR_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

# Label radius: tornado within 40 km counts as a positive sample
_LABEL_RADIUS_KM = 40.0
_LABEL_RADIUS_DEG = _LABEL_RADIUS_KM / 111.0

# Label window must match the inference aggregation period (F001–F018 = 18 hours).
_LABEL_WINDOW_HOURS = 18

# HRRR native grid spacing (~3 km). Used to convert spatial scale → Gaussian sigma in pixels.
_GRID_KM = 3.0

# Fields to fetch from HRRR for feature extraction
HRRR_TRAINING_FIELDS = [
    ("cape_ml",      "CAPE",  "90-0 mb above ground"),
    ("cin_ml",       "CIN",   "90-0 mb above ground"),
    ("hlcy_3km",     "HLCY",  "3000-0 m above ground"),
    ("vwsh_0_6km",   "VWSH",  "6000-0 m above ground"),
    ("tmp_2m",       "TMP",   "2 m above ground"),
    ("dpt_2m",       "DPT",   "2 m above ground"),
    ("cape_surface", "CAPE",  "surface"),
    ("cape_mu",      "CAPE",  "180-0 mb above ground"),
    ("cin_surface",  "CIN",   "surface"),
    ("ugrd_10m",     "UGRD",  "10 m above ground"),
    ("vgrd_10m",     "VGRD",  "10 m above ground"),
    ("ugrd_850",     "UGRD",  "850 mb"),
    ("vgrd_850",     "VGRD",  "850 mb"),
]

# Feature columns — 18 base thermodynamic/kinematic + 10 spatial gradient + 3 temporal = 31
FEATURE_COLS = [
    # Base thermodynamic / kinematic fields
    "cape_ml", "cin_ml", "hlcy_3km", "vwsh_0_6km",
    "tmp_2m", "dpt_2m", "cape_surface", "cape_mu", "cin_surface",
    "ugrd_10m", "vgrd_10m", "bwd_850_sfc",
    "tc_cape_term", "tc_lcl_term", "tc_cin_term",
    "tc_srh_term", "tc_bwd_term", "tc_composite",
    # Spatial gradient features — detect drylines, fronts, cap break zones.
    # Gradient magnitude at 25 km and 50 km scales for the five fields that
    # change most sharply at convective boundaries.
    "cape_ml_grad_25km", "cape_ml_grad_50km",
    "hlcy_3km_grad_25km", "hlcy_3km_grad_50km",
    "cin_ml_grad_25km",
    "tmp_2m_grad_25km",
    "dpt_2m_grad_25km",
    # Temporal features — suppress false alarms by time-of-day and season.
    "hour_utc",    # valid hour 0–23
    "doy_sin",     # sin(2π·doy/365) — cyclical day-of-year encoding
    "doy_cos",     # cos(2π·doy/365)
]


def _grad_mag(arr2d: np.ndarray, scale_km: float) -> np.ndarray:
    """
    Gradient magnitude of a 2D field smoothed to *scale_km* spatial scale.
    Identifies sharp boundaries (drylines, fronts, outflow boundaries).
    """
    sigma = scale_km / _GRID_KM
    smoothed = gaussian_filter(arr2d.astype(np.float64), sigma=sigma)
    gy, gx = np.gradient(smoothed)
    return np.hypot(gx, gy).astype(np.float32)


def _hrrr_url(date: datetime, cycle: int, fhour: int = 1) -> str:
    ds = date.strftime("%Y%m%d")
    return (
        f"{_HRRR_BASE}/hrrr.{ds}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsfcf{fhour:02d}.grib2"
    )


async def _fetch_hrrr_fields(
    date: datetime, cycle: int, fhour: int = 1
) -> Optional[tuple[dict, np.ndarray, np.ndarray]]:
    """Fetch HRRR fields for a single run. Returns (fields, lat2d, lon2d) or None."""
    url = _hrrr_url(date, cycle, fhour)
    fetcher = NOAAIndexFetcher()
    try:
        idx_text = await fetcher.fetch_idx_file(f"{url}.idx")
        records  = fetcher.parse_idx_text(idx_text)
        specs    = [(n, v, l) for (n, v, l) in HRRR_TRAINING_FIELDS]
        fields   = await fetcher.fetch_named_fields(url, records, specs)
        if not fields:
            return None

        lat2d = lon2d = None
        for da in fields.values():
            if "latitude" in da.coords and "longitude" in da.coords:
                lat2d = da.coords["latitude"].values.astype(np.float32)
                lon2d = da.coords["longitude"].values.astype(np.float32)
                lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
                break
        if lat2d is None:
            return None
        return fields, lat2d, lon2d
    except Exception as exc:
        logger.debug("HRRR fetch failed %s %02dZ f%02d: %s", date.strftime("%Y%m%d"), cycle, fhour, exc)
        return None
    finally:
        await fetcher.close()


def _extract_features(
    fields: dict,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    valid_start: Optional[datetime] = None,
) -> Optional[np.ndarray]:
    """
    Compute per-grid-point feature matrix from fetched HRRR fields.
    Returns float32 array of shape (H*W, len(FEATURE_COLS)) or None.
    """
    def v(name, default=0.0):
        da = fields.get(name)
        if da is None:
            return np.full(lat2d.shape, default, dtype=np.float64)
        return np.asarray(da.values, dtype=np.float64)

    cape_ml    = v("cape_ml")
    cin_ml     = v("cin_ml")
    hlcy_3km   = v("hlcy_3km")
    vwsh       = v("vwsh_0_6km")
    tmp_2m     = v("tmp_2m")
    dpt_2m     = v("dpt_2m")
    cape_sfc   = v("cape_surface")
    cape_mu    = v("cape_mu")
    cin_sfc    = v("cin_surface")
    ugrd_10m   = v("ugrd_10m")
    vgrd_10m   = v("vgrd_10m")
    ugrd_850   = v("ugrd_850")
    vgrd_850   = v("vgrd_850")

    # Derived bulk shear proxy (850mb - 10m vector)
    bwd = np.hypot(ugrd_850 - ugrd_10m, vgrd_850 - vgrd_10m)

    # TC component terms
    lcl_m    = 122.0 * np.clip(tmp_2m - dpt_2m, 0.0, None)
    lcl_term = np.clip((2000.0 - lcl_m) / 1000.0, 0.0, 1.0)
    cin_term = np.where(cin_ml < -50.0, 0.0, np.clip((200.0 + cin_ml) / 150.0, 0.0, 1.0))
    bwd_term = np.clip(vwsh, 0.0, None) / 12.0
    srh_term = np.clip(hlcy_3km, 0.0, None) / 150.0

    derived = build_first_pass_derived_fields(fields)
    tc = compute_tornado_composite({**fields, **derived})
    if tc is None:
        tc = np.zeros_like(cape_ml, dtype=np.float32)

    # ── Spatial gradient features ────────────────────────────────────────────
    # These detect the sharp boundaries (drylines, fronts, outflow) where
    # convection actually initiates, preventing the model from painting the
    # entire warm sector as high-probability.
    cape_grad_25  = _grad_mag(cape_ml,  25.0)
    cape_grad_50  = _grad_mag(cape_ml,  50.0)
    hlcy_grad_25  = _grad_mag(hlcy_3km, 25.0)
    hlcy_grad_50  = _grad_mag(hlcy_3km, 50.0)
    cin_grad_25   = _grad_mag(cin_ml,   25.0)
    tmp_grad_25   = _grad_mag(tmp_2m,   25.0)
    dpt_grad_25   = _grad_mag(dpt_2m,   25.0)

    # ── Temporal features ────────────────────────────────────────────────────
    if valid_start is not None:
        hour_utc = float(valid_start.hour)
        doy      = float(valid_start.timetuple().tm_yday)
    else:
        hour_utc = 12.0
        doy      = 150.0
    doy_sin = np.sin(2.0 * np.pi * doy / 365.0)
    doy_cos = np.cos(2.0 * np.pi * doy / 365.0)

    H, W = lat2d.shape if lat2d.ndim == 2 else (lat2d.shape[0], lon2d.shape[0])
    N = H * W

    cols = np.stack([
        cape_ml, cin_ml, hlcy_3km, vwsh,
        tmp_2m, dpt_2m, cape_sfc, cape_mu, cin_sfc,
        ugrd_10m, vgrd_10m, bwd,
        np.clip(cape_ml, 0, None) / 1500.0,  # tc_cape_term
        lcl_term, cin_term, srh_term, bwd_term,
        tc.astype(np.float64),
        # gradients
        cape_grad_25.astype(np.float64),
        cape_grad_50.astype(np.float64),
        hlcy_grad_25.astype(np.float64),
        hlcy_grad_50.astype(np.float64),
        cin_grad_25.astype(np.float64),
        tmp_grad_25.astype(np.float64),
        dpt_grad_25.astype(np.float64),
        # temporal (broadcast scalar to grid)
        np.full((H, W), hour_utc),
        np.full((H, W), doy_sin),
        np.full((H, W), doy_cos),
    ], axis=-1)  # (H, W, 31)

    return cols.reshape(N, len(FEATURE_COLS)).astype(np.float32)


def _make_labels(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    reports: pd.DataFrame,
    valid_start: datetime,
    valid_end: datetime,
) -> np.ndarray:
    """
    Build binary label array (H*W,) — 1 if a tornado was reported within
    _LABEL_RADIUS_KM of the grid point during [valid_start, valid_end].
    """
    window = reports[
        (reports["utc_time"] >= valid_start) &
        (reports["utc_time"] < valid_end)
    ]

    if lat2d.ndim == 2:
        flat_lat = lat2d.ravel()
        flat_lon = lon2d.ravel()
    else:
        grid_lon, grid_lat = np.meshgrid(lon2d, lat2d)
        flat_lat = grid_lat.ravel()
        flat_lon = grid_lon.ravel()

    labels = np.zeros(len(flat_lat), dtype=np.int8)
    if len(window) == 0:
        return labels

    tor_lats = window["slat"].values
    tor_lons = window["slon"].values

    for tlat, tlon in zip(tor_lats, tor_lons):
        dlat = flat_lat - tlat
        dlon = (flat_lon - tlon) * np.cos(np.radians(tlat))
        dist_km = np.hypot(dlat, dlon) * 111.0
        labels[dist_km <= _LABEL_RADIUS_KM] = 1

    return labels


async def sample_day(
    date: datetime,
    reports: pd.DataFrame,
    cycles: list[int] = (0, 6, 12, 18),
    fhour: int = 1,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Fetch all HRRR cycles for *date*, compute features and labels.
    Returns (features, labels) arrays concatenated across cycles, or None.
    """
    async def _fetch_one_cycle(cycle: int):
        result = await _fetch_hrrr_fields(date, cycle, fhour)
        if result is None:
            return None
        fields, lat2d, lon2d = result
        valid_start = date.replace(hour=cycle, minute=0, second=0, microsecond=0,
                                   tzinfo=timezone.utc) + timedelta(hours=fhour)
        valid_end   = valid_start + timedelta(hours=_LABEL_WINDOW_HOURS)
        feats = _extract_features(fields, lat2d, lon2d, valid_start=valid_start)
        if feats is None:
            return None
        labels = _make_labels(lat2d, lon2d, reports, valid_start, valid_end)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n_pos   = len(pos_idx)
        n_neg   = min(len(neg_idx), max(n_pos * 5, 8000))
        rng     = np.random.default_rng(seed=int(date.timestamp()) + cycle)
        neg_sampled = rng.choice(neg_idx, size=n_neg, replace=False)
        keep = np.concatenate([pos_idx, neg_sampled])
        logger.info("  cycle %02dZ: %d pos / %d neg samples", cycle, n_pos, n_neg)
        return feats[keep], labels[keep]

    cycle_results = await asyncio.gather(*[_fetch_one_cycle(c) for c in cycles])

    all_feats  = [r[0] for r in cycle_results if r is not None]
    all_labels = [r[1] for r in cycle_results if r is not None]

    if not all_feats:
        return None
    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)
