"""
climatology.py — Pre-compute a monthly EF1+ tornado frequency grid from
SPC reports (1990-2023) on a regular 0.25-degree lat/lon grid.

The resulting grid encodes "how tornado-prone is this location in this month"
and is used as a feature to suppress false alarms in non-tornado-prone regions
(Northeast, Pacific Northwest, etc.).

Usage:
    python -m src.training.climatology   # builds and saves the grid
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.training.storm_reports import load_reports

logger = logging.getLogger(__name__)

# Output grid — 0.25-degree CONUS coverage
_LAT_MIN, _LAT_MAX = 20.0, 55.0
_LON_MIN, _LON_MAX = -130.0, -60.0
_GRID_DEG = 0.25

_CLIMO_PATH = Path(__file__).parent.parent.parent / "data" / "climatology" / "climo_ef1plus.npz"

# Smooth the raw counts with a Gaussian (sigma in grid cells).
# 300 km / 25 km per cell = 12 cells.
_SMOOTH_SIGMA = 12

_CLIMO_YEARS = list(range(1990, 2024))


def _build_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download SPC reports and compute monthly frequency grid.

    Returns
    -------
    freq   : (12, nlat, nlon) float32 — smoothed tornado count per cell per month
    lats   : (nlat,) center latitudes
    lons   : (nlon,) center longitudes
    """
    from scipy.ndimage import gaussian_filter

    lats = np.arange(_LAT_MIN + _GRID_DEG / 2, _LAT_MAX, _GRID_DEG, dtype=np.float32)
    lons = np.arange(_LON_MIN + _GRID_DEG / 2, _LON_MAX, _GRID_DEG, dtype=np.float32)
    nlat, nlon = len(lats), len(lons)

    counts = np.zeros((12, nlat, nlon), dtype=np.float32)

    logger.info("Downloading SPC reports for climatology (%d-%d, EF1+ only)...",
                _CLIMO_YEARS[0], _CLIMO_YEARS[-1])

    # Load in chunks to avoid hammering SPC servers
    chunk = 5
    all_reports = []
    for start in range(0, len(_CLIMO_YEARS), chunk):
        yrs = _CLIMO_YEARS[start:start + chunk]
        try:
            df = load_reports(yrs, min_ef=1)
            all_reports.append(df)
            logger.info("  Loaded %d-%d: %d EF1+ reports", yrs[0], yrs[-1], len(df))
        except Exception as exc:
            logger.warning("  Failed chunk %d-%d: %s", yrs[0], yrs[-1], exc)

    if not all_reports:
        raise RuntimeError("No climatology data loaded.")

    import pandas as pd
    reports = pd.concat(all_reports, ignore_index=True)
    reports["month"] = pd.to_datetime(reports["utc_time"]).dt.month

    for _, row in reports.iterrows():
        lat, lon, month = float(row["slat"]), float(row["slon"]), int(row["month"])
        if not (_LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX):
            continue
        i = int((lat - _LAT_MIN) / _GRID_DEG)
        j = int((lon - _LON_MIN) / _GRID_DEG)
        i = min(i, nlat - 1)
        j = min(j, nlon - 1)
        counts[month - 1, i, j] += 1.0

    # Normalize by number of years so units are "events per year per cell"
    n_years = len(_CLIMO_YEARS)
    counts /= n_years

    # Smooth spatially
    freq = np.stack([
        gaussian_filter(counts[m], sigma=_SMOOTH_SIGMA).astype(np.float32)
        for m in range(12)
    ])

    logger.info("Climatology grid: %dx%d, max=%.4f events/yr/cell",
                nlat, nlon, float(freq.max()))
    return freq, lats, lons


def build_and_save() -> None:
    _CLIMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    freq, lats, lons = _build_grid()
    np.savez_compressed(_CLIMO_PATH, freq=freq, lats=lats, lons=lons)
    logger.info("Saved climatology to %s", _CLIMO_PATH)


def load_climo() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-computed climatology. Builds it if not present."""
    if not _CLIMO_PATH.exists():
        logger.info("Climatology not found — building now...")
        build_and_save()
    data = np.load(_CLIMO_PATH)
    return data["freq"], data["lats"], data["lons"]


def get_climo_freq(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    month: int,
) -> np.ndarray:
    """
    Interpolate climatological EF1+ tornado frequency to an arbitrary grid.

    Parameters
    ----------
    lat2d, lon2d : 2D arrays of shape (H, W)
    month        : 1-12

    Returns
    -------
    float32 array of shape (H, W)
    """
    freq, lats, lons = load_climo()

    monthly = freq[month - 1]  # (nlat, nlon)

    # Nearest-neighbor lookup (fast, sufficient for a smooth field)
    lat_idx = np.clip(
        ((lat2d - _LAT_MIN) / _GRID_DEG).astype(int), 0, len(lats) - 1
    )
    lon_idx = np.clip(
        ((lon2d - _LON_MIN) / _GRID_DEG).astype(int), 0, len(lons) - 1
    )
    return monthly[lat_idx, lon_idx].astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_and_save()
