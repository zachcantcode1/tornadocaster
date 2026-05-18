"""
storm_reports.py — Download and parse SPC tornado reports.

Reports come from the SPC warning coordination page:
  https://www.spc.noaa.gov/wcm/data/YYYY_torn.csv

Columns of interest:
  date (YYYY-MM-DD), time (HHMM local), tz (timezone offset from UTC),
  slat/slon (start lat/lon), mag (EF scale, -9 = unknown)
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_SPC_BASE = "https://www.spc.noaa.gov/wcm/data"
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "storm_reports"


def _report_url(year: int) -> str:
    return f"{_SPC_BASE}/{year}_torn.csv"


def _cache_path(year: int) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{year}_torn.csv"


def download_year(year: int, force: bool = False) -> pd.DataFrame:
    """Download (and cache) the SPC tornado CSV for *year*."""
    path = _cache_path(year)
    if path.exists() and not force:
        logger.info("Using cached storm reports: %s", path)
        raw = path.read_text(encoding="latin-1")
    else:
        url = _report_url(year)
        logger.info("Downloading SPC storm reports: %s", url)
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        raw = resp.text
        path.write_text(raw, encoding="utf-8")

    df = pd.read_csv(
        io.StringIO(raw),
        header=0,
        dtype=str,
        on_bad_lines="skip",
    )
    df.columns = [c.strip().lower() for c in df.columns]
    return _clean(df, year)


def _clean(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Parse dates/times and drop unusable rows."""
    # Normalise column names across SPC format versions
    rename = {}
    for col in df.columns:
        if col in ("slat", "lat"):   rename[col] = "slat"
        if col in ("slon", "lon"):   rename[col] = "slon"
        if col in ("mag", "f_scale"): rename[col] = "mag"
    df = df.rename(columns=rename)

    needed = {"date", "time", "tz", "slat", "slon"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"SPC CSV for {year} is missing columns: {missing}")

    df = df.copy()
    df["slat"] = pd.to_numeric(df["slat"], errors="coerce")
    df["slon"] = pd.to_numeric(df["slon"], errors="coerce")
    df["tz"]   = pd.to_numeric(df["tz"],   errors="coerce").fillna(0)
    df["mag"]  = pd.to_numeric(df.get("mag", pd.Series([-9] * len(df))), errors="coerce").fillna(-9)

    # Drop rows with invalid coordinates
    df = df.dropna(subset=["slat", "slon"])
    df = df[(df["slat"].abs() > 0) & (df["slon"] != 0)]

    # SPC timezone code → hours offset from UTC
    _TZ_OFFSET = {0: 0, 3: 6, 4: 5, 6: 5, 7: 4, 9: 0}  # 3=CST, 4=CDT, 6=EST, 7=EDT, 9=GMT

    # Build UTC datetime
    def _to_utc(row) -> Optional[datetime]:
        try:
            t = str(row["time"]).strip()
            d = str(row["date"]).strip()
            # Handle both HH:MM:SS and HHMM formats
            if ":" in t:
                dt_local = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
            else:
                dt_local = datetime.strptime(f"{d} {t.zfill(4)}", "%Y-%m-%d %H%M")
            offset = _TZ_OFFSET.get(int(float(row["tz"])), 0)
            return dt_local.replace(tzinfo=timezone.utc) - timedelta(hours=-offset)
        except Exception:
            return None

    df["utc_time"] = df.apply(_to_utc, axis=1)
    df = df.dropna(subset=["utc_time"])
    df = df.sort_values("utc_time").reset_index(drop=True)
    logger.info("  Year %d: %d tornado reports after cleaning", year, len(df))
    return df


def load_reports(years: list[int], force: bool = False) -> pd.DataFrame:
    """Load multiple years of SPC tornado reports into a single DataFrame."""
    frames = []
    for yr in years:
        try:
            frames.append(download_year(yr, force=force))
        except Exception as exc:
            logger.warning("Could not load %d storm reports: %s", yr, exc)
    if not frames:
        raise RuntimeError("No storm report data loaded.")
    return pd.concat(frames, ignore_index=True)


def tornado_days(reports: pd.DataFrame) -> list[datetime]:
    """Return sorted list of unique UTC dates that had tornado reports."""
    utc = pd.to_datetime(reports["utc_time"], utc=True)
    dates = utc.dt.normalize().unique()
    return sorted(pd.Timestamp(d).to_pydatetime().replace(tzinfo=timezone.utc) for d in dates)
