"""
build_dataset.py — Orchestrate historical HRRR sampling into a Parquet dataset.

Usage:
    python -m src.training.build_dataset --years 2021 2022 2023 --out data/training

For each tornado day in the requested years, fetches HRRR runs and writes
one Parquet shard per day to --out.  Safe to re-run — already-written shards
are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.storm_reports import load_reports, tornado_days
from src.training.historical_sampler import sample_day, FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# For each tornado day we also sample this many random "quiet" days to
# ensure the model sees plenty of null-environment examples.
_NULL_DAYS_PER_TORNADO_DAY = 0.25   # 1 null day per 4 tornado days


async def process_day(
    date: datetime,
    reports: pd.DataFrame,
    out_dir: Path,
    label: str,
) -> bool:
    """Process one day, write shard, return True if successful."""
    tag  = f"{label}_{date.strftime('%Y%m%d')}"
    path = out_dir / f"{tag}.parquet"
    if path.exists():
        logger.info("Skipping %s (already exists)", tag)
        return True

    logger.info("Processing %s ...", tag)
    try:
        result = await sample_day(date, reports)
    except Exception as exc:
        logger.error("EXCEPTION in sample_day for %s: %s", tag, exc, exc_info=True)
        return False

    if result is None:
        logger.warning("No data for %s", tag)
        return False

    feats, labels = result
    try:
        df = pd.DataFrame(feats, columns=FEATURE_COLS)
        df["label"] = labels.astype(np.int8)
        df["date"]  = date.strftime("%Y-%m-%d")
        df.to_parquet(path, index=False, compression="snappy")
    except Exception as exc:
        logger.error("EXCEPTION writing shard for %s: %s", tag, exc, exc_info=True)
        return False

    n_pos = int(labels.sum())
    logger.info("  Wrote %s - %d rows (%d positive)", path.name, len(df), n_pos)
    return True


async def main(years: list[int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SPC storm reports for %s (EF1+ only) …", years)
    reports = load_reports(years, min_ef=1)
    reports["utc_time"] = pd.to_datetime(reports["utc_time"], utc=True)

    t_days = tornado_days(reports)
    logger.info("Found %d tornado days across %d years", len(t_days), len(years))

    # Select quiet days (random days not in the tornado list)
    all_dates_set = {d.date() for d in t_days}
    year_start = datetime(min(years), 1, 1, tzinfo=timezone.utc)
    year_end   = datetime(max(years) + 1, 1, 1, tzinfo=timezone.utc)
    all_days   = []
    cur = year_start
    while cur < year_end:
        if cur.date() not in all_dates_set:
            all_days.append(cur)
        cur += timedelta(days=1)

    n_null = max(1, int(len(t_days) * _NULL_DAYS_PER_TORNADO_DAY))
    rng = random.Random(42)
    null_days = rng.sample(all_days, min(n_null, len(all_days)))
    logger.info("Adding %d quiet (null) days for negative examples", len(null_days))

    # Process tornado days
    sem = asyncio.Semaphore(8)   # 8 concurrent days — S3 rate limit safe
    async def bounded(date, rpts, label):
        async with sem:
            return await process_day(date, rpts, out_dir, label)

    tornado_tasks = [bounded(d, reports, "tornado") for d in t_days]
    null_tasks    = [bounded(d, reports, "null")    for d in null_days]

    results = await asyncio.gather(*(tornado_tasks + null_tasks), return_exceptions=True)
    ok  = sum(1 for r in results if r is True)
    err = sum(1 for r in results if r is not True)
    logger.info("Done — %d shards written, %d skipped/failed", ok, err)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument("--out",   type=Path,   default=Path("data/training"))
    args = parser.parse_args()
    asyncio.run(main(args.years, args.out))
