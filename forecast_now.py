"""
NADOCast-first forecast map generator.

This CLI fetches a published NADOCast GRIB2 probability grid, summarizes it,
and optionally renders a CONUS map. Tornado Caster's core job is now to make
NADOCast easier to inspect, customize, and contextualize.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date
from pathlib import Path

from src.analysis.probability import probability_summary
from src.sources.nadocast import NadocastRequest, NadocastSource
from src.visualization.plot_forecast import plot_conus_forecast

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}")


async def run(args: argparse.Namespace) -> None:
    source = NadocastSource()

    request = NadocastRequest(
        run_date=_parse_date(args.date),
        cycle=args.cycle,
        hazard=args.hazard,
        model_set=args.model_set,
        calibrated=args.calibrated,
        window=args.window,
        filename=args.filename,
    )

    if request.run_date is None or request.cycle is None:
        request = await source.find_latest(request, max_days=args.search_days)

    grid = await source.fetch_grid(request)
    summary = probability_summary(grid.probability)

    print(f"NADOCast source: {grid.url}")
    print(f"Run: {grid.run_label}")
    print(f"Variable: {grid.variable_name} ({grid.units})")
    print(f"Grid: {grid.probability.shape[0]} x {grid.probability.shape[1]}")
    print(
        "Probability summary: "
        f"max={summary.max_probability:.1%}, "
        f"mean={summary.mean_probability:.2%}, "
        f"cells >=2%={summary.cells_ge_2pct}, "
        f"cells >=5%={summary.cells_ge_5pct}, "
        f"cells >=10%={summary.cells_ge_10pct}"
    )

    if args.summary_only:
        return

    output = Path(args.output)
    title = f"NADOCast {grid.hazard_label}"
    subtitle = grid.run_label
    plot_conus_forecast(
        grid.latitude,
        grid.longitude,
        grid.probability,
        title=title,
        subtitle=subtitle,
        output_path=str(output),
        prob_mode=True,
        map_style=args.map_style,
    )
    print(f"Map saved to: {output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and plot NADOCast GRIB2 probability guidance."
    )
    parser.add_argument("--date", help="Run date as YYYYMMDD. Defaults to latest found.")
    parser.add_argument("--cycle", type=int, choices=(0, 12, 18), help="Run cycle hour.")
    parser.add_argument(
        "--hazard",
        default="tornado",
        help="NADOCast hazard token, e.g. tornado, sig_tornado, hail, wind, wind_adj.",
    )
    parser.add_argument(
        "--model-set",
        default="2022",
        choices=("2024", "2022"),
        help="Published NADOCast model family to prefer.",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Prefer the abs_calib grid when available.",
    )
    parser.add_argument("--window", help="Forecast window token such as f12-35.")
    parser.add_argument("--filename", help="Exact NADOCast GRIB2 filename to fetch.")
    parser.add_argument("--output", default="forecast.png", help="Output PNG path.")
    parser.add_argument(
        "--map-style",
        default="dark",
        choices=("light", "dark"),
        help="Basemap style preset.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Decode and summarize the GRIB2 without rendering a map.",
    )
    parser.add_argument(
        "--search-days",
        type=int,
        default=14,
        help="Days to search backward when date/cycle are omitted.",
    )
    return parser


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
