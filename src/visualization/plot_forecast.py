"""Forecast visualization for NADOCast probability grids."""
from __future__ import annotations

import logging
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_PROB_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60, 1.0]
_PROB_COLORS = [
    "#d0d0d0",  # 1-2%
    "#008000",  # 2-3%
    "#32cd32",  # 3-5%
    "#8b4513",  # 5-10%
    "#ffd400",  # 10-15%
    "#ff2020",  # 15-30%
    "#ff33ff",  # 30-45%
    "#9b35e6",  # 45-60%
    "#1f4e79",  # 60%+
]
_LEGEND_LABELS = ["1%", "2%", "3%", "5%", "10%", "15%", "30%", "45%", "60%"]


def plot_conus_forecast(
    lat: np.ndarray,
    lon: np.ndarray,
    stp: np.ndarray,
    title: str = "NADOCast Forecast",
    subtitle: str = "",
    output_path: str = "forecast.png",
    dpi: int = 150,
    mxuphl: Optional[np.ndarray] = None,
    prob_mode: bool = True,
    report_points: list[tuple[float, float]] | None = None,
    map_style: str = "dark",
) -> str:
    """Render a CONUS probability map.

    The parameter names keep backward compatibility with the old CLI, but the
    data is expected to be a probability fraction in the range `0.0-1.0`.
    """
    del prob_mode  # Probability mode is now the only supported rendering mode.
    proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39)
    data_crs = ccrs.PlateCarree()
    style = _style_tokens(map_style)

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-125, -66, 22, 50], crs=data_crs)

    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=style["water"], zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=style["land"], zorder=0)
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor=style["water"],
        edgecolor=style["lake_edge"],
        linewidth=0.35,
        zorder=1,
    )

    probability = np.clip(np.asarray(stp, dtype=np.float64), 0.0, 1.0)

    ax.contourf(
        lon,
        lat,
        probability,
        levels=_PROB_LEVELS,
        colors=_style_probability_colors(style),
        transform=data_crs,
        zorder=2,
    )
    ax.contour(
        lon,
        lat,
        probability,
        levels=_PROB_LEVELS[:-1],
        colors=[style["threat_line"]],
        linewidths=[0.55],
        transform=data_crs,
        zorder=3,
    )

    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=style["state"], linewidth=0.65, zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=style["border"], linewidth=0.9, zorder=7)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=style["coast"], linewidth=0.75, zorder=7)

    if mxuphl is not None and mxuphl.shape == probability.shape:
        hatch_zone = np.where((mxuphl.astype(np.float64) >= 50.0) & (probability >= 0.05), 1.0, np.nan)
        ax.contourf(
            lon,
            lat,
            np.ma.masked_invalid(hatch_zone),
            levels=[0.5, 1.5],
            hatches=["//"],
            colors=["none"],
            transform=data_crs,
            zorder=4,
        )

    if report_points:
        report_lats = [p[0] for p in report_points]
        report_lons = [p[1] for p in report_points]
        ax.scatter(
            report_lons,
            report_lats,
            s=28,
            marker="o",
            facecolor="#ff2d2d",
            edgecolor="#111111",
            linewidth=0.55,
            transform=data_crs,
            zorder=6,
        )

    _draw_nadocast_legend(fig, style, float(np.nanmax(probability)))

    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=6, color=style["title"])
    ax.set_title(subtitle, fontsize=9, fontweight="normal", loc="right", pad=6, color=style["subtitle"])

    fig.patch.set_facecolor(style["figure"])
    ax.set_facecolor(style["water"])

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved forecast plot to %s", output_path)
    return output_path


def _draw_nadocast_legend(fig: plt.Figure, style: dict[str, str], max_probability: float) -> None:
    """Draw a compact NADOCast-style threshold swatch legend."""
    legend_ax = fig.add_axes([0.54, 0.055, 0.32, 0.055])
    legend_ax.set_axis_off()
    legend_ax.text(
        0.0,
        0.95,
        "Chance of a tornado within 25 miles of a point.",
        fontsize=8,
        va="bottom",
        color=style["legend_text"],
    )

    colors, labels = _legend_entries(max_probability)
    n = len(colors)
    for idx, color in enumerate(colors):
        x0 = idx / n
        legend_ax.add_patch(
            plt.Rectangle((x0, 0.28), 1 / n, 0.36, facecolor=color, edgecolor=style["legend_edge"], linewidth=0.5)
        )
    for idx, label in enumerate(labels):
        legend_ax.text((idx + 0.5) / n, 0.1, label, fontsize=7, ha="center", va="top", color=style["legend_text"])


def _legend_entries(max_probability: float) -> tuple[list[str], list[str]]:
    upper_idx = len(_PROB_COLORS) - 1
    for idx, threshold in enumerate(_PROB_LEVELS[1:]):
        if max_probability < threshold:
            upper_idx = idx
            break
    upper_idx = max(2, upper_idx)
    return _PROB_COLORS[: upper_idx + 1], _LEGEND_LABELS[: upper_idx + 1]


def _style_tokens(map_style: str) -> dict[str, str]:
    if map_style == "dark":
        return {
            "figure": "#0b1016",
            "land": "#151b22",
            "water": "#08111a",
            "lake_edge": "#304455",
            "state": "#75828e",
            "border": "#aeb8c1",
            "coast": "#aeb8c1",
            "threat_line": "#05080b",
            "title": "#eef4f8",
            "subtitle": "#b8c3cc",
            "legend_text": "#eef4f8",
            "legend_edge": "#0b1016",
            "name": "dark",
        }
    return {
        "figure": "#ffffff",
        "land": "#fafaf6",
        "water": "#dfe8ef",
        "lake_edge": "#8a969e",
        "state": "#5f666a",
        "border": "#303438",
        "coast": "#303438",
        "threat_line": "#333333",
        "title": "#111111",
        "subtitle": "#555555",
        "legend_text": "#111111",
        "legend_edge": "#000000",
        "name": "light",
    }


def _style_probability_colors(style: dict[str, str]) -> list[str | tuple[float, float, float, float]]:
    if style["name"] != "dark":
        return _PROB_COLORS
    return [
        (0.82, 0.82, 0.82, 0.46),  # 1-2%
        (0.00, 0.50, 0.00, 0.86),  # 2-3%
        (0.20, 0.80, 0.20, 0.86),  # 3-5%
        (0.55, 0.27, 0.07, 0.90),
        (1.00, 0.83, 0.00, 0.92),
        (1.00, 0.13, 0.13, 0.92),
        (1.00, 0.20, 1.00, 0.92),
        (0.61, 0.21, 0.90, 0.92),
        (0.12, 0.31, 0.47, 0.92),
    ]
