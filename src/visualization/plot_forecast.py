"""Forecast visualization for NADOCast probability grids."""
from __future__ import annotations

import logging
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cartopy.mpl.path import shapely_to_path
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, PathPatch
import numpy as np
from shapely.geometry.base import BaseGeometry

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
_SPC_PROBABILITY_LEGENDS = {
    "tornado": [
        ("60%", "#8ABEF7", "#2A5EA8"),
        ("45%", "#C77DFF", "#7A2BCC"),
        ("30%", "#FF66FF", "#CC00CC"),
        ("15%", "#FF7B7B", "#D7191C"),
        ("10%", "#FFE481", "#FD8A2B"),
        ("5%", "#BD998A", "#7F3F27"),
        ("2%", "#79BA7A", "#1A731D"),
    ],
    "hail": [
        ("60%", "#8ABEF7", "#2A5EA8"),
        ("45%", "#C77DFF", "#7A2BCC"),
        ("30%", "#FF66FF", "#CC00CC"),
        ("15%", "#FF7B7B", "#D7191C"),
        ("5%", "#BD998A", "#7F3F27"),
    ],
    "wind": [
        ("60%", "#8ABEF7", "#2A5EA8"),
        ("45%", "#C77DFF", "#7A2BCC"),
        ("30%", "#FF66FF", "#CC00CC"),
        ("15%", "#FF7B7B", "#D7191C"),
        ("5%", "#BD998A", "#7F3F27"),
    ],
    "cat": [
        ("HIGH", "#FF66FF", "#CC00CC"),
        ("MDT", "#FF7B7B", "#D7191C"),
        ("ENH", "#FFA366", "#FF6600"),
        ("SLGT", "#FFE066", "#DDAA00"),
        ("MRGL", "#66A366", "#005500"),
        ("TSTM", "#C1E9C1", "#55BB55"),
    ],
}


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


def plot_conus_spc_outlook(
    outlook: object,
    title: str = "SPC Day 1 Outlook",
    subtitle: str = "",
    output_path: str = "forecast.png",
    dpi: int = 150,
    map_style: str = "dark",
) -> str:
    """Render SPC outlook polygons on the same CONUS base map."""
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

    _draw_spc_outlook(ax, outlook, data_crs, fill_alpha=0.58)

    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=style["state"], linewidth=0.65, zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=style["border"], linewidth=0.9, zorder=7)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=style["coast"], linewidth=0.75, zorder=7)

    _draw_spc_legend(fig, outlook, style)

    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=6, color=style["title"])
    ax.set_title(subtitle, fontsize=9, fontweight="normal", loc="right", pad=6, color=style["subtitle"])

    fig.patch.set_facecolor(style["figure"])
    ax.set_facecolor(style["water"])

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved SPC outlook plot to %s", output_path)
    return output_path


def _draw_spc_outlook(
    ax: plt.Axes,
    outlook: object,
    data_crs: ccrs.CRS,
    fill_alpha: float = 0.30,
) -> None:
    """Draw SPC outlook polygons."""
    for polygon in getattr(outlook, "polygons", ()):
        geometry = getattr(polygon, "geometry", None)
        if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
            continue
        if _is_spc_intensity_polygon(polygon):
            label = str(getattr(polygon, "label", ""))
            ax.add_geometries(
                [geometry],
                crs=data_crs,
                facecolor=(0.0, 0.0, 0.0, 0.0),
                edgecolor="#111111",
                linewidth=0.85,
                zorder=5.6,
            )
            if label.upper() == "CIG1":
                _draw_dashed_intensity(ax, geometry, data_crs)
            else:
                ax.add_geometries(
                    [geometry],
                    crs=data_crs,
                    facecolor=(0.0, 0.0, 0.0, 0.0),
                    edgecolor="#111111",
                    linewidth=0.0,
                    hatch=_spc_intensity_hatch(label),
                    zorder=5.5,
                )
            continue
        ax.add_geometries(
            [geometry],
            crs=data_crs,
            facecolor=_with_alpha(getattr(polygon, "fill", "#000000"), fill_alpha),
            edgecolor=getattr(polygon, "stroke", "#000000"),
            linewidth=1.15,
            zorder=5,
        )


def _draw_dashed_intensity(ax: plt.Axes, geometry: BaseGeometry, data_crs: ccrs.CRS) -> None:
    minx, miny, maxx, maxy = geometry.bounds
    span = max(maxx - minx, maxy - miny)
    spacing = span / 9.0 if span > 0 else 1.0
    lines = []
    start = minx - span
    stop = maxx + span
    for offset in np.arange(start, stop, spacing):
        lines.append([(offset, miny - span * 0.25), (offset + span * 0.75, maxy + span * 0.25)])

    path = shapely_to_path(geometry)
    transform = data_crs._as_mpl_transform(ax)
    clip_patch = PathPatch(path, transform=transform)
    collection = LineCollection(
        lines,
        colors="#111111",
        linewidths=1.35,
        linestyles=(0, (3.5, 5.0)),
        transform=transform,
        zorder=5.7,
    )
    collection.set_clip_path(clip_patch)
    ax.add_collection(collection)


def _draw_spc_legend(fig: plt.Figure, outlook: object, style: dict[str, str]) -> None:
    del style
    product = str(getattr(outlook, "product", "cat"))
    probability_entries = _spc_probability_legend_entries(product)
    has_intensity = _spc_product_has_intensity(product)

    legend_ax = fig.add_axes([0.025, 0.690, 0.135, 0.245])
    legend_ax.set_axis_off()
    legend_ax.add_patch(
        FancyBboxPatch(
            (0.018, -0.018),
            1.0,
            1.0,
            boxstyle="round,pad=0.024,rounding_size=0.085",
            facecolor=(0.0, 0.0, 0.0, 0.28),
            edgecolor="none",
            clip_on=False,
            zorder=0,
        )
    )
    legend_ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.024,rounding_size=0.085",
            facecolor="#fbfbf7",
            edgecolor="#d9ddd8",
            linewidth=1.1,
            clip_on=False,
            zorder=1,
        )
    )
    legend_ax.text(
        0.5,
        0.925,
        getattr(outlook, "product_label", "Outlook"),
        fontsize=8.8,
        fontweight="bold",
        ha="center",
        va="center",
        color="#151a1f",
        zorder=2,
    )
    legend_ax.plot([0.09, 0.91], [0.865, 0.865], color="#d7dcd5", linewidth=0.9, zorder=2)
    legend_ax.text(
        0.08,
        0.805,
        "Probability",
        fontsize=7.5,
        fontweight="bold",
        va="top",
        color="#151a1f",
        zorder=2,
    )

    for idx, (label, fill, stroke) in enumerate(probability_entries):
        y = 0.715 - idx * 0.082
        legend_ax.add_patch(
            FancyBboxPatch(
                (0.09, y - 0.030),
                0.23,
                0.060,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=fill,
                edgecolor=stroke,
                linewidth=0.8,
                zorder=2,
            )
        )
        legend_ax.text(
            0.205,
            y,
            label,
            fontsize=6.8,
            fontweight="bold",
            ha="center",
            va="center",
            color="#111111",
            zorder=3,
        )

    if has_intensity:
        legend_ax.text(
            0.67,
            0.645,
            "Intensity",
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
            color="#151a1f",
            zorder=2,
        )
        for idx, label in enumerate(("CIG3", "CIG2", "CIG1")):
            y = 0.525 - idx * 0.142
            sample = FancyBboxPatch(
                (0.55, y - 0.048),
                0.22,
                0.096,
                boxstyle="round,pad=0.004,rounding_size=0.010",
                facecolor="#ffffff",
                edgecolor="#111111",
                linewidth=0.8,
                hatch="" if label == "CIG1" else _spc_intensity_hatch(label),
                zorder=2,
            )
            legend_ax.add_patch(sample)
            if label == "CIG1":
                _draw_legend_dashed_intensity(legend_ax, sample, 0.55, y - 0.048, 0.22, 0.096)
            legend_ax.text(
                0.84,
                y,
                label.replace("CIG", ""),
                fontsize=9,
                fontweight="bold",
                va="center",
                color="#111111",
                zorder=3,
            )


def _draw_legend_dashed_intensity(
    legend_ax: plt.Axes,
    clip_patch: FancyBboxPatch,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    lines = []
    for offset in np.linspace(x - width * 0.25, x + width * 0.95, 5):
        lines.append([(offset, y - height * 0.05), (offset + width * 0.42, y + height * 1.05)])
    collection = LineCollection(
        lines,
        colors="#111111",
        linewidths=0.75,
        linestyles=(0, (2.0, 2.8)),
        zorder=3,
    )
    collection.set_clip_path(clip_patch)
    legend_ax.add_collection(collection)


def _draw_nadocast_legend(fig: plt.Figure, style: dict[str, str], max_probability: float) -> None:
    """Draw a rounded full-scale NADOCast probability legend."""
    del max_probability, style
    legend_ax = fig.add_axes([0.025, 0.690, 0.105, 0.245])
    legend_ax.set_axis_off()
    legend_ax.add_patch(
        FancyBboxPatch(
            (0.018, -0.018),
            1.0,
            1.0,
            boxstyle="round,pad=0.024,rounding_size=0.085",
            facecolor=(0.0, 0.0, 0.0, 0.28),
            edgecolor="none",
            clip_on=False,
            zorder=0,
        )
    )
    legend_ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.024,rounding_size=0.085",
            facecolor="#fbfbf7",
            edgecolor="#d9ddd8",
            linewidth=1.1,
            clip_on=False,
            zorder=1,
        )
    )
    legend_ax.text(
        0.5,
        0.925,
        "NADOCast",
        fontsize=8.8,
        fontweight="bold",
        ha="center",
        va="center",
        color="#151a1f",
        zorder=2,
    )
    legend_ax.plot([0.09, 0.91], [0.865, 0.865], color="#d7dcd5", linewidth=0.9, zorder=2)
    legend_ax.text(
        0.5,
        0.805,
        "Probability",
        fontsize=7.5,
        fontweight="bold",
        va="top",
        ha="center",
        color="#151a1f",
        zorder=2,
    )

    for idx, (label, color) in enumerate(reversed(list(zip(_LEGEND_LABELS, _PROB_COLORS)))):
        y = 0.700 - idx * 0.073
        legend_ax.add_patch(
            FancyBboxPatch(
                (0.31, y - 0.024),
                0.38,
                0.048,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=color,
                edgecolor="#2b2f33",
                linewidth=0.7,
                zorder=2,
            )
        )
        legend_ax.text(
            0.50,
            y,
            label,
            fontsize=6.6,
            fontweight="bold",
            ha="center",
            va="center",
            color="#111111",
            zorder=3,
        )


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


def _with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    red, green, blue, _ = to_rgba(color)
    return red, green, blue, alpha


def _is_spc_intensity_polygon(polygon: object) -> bool:
    return str(getattr(polygon, "label", "")).upper().startswith("CIG")


def _spc_intensity_hatch(label: str) -> str:
    return {
        "CIG1": "///",
        "CIG2": "\\\\\\",
        "CIG3": "xxx",
    }.get(label.upper(), "///")


def _spc_legend_label(label: str) -> str:
    try:
        value = float(label)
    except ValueError:
        return label
    return f"{value:.0%}"


def _spc_probability_legend_entries(product: str) -> list[tuple[str, str, str]]:
    return _SPC_PROBABILITY_LEGENDS.get(product, _SPC_PROBABILITY_LEGENDS["cat"])


def _spc_product_has_intensity(product: str) -> bool:
    return product in {"tornado", "hail", "wind"}
