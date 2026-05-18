"""
Forecast visualization: tornado composite map on CONUS Lambert Conformal projection.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# TC composite index thresholds (physics mode, no ML model)
_TC_LEVELS = [1.0, 2.0, 3.0, 5.0, 8.0]
_TC_COLORS = [
    "#7dc57d",  # 1–2   marginal   light green
    "#f5f540",  # 2–3   slight     yellow
    "#e8a030",  # 3–5   enhanced   orange
    "#e03030",  # 5–8   moderate   red
    "#e030e0",  # 8+    high       magenta
]

# Calibrated probability levels — matches SPC/Nadocast outlook thresholds
_PROB_LEVELS = [0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60]
_PROB_COLORS = [
    "#008b00",  # 2–5%    dark green
    "#8b4500",  # 5–10%   brown
    "#ffff00",  # 10–15%  yellow
    "#ffa500",  # 15–30%  orange
    "#ff0000",  # 30–45%  red
    "#ff00ff",  # 45–60%  magenta
    "#912cee",  # 60%+    purple
]
_PROB_LABELS = ["2%", "5%", "10%", "15%", "30%", "45%", "60%"]


def compute_tornado_composite(fields: dict[str, xr.DataArray]) -> Optional[np.ndarray]:
    """
    RRFS Tornado Composite Index.

    Builds on the SPC STP formula but uses RRFS-native fields when available:
      - LCL height: direct from model (HGT:level of adiabatic condensation from sfc)
        rather than the T2m-Td2m approximation used for HRRR.
      - MXUPHL gate: max updraft helicity > threshold confirms rotating updraft.
      - Fallback: standard STP formula when RRFS-specific fields are absent (HRRR mode).

    Formula:
        TC = (MLCAPE/1500) * lcl_term * cin_term * (SRH/150) * bwd_term * uh_gate

    Returns float32 array clipped to [0, 10], or None if required fields are missing.
    """
    cape_ml = fields.get("cape_ml")
    cin_ml  = fields.get("cin_ml")
    hlcy    = fields.get("hlcy_3km")

    if cape_ml is None or cin_ml is None or hlcy is None:
        logger.warning("compute_tornado_composite: missing required field(s) — returning None")
        return None

    cape_v = np.asarray(cape_ml.values, dtype=np.float64)
    cin_v  = np.asarray(cin_ml.values,  dtype=np.float64)
    hlcy_v = np.asarray(hlcy.values,    dtype=np.float64)

    # ── LCL height term ─────────────────────────────────────────────────────
    # Prefer direct model output; fall back to Bolton (1980) T2m-Td2m estimate.
    hgt_lcl = fields.get("hgt_lcl")
    if hgt_lcl is not None:
        lcl_m = np.asarray(hgt_lcl.values, dtype=np.float64)
        lcl_m = np.clip(lcl_m, 0.0, None)
        logger.debug("Using model-native LCL height field.")
    else:
        t2m  = fields.get("tmp_2m")
        td2m = fields.get("dpt_2m")
        if t2m is not None and td2m is not None:
            lcl_m = 122.0 * np.clip(
                np.asarray(t2m.values,  dtype=np.float64) -
                np.asarray(td2m.values, dtype=np.float64),
                0.0, None,
            )
        else:
            lcl_m = np.zeros_like(cape_v)

    lcl_term = np.clip((2000.0 - lcl_m) / 1000.0, 0.0, 1.0)

    # ── CIN gating ──────────────────────────────────────────────────────────
    cin_term = np.where(cin_v < -50.0, 0.0, np.clip((200.0 + cin_v) / 150.0, 0.0, 1.0))

    # ── BWD term ─────────────────────────────────────────────────────────────
    # RRFS hi file has no direct 0-6km shear field.
    # Use VWSH:6000-0 m (HRRR) when present; otherwise derive a low-level shear
    # proxy from 10m vs PBL winds and cap at 1.0 (conservative / max assumption).
    bwd = fields.get("vwsh_0_6km")
    if bwd is not None:
        bwd_v = np.asarray(bwd.values, dtype=np.float64)
        bwd_term = np.clip(bwd_v, 0.0, None) / 12.0
    else:
        ugrd_lo = fields.get("ugrd_10m")
        vgrd_lo = fields.get("vgrd_10m")
        _upbl = fields.get("ugrd_pbl")
        ugrd_hi = _upbl if _upbl is not None else fields.get("ugrd_80m")
        _vpbl = fields.get("vgrd_pbl")
        vgrd_hi = _vpbl if _vpbl is not None else fields.get("vgrd_80m")
        if (ugrd_lo is not None and vgrd_lo is not None and
                ugrd_hi is not None and vgrd_hi is not None):
            du = np.asarray(ugrd_hi.values, dtype=np.float64) - np.asarray(ugrd_lo.values, dtype=np.float64)
            dv = np.asarray(vgrd_hi.values, dtype=np.float64) - np.asarray(vgrd_lo.values, dtype=np.float64)
            # Scale the shallow shear to approximately 0-6km equivalent.
            # 0-1km shear is typically ~40-60% of 0-6km BWD over the Plains.
            bwd_term = np.clip(np.hypot(du, dv) / 7.0, 0.0, 1.0)
        else:
            # No wind data at all — assume climatological-average BWD (max term = 1.0).
            bwd_term = np.ones_like(cape_v)

    # ── Max updraft helicity gate (RRFS only) ────────────────────────────────
    # MXUPHL > 25 J/kg indicates a rotating updraft resolved by the model.
    # Blend: below 10 J/kg no enhancement; 10-100 J/kg ramps from 1.0 → 1.5.
    mxuphl = fields.get("mxuphl_03km")
    if mxuphl is not None:
        uh_v = np.clip(np.asarray(mxuphl.values, dtype=np.float64), 0.0, None)
        uh_gate = np.clip(1.0 + 0.5 * (uh_v - 10.0) / 90.0, 1.0, 1.5)
    else:
        uh_gate = np.ones_like(cape_v)

    composite = (
        np.clip(cape_v, 0.0, None) / 1500.0
        * lcl_term
        * cin_term
        * np.clip(hlcy_v, 0.0, None) / 150.0
        * bwd_term
        * uh_gate
    )
    return np.clip(composite, 0.0, 10.0).astype(np.float32)


# Keep old name as alias so any existing test imports still work.
def compute_stp(fields: dict[str, xr.DataArray]) -> Optional[np.ndarray]:
    return compute_tornado_composite(fields)


def plot_conus_forecast(
    lat: np.ndarray,
    lon: np.ndarray,
    stp: np.ndarray,
    title: str = "Tornado Forecast",
    subtitle: str = "",
    output_path: str = "forecast.png",
    dpi: int = 150,
    mxuphl: Optional[np.ndarray] = None,
    prob_mode: bool = False,
) -> str:
    """
    Render the tornado composite on a Lambert Conformal CONUS map.

    When *mxuphl* is provided (RRFS MXUPHL 0-3km grid), areas with
    MXUPHL >= 25 J/kg are overlaid with black diagonal hatching to indicate
    resolved rotating updrafts — analogous to the EF2+ hatching on Nadocast.

    Returns the resolved output path.
    """
    proj     = ccrs.LambertConformal(central_longitude=-96, central_latitude=39)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    ax  = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-125, -66, 22, 50], crs=data_crs)

    ax.add_feature(cfeature.LAND.with_scale("50m"),    facecolor="#f8f8f2", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),   facecolor="#e8eef2", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),   facecolor="#e8eef2", zorder=1)
    ax.add_feature(cfeature.STATES.with_scale("50m"),  edgecolor="#aaaaaa", linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#666666", linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#555555", linewidth=0.6, zorder=2)

    # ── Smooth the raw 3km field ─────────────────────────────────────────────
    # ML mode: model already encodes 25-50km spatial context via gradient
    # features — use light smoothing (sigma=4, ~12km) to remove speckle only.
    # TC composite mode: heavier smoothing (sigma=15, ~45km) needed since
    # the physics index is deterministic and noisy at cell scale.
    sigma = 10 if prob_mode else 15
    stp_smooth = gaussian_filter(stp.astype(np.float64), sigma=sigma)
    stp_smooth = np.clip(stp_smooth, 0.0, 10.0)

    # ── Filled contours — SPC/Nadocast color style ───────────────────────────
    levels = _PROB_LEVELS if prob_mode else _TC_LEVELS
    colors = _PROB_COLORS if prob_mode else _TC_COLORS

    ax.contourf(
        lon, lat, stp_smooth,
        levels=levels,
        colors=colors,
        extend="max",
        transform=data_crs,
        zorder=3,
        alpha=0.90,
    )
    ax.contour(
        lon, lat, stp_smooth,
        levels=levels,
        colors=["#222222"],
        linewidths=[0.6],
        transform=data_crs,
        zorder=3,
    )

    # ── MXUPHL hatching — gated inside threat zones only ────────────────────
    min_thresh = levels[0]
    if mxuphl is not None and mxuphl.shape == stp.shape:
        uh_smooth  = gaussian_filter(mxuphl.astype(np.float64), sigma=8)
        hatch_zone = np.where((uh_smooth >= 5.0) & (stp_smooth >= min_thresh), 1.0, np.nan)
        hatch_mask = np.ma.masked_invalid(hatch_zone)
        ax.contourf(
            lon, lat, hatch_mask,
            levels=[0.5, 1.5],
            hatches=["//"],
            colors=["none"],
            transform=data_crs,
            zorder=4,
        )

    # ── Legend — only show levels that actually appear in this run ────────────
    import matplotlib.patches as mpatches
    if prob_mode:
        level_labels = [(f"≥ {lbl}  tornado probability", thresh)
                        for lbl, thresh in zip(_PROB_LABELS, _PROB_LEVELS)]
    else:
        level_labels = [
            ("TC ≥ 1  Marginal",  1.0),
            ("TC ≥ 2  Slight",    2.0),
            ("TC ≥ 3  Enhanced",  3.0),
            ("TC ≥ 5  Moderate",  5.0),
            ("TC ≥ 8  High",      8.0),
        ]

    legend_handles = [
        mpatches.Patch(facecolor=c, edgecolor="#555555", linewidth=0.5, label=lbl)
        for (lbl, thresh), c in zip(level_labels, colors)
        if np.any(stp_smooth >= thresh)
    ]
    if mxuphl is not None and np.any(
        (gaussian_filter(mxuphl.astype(np.float64), sigma=8) >= 5.0) & (stp_smooth >= min_thresh)
    ):
        legend_handles.append(mpatches.Patch(
            facecolor="none", edgecolor="#444444", hatch="//",
            label="Rotating updraft (MXUPHL)",
        ))
    ax.legend(
        handles=legend_handles,
        loc="lower left", fontsize=8,
        framealpha=0.9, frameon=True,
        edgecolor="#cccccc",
        borderpad=0.7,
    )

    ax.set_title(title,    fontsize=13, fontweight="bold", loc="left",  pad=6, color="#111111")
    ax.set_title(subtitle, fontsize=9,  fontweight="normal", loc="right", pad=6, color="#555555")

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#e8eef2")

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved forecast plot to %s", output_path)
    return output_path
