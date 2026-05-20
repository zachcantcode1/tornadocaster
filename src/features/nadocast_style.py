from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, map_coordinates


_GRID_KM = 3.0

_BASE_EXTRA_FIELDS = [
    "cin_mu",
    "ugrd_80m", "vgrd_80m",
    "ugrd_925", "vgrd_925",
    "ugrd_850", "vgrd_850",
    "ugrd_700", "vgrd_700",
    "ugrd_500", "vgrd_500",
    "ugrd_250", "vgrd_250",
    "tmp_925", "tmp_850", "tmp_700", "tmp_500", "tmp_250",
    "dpt_925", "dpt_850", "dpt_700", "dpt_500",
    "rh_700", "pwat",
    "refc_atm", "mxuphl_03km", "mxuphl_25km", "gust_surface",
    "wind_10m", "wind_80m", "wind_925", "wind_850", "wind_700", "wind_500", "wind_250",
]

_DERIVED_FIELDS = [
    "calc_700_500_lapse", "calc_925_700_lapse",
    "calc_wind_700", "calc_wind_500",
    "calc_sbcape_hlcy", "calc_mlcape_hlcy",
    "calc_sqrt_sbcape_hlcy", "calc_sqrt_mlcape_hlcy",
    "calc_sbcape_bwd", "calc_mlcape_bwd",
    "calc_sqrt_sbcape_bwd", "calc_sqrt_mlcape_bwd",
    "calc_sbcape_200_plus_sbcin", "calc_mlcape_200_plus_mlcin",
    "calc_sqrt_sbcape_200_plus_sbcin", "calc_sqrt_mlcape_200_plus_mlcin",
    "calc_sbcape_hlcy_cin", "calc_mlcape_hlcy_cin",
    "calc_sqrt_sbcape_hlcy_cin", "calc_sqrt_mlcape_hlcy_cin",
    "calc_sbcape_bwd_hlcy", "calc_mlcape_bwd_hlcy",
    "calc_sqrt_sbcape_bwd_hlcy", "calc_sqrt_mlcape_bwd_hlcy",
    "calc_sbcape_bwd_hlcy_cin", "calc_mlcape_bwd_hlcy_cin",
    "calc_sqrt_sbcape_bwd_hlcy_cin", "calc_sqrt_mlcape_bwd_hlcy_cin",
    "calc_lapse_bwd", "calc_lapse_cold500_bwd",
    "calc_mucape_bwd", "calc_mucape_lapse_bwd", "calc_mucape_lapse_cold500_bwd",
    "calc_mucape_mixr925_lapse_cold500_bwd", "calc_mucape_mixr850_lapse_cold500_bwd",
    "calc_max_wind_le_850", "calc_sum_wind_le_850",
    "calc_max_wind_le_700", "calc_sum_wind_le_700",
    "calc_scpish_rm", "calc_scpish_gt1",
    "calc_umean", "calc_vmean", "calc_ushear", "calc_vshear", "calc_shear",
    "calc_ustm", "calc_vstm", "calc_ustm_500_mean", "calc_vstm_500_mean",
    "calc_div_925", "calc_div_850", "calc_div_250", "calc_diff_div_250_925",
    "calc_conv_only_925", "calc_conv_only_850",
    "calc_abs_vort_925", "calc_abs_vort_850", "calc_abs_vort_700",
    "calc_abs_vort_500", "calc_abs_vort_250",
]

_BOUNDARY_FIELDS = [
    "cape_ml", "cin_ml", "hlcy_3km", "vwsh_0_6km", "dpt_2m", "tmp_2m",
    "cape_surface", "cape_mu", "cin_surface", "cin_mu",
    "refc_atm", "mxuphl_03km", "mxuphl_25km",
    "calc_scpish_rm", "calc_conv_only_925", "calc_conv_only_850",
    "calc_diff_div_250_925", "calc_abs_vort_925", "calc_abs_vort_850",
    "calc_700_500_lapse", "calc_925_700_lapse",
]

_SPATIAL_SCALES_KM = (25, 50, 100)

_THRESHOLD_FEATURES = [
    ("cape_ml", "gt", 500.0),
    ("cape_ml", "gt", 1000.0),
    ("cape_ml", "gt", 1500.0),
    ("cape_ml", "gt", 2000.0),
    ("cape_ml", "gt", 3000.0),
    ("cin_ml", "lt", 0.0),
    ("cin_ml", "lt", -50.0),
    ("cin_ml", "lt", -100.0),
    ("hlcy_3km", "gt", 100.0),
    ("hlcy_3km", "gt", 200.0),
    ("hlcy_3km", "gt", 400.0),
    ("dpt_2m", "gt", 283.15),
    ("dpt_2m", "gt", 288.71),
    ("dpt_2m", "gt", 291.48),
    ("pwat", "gt", 25.0),
    ("pwat", "gt", 37.5),
    ("pwat", "gt", 50.0),
    ("refc_atm", "gt", 20.0),
    ("refc_atm", "gt", 30.0),
    ("refc_atm", "gt", 40.0),
    ("refc_atm", "gt", 50.0),
    ("mxuphl_25km", "gt", 25.0),
    ("gust_surface", "gt", 20.6),
]

_UPSTREAM_FEATURES = [
    "upstream_cape_ml_2hr",
    "upstream_mlcape_2hr",
    "upstream_dpt_2m_1hr",
    "upstream_dpt_2m_2hr",
    "upstream_refc_atm_2hr",
    "upstream_crain_2hr",
    "storm_upstream_conv_925_3hr_gated_scp",
    "storm_upstream_conv_925_6hr_gated_scp",
    "storm_upstream_conv_925_9hr_gated_scp",
    "storm_upstream_conv_850_3hr_gated_scp",
    "storm_upstream_conv_850_6hr_gated_scp",
    "storm_upstream_conv_850_9hr_gated_scp",
    "storm_upstream_diff_div_250_925_3hr_gated_scp",
    "storm_upstream_diff_div_250_925_6hr_gated_scp",
    "storm_upstream_diff_div_250_925_9hr_gated_scp",
]

_CLIMO_FEATURES = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
]


def _threshold_name(field: str, op: str, threshold: float) -> str:
    value = str(threshold).replace("-", "neg_").replace(".", "p")
    return f"{field}_prob_{op}_{value}"


def _spatial_feature_names() -> list[str]:
    names: list[str] = []
    for field in _BOUNDARY_FIELDS:
        for scale in _SPATIAL_SCALES_KM:
            stem = f"{field}_{scale}km"
            names.extend([
                f"{stem}_mean",
                f"{stem}_grad_mag",
                f"{stem}_forward_grad",
                f"{stem}_leftward_grad",
                f"{stem}_linestraddling_grad",
            ])
    return names


NADOCAST_STYLE_FEATURE_COLS = (
    _BASE_EXTRA_FIELDS
    + _DERIVED_FIELDS
    + [_threshold_name(field, op, threshold) for field, op, threshold in _THRESHOLD_FEATURES]
    + [f"{_threshold_name(field, op, threshold)}_25km_mean" for field, op, threshold in _THRESHOLD_FEATURES]
    + _spatial_feature_names()
    + _UPSTREAM_FEATURES
    + _CLIMO_FEATURES
)


def _as_array(value: object, shape: tuple[int, int], default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=np.float32)
    if isinstance(value, xr.DataArray):
        arr = np.asarray(value.values, dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        return np.full(shape, default, dtype=np.float32)
    return np.nan_to_num(arr, nan=default, posinf=default, neginf=default).astype(np.float32)


def _get_array(
    feature_map: dict[str, np.ndarray],
    fields: dict[str, object],
    name: str,
    shape: tuple[int, int],
    default: float = 0.0,
) -> np.ndarray:
    if name in feature_map:
        return _as_array(feature_map[name], shape, default)
    return _as_array(fields.get(name), shape, default)


def _smoothed_gradients(arr: np.ndarray, scale_km: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = max(scale_km / _GRID_KM, 0.1)
    smoothed = gaussian_filter(arr.astype(np.float64), sigma=sigma)
    gy, gx = np.gradient(smoothed)
    mag = np.hypot(gx, gy)
    return gx.astype(np.float32), gy.astype(np.float32), mag.astype(np.float32)


def _motion_components(
    feature_map: dict[str, np.ndarray],
    fields: dict[str, object],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    u_stm = _get_array(feature_map, fields, "calc_ustm", shape)
    v_stm = _get_array(feature_map, fields, "calc_vstm", shape)
    if np.nanmax(np.hypot(u_stm, v_stm)) > 0.1:
        return u_stm, v_stm

    u10 = _get_array(feature_map, fields, "ugrd_10m", shape)
    v10 = _get_array(feature_map, fields, "vgrd_10m", shape)
    u500 = _get_array(feature_map, fields, "ugrd_500", shape)
    v500 = _get_array(feature_map, fields, "vgrd_500", shape)
    if np.nanmax(np.hypot(u500, v500)) > 0.1:
        return 0.5 * (u10 + u500), 0.5 * (v10 + v500)
    return u10, v10


def _motion_unit(
    feature_map: dict[str, np.ndarray],
    fields: dict[str, object],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    u_stm, v_stm = _motion_components(feature_map, fields, shape)
    speed = np.hypot(u_stm, v_stm)
    return (
        np.divide(u_stm, speed + 1e-3).astype(np.float32),
        np.divide(v_stm, speed + 1e-3).astype(np.float32),
    )


def _upstream_sample(arr: np.ndarray, u_ms: np.ndarray, v_ms: np.ndarray, hours: float) -> np.ndarray:
    height, width = arr.shape
    y, x = np.indices(arr.shape, dtype=np.float32)
    distance_px = (hours * 3600.0 / 1000.0) / _GRID_KM
    coords = np.array([
        y + v_ms.astype(np.float32) * distance_px,
        x - u_ms.astype(np.float32) * distance_px,
    ])
    sampled = map_coordinates(
        arr.astype(np.float32),
        coords,
        order=1,
        mode="nearest",
    )
    return sampled.reshape(height, width).astype(np.float32)


def _add_threshold_features(
    out: dict[str, np.ndarray],
    feature_map: dict[str, np.ndarray],
    fields: dict[str, object],
    shape: tuple[int, int],
) -> None:
    for field, op, threshold in _THRESHOLD_FEATURES:
        arr = _get_array(feature_map, fields, field, shape)
        if op == "gt":
            binary = (arr > threshold).astype(np.float32)
        else:
            binary = (arr < threshold).astype(np.float32)
        name = _threshold_name(field, op, threshold)
        out[name] = binary
        out[f"{name}_25km_mean"] = gaussian_filter(binary, sigma=25.0 / _GRID_KM).astype(np.float32)


def _add_wind_speed_proxies(
    out: dict[str, np.ndarray],
    fields: dict[str, object],
    shape: tuple[int, int],
) -> None:
    level_pairs = {
        "10m": ("ugrd_10m", "vgrd_10m"),
        "80m": ("ugrd_80m", "vgrd_80m"),
        "925": ("ugrd_925", "vgrd_925"),
        "850": ("ugrd_850", "vgrd_850"),
        "700": ("ugrd_700", "vgrd_700"),
        "500": ("ugrd_500", "vgrd_500"),
        "250": ("ugrd_250", "vgrd_250"),
    }
    for level, (u_name, v_name) in level_pairs.items():
        out_name = f"wind_{level}"
        if out_name in out and np.nanmax(np.abs(out[out_name])) > 0.0:
            continue
        u = _get_array(out, fields, u_name, shape)
        v = _get_array(out, fields, v_name, shape)
        out[out_name] = np.hypot(u, v).astype(np.float32)


def _add_spatial_features(
    out: dict[str, np.ndarray],
    feature_map: dict[str, np.ndarray],
    fields: dict[str, object],
    shape: tuple[int, int],
) -> None:
    unit_u, unit_v = _motion_unit(feature_map, fields, shape)
    for field in _BOUNDARY_FIELDS:
        arr = _get_array(feature_map, fields, field, shape)
        for scale in _SPATIAL_SCALES_KM:
            sigma = scale / _GRID_KM
            mean = gaussian_filter(arr.astype(np.float64), sigma=sigma).astype(np.float32)
            gx, gy, mag = _smoothed_gradients(arr, scale)
            forward = gx * unit_u + gy * unit_v
            leftward = -gx * unit_v + gy * unit_u
            stem = f"{field}_{scale}km"
            out[f"{stem}_mean"] = mean
            out[f"{stem}_grad_mag"] = mag
            out[f"{stem}_forward_grad"] = forward.astype(np.float32)
            out[f"{stem}_leftward_grad"] = leftward.astype(np.float32)
            out[f"{stem}_linestraddling_grad"] = np.abs(leftward).astype(np.float32)


def _add_upstream_features(
    out: dict[str, np.ndarray],
    feature_map: dict[str, np.ndarray],
    fields: dict[str, object],
    shape: tuple[int, int],
) -> None:
    u_stm, v_stm = _motion_components(feature_map, fields, shape)
    scp = np.clip(_get_array(feature_map, fields, "calc_scpish_rm", shape), 0.0, None)
    scp_gate = np.clip(scp / 1.0, 0.0, 1.0).astype(np.float32)

    cape_ml = _get_array(feature_map, fields, "cape_ml", shape)
    dpt_2m = _get_array(feature_map, fields, "dpt_2m", shape)
    refc = _get_array(feature_map, fields, "refc_atm", shape)
    crain = _get_array(feature_map, fields, "crain_surface", shape)
    out["upstream_cape_ml_2hr"] = _upstream_sample(cape_ml, u_stm, v_stm, 2.0)
    out["upstream_mlcape_2hr"] = out["upstream_cape_ml_2hr"]
    out["upstream_dpt_2m_1hr"] = _upstream_sample(dpt_2m, u_stm, v_stm, 1.0)
    out["upstream_dpt_2m_2hr"] = _upstream_sample(dpt_2m, u_stm, v_stm, 2.0)
    out["upstream_refc_atm_2hr"] = _upstream_sample(refc, u_stm, v_stm, 2.0)
    out["upstream_crain_2hr"] = _upstream_sample(crain, u_stm, v_stm, 2.0)

    forcing_sources = [
        ("conv_925", "calc_conv_only_925"),
        ("conv_850", "calc_conv_only_850"),
        ("diff_div_250_925", "calc_diff_div_250_925"),
    ]
    for label, field in forcing_sources:
        arr = _get_array(feature_map, fields, field, shape)
        for hours in (3, 6, 9):
            sampled = _upstream_sample(arr, u_stm, v_stm, float(hours))
            out[f"storm_upstream_{label}_{hours}hr_gated_scp"] = (
                sampled * scp_gate
            ).astype(np.float32)


def _add_climo_features(
    out: dict[str, np.ndarray],
    shape: tuple[int, int],
    valid_dt: datetime | None,
) -> None:
    if valid_dt is None:
        hour = 18.0
        month = 5.0
    else:
        hour = float(valid_dt.hour)
        month = float(valid_dt.month)
    out["hour_sin"] = np.full(shape, np.sin(2.0 * np.pi * hour / 24.0), dtype=np.float32)
    out["hour_cos"] = np.full(shape, np.cos(2.0 * np.pi * hour / 24.0), dtype=np.float32)
    out["month_sin"] = np.full(shape, np.sin(2.0 * np.pi * month / 12.0), dtype=np.float32)
    out["month_cos"] = np.full(shape, np.cos(2.0 * np.pi * month / 12.0), dtype=np.float32)


def add_nadocast_style_features(
    feature_map: dict[str, np.ndarray],
    fields: dict[str, object],
    shape: tuple[int, int],
    valid_dt: datetime | None = None,
) -> dict[str, np.ndarray]:
    """
    Extend a per-grid feature map with Nadocast-inspired predictors.

    These are deterministic HRRR/RRFS approximations of upstream Nadocast ideas:
    motion-relative spatial gradients, threshold-probability proxies, low-level
    forcing, storm-upstream features, and richer derived severe-weather terms.
    """
    out = dict(feature_map)

    for name in _BASE_EXTRA_FIELDS:
        if name not in out:
            out[name] = _get_array(out, fields, name, shape)
    _add_wind_speed_proxies(out, fields, shape)
    for name in _DERIVED_FIELDS:
        if name not in out:
            out[name] = _get_array(out, fields, name, shape)
    if np.nanmax(np.abs(out.get("calc_max_wind_le_850", 0.0))) <= 0.0:
        out["calc_max_wind_le_850"] = np.maximum.reduce([
            out["wind_10m"], out["wind_80m"], out["wind_925"], out["wind_850"],
        ]).astype(np.float32)
        out["calc_sum_wind_le_850"] = (
            out["wind_10m"] + out["wind_80m"] + out["wind_925"] + out["wind_850"]
        ).astype(np.float32)
    if np.nanmax(np.abs(out.get("calc_max_wind_le_700", 0.0))) <= 0.0:
        out["calc_max_wind_le_700"] = np.maximum(out["calc_max_wind_le_850"], out["wind_700"]).astype(np.float32)
        out["calc_sum_wind_le_700"] = (out["calc_sum_wind_le_850"] + out["wind_700"]).astype(np.float32)

    _add_threshold_features(out, out, fields, shape)
    _add_spatial_features(out, out, fields, shape)
    _add_upstream_features(out, out, fields, shape)
    _add_climo_features(out, shape, valid_dt)
    return out


def stack_feature_columns(
    feature_map: dict[str, np.ndarray],
    feature_names: Iterable[str],
    shape: tuple[int, int],
) -> np.ndarray:
    cols = [
        _as_array(feature_map.get(name), shape).ravel()
        for name in feature_names
    ]
    if not cols:
        return np.zeros((shape[0] * shape[1], 0), dtype=np.float32)
    return np.stack(cols, axis=1).astype(np.float32)
