"""
calibrated_predictor.py — Load a trained tornado probability model and run
inference on a dict of xarray DataArrays (same interface as compute_tornado_composite).

The model outputs calibrated probability in [0, 1]: the estimated chance of
a tornado within 40 km of each grid point during the forecast valid period.
"""
from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from src.features.derived import build_first_pass_derived_fields
from src.features.nadocast_style import add_nadocast_style_features
from src.training.climatology import get_climo_freq

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "tornado_lgbm.pkl"

# Must match _GRID_KM in historical_sampler.py
_GRID_KM = 3.0


def _grad_mag(arr2d: np.ndarray, scale_km: float) -> np.ndarray:
    """Gradient magnitude at given spatial scale — detects drylines / fronts."""
    sigma = scale_km / _GRID_KM
    smoothed = gaussian_filter(arr2d.astype(np.float64), sigma=sigma)
    gy, gx = np.gradient(smoothed)
    return np.hypot(gx, gy).astype(np.float32)


def _nbr_mean(arr2d: np.ndarray, scale_km: float) -> np.ndarray:
    """Neighborhood mean at scale_km — captures background synoptic state."""
    sigma = scale_km / _GRID_KM
    return gaussian_filter(arr2d.astype(np.float64), sigma=sigma).astype(np.float32)


def _soft_ramp(arr2d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map values to a clipped 0-1 support score."""
    if hi <= lo:
        raise ValueError("hi must be greater than lo")
    return np.clip((arr2d.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)


def _convective_support(fields: dict[str, xr.DataArray], shape: tuple[int, int]) -> np.ndarray:
    """
    Estimate whether convection exists near each grid point.

    The ML model is an environment model; this support field prevents high
    tornado probabilities in places with no storm signal. It intentionally uses
    several optional RRFS/HRRR fields and degrades gracefully when some are
    unavailable.
    """
    core_support = np.zeros(shape, dtype=np.float64)
    rotation_support = np.zeros(shape, dtype=np.float64)
    weak_support = np.zeros(shape, dtype=np.float64)

    def arr(name: str) -> Optional[np.ndarray]:
        da = fields.get(name)
        if da is None:
            return None
        return np.nan_to_num(np.asarray(da.values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    refc = arr("refc_atm")
    if refc is not None:
        # Require a real convective core for meaningful tornado probabilities.
        # Weak stratiform/light precip should not unlock broad risk areas.
        core_support = np.maximum(core_support, _soft_ramp(refc, 35.0, 55.0))
        weak_support = np.maximum(weak_support, 0.30 * _soft_ramp(refc, 25.0, 40.0))

    mxuphl = arr("mxuphl_03km")
    mxuphl_25km = arr("mxuphl_25km")
    if mxuphl is not None or mxuphl_25km is not None:
        uh = np.zeros(shape, dtype=np.float64)
        if mxuphl is not None:
            uh = np.maximum(uh, np.clip(mxuphl, 0.0, None))
        if mxuphl_25km is not None:
            uh = np.maximum(uh, np.clip(mxuphl_25km, 0.0, None))
        rotation_support = np.maximum(rotation_support, _soft_ramp(uh, 25.0, 125.0))
        weak_support = np.maximum(weak_support, 0.25 * _soft_ramp(uh, 10.0, 50.0))

    relv = arr("relv_1km")
    if relv is not None:
        relv_scaled = np.abs(relv)
        if np.nanmax(relv_scaled) < 1.0:
            relv_scaled = relv_scaled * 1e5
        rotation_support = np.maximum(rotation_support, 0.70 * _soft_ramp(relv_scaled, 10.0, 40.0))

    efhl = arr("efhl_surface")
    if efhl is not None:
        rotation_support = np.maximum(rotation_support, 0.55 * _soft_ramp(np.clip(efhl, 0.0, None), 75.0, 300.0))

    hail = arr("hail_surface")
    if hail is not None:
        core_support = np.maximum(core_support, 0.60 * _soft_ramp(np.clip(hail, 0.0, None), 5.0, 25.0))

    gust = arr("gust_surface")
    if gust is not None:
        weak_support = np.maximum(weak_support, 0.20 * _soft_ramp(gust, 20.0, 35.0))

    support = np.maximum(
        weak_support,
        np.maximum(
            core_support,
            np.sqrt(np.clip(core_support * rotation_support, 0.0, 1.0)),
        ),
    )
    if np.nanmax(support) <= 0.0:
        return support.astype(np.float32)

    # Lightly spread only around storm cores. Larger spreading is handled by the
    # map renderer, and too much here turns storm corridors into outlook blobs.
    support = gaussian_filter(support, sigma=3.0)
    return np.clip(support, 0.0, 1.0).astype(np.float32)


def _tornado_environment_support(
    cape_ml: np.ndarray,
    cin_ml: np.ndarray,
    hlcy_3km: np.ndarray,
    vwsh: np.ndarray,
    lcl_m: np.ndarray,
) -> np.ndarray:
    """Score whether the environment near a storm is tornado-capable."""
    cape_term = _soft_ramp(np.clip(cape_ml, 0.0, None), 100.0, 1000.0)
    srh_term = _soft_ramp(np.clip(hlcy_3km, 0.0, None), 40.0, 150.0)
    shear_term = _soft_ramp(np.clip(vwsh, 0.0, None), 7.0, 18.0)
    lcl_term = np.clip((1800.0 - lcl_m.astype(np.float64)) / 1100.0, 0.0, 1.0)
    cin_term = np.clip((150.0 + cin_ml.astype(np.float64)) / 125.0, 0.0, 1.0)

    env = cape_term * lcl_term * cin_term * np.sqrt(np.clip(srh_term * shear_term, 0.0, 1.0))
    env = gaussian_filter(env, sigma=4.0)
    return np.clip(env, 0.0, 1.0).astype(np.float32)


def _apply_storm_gate(
    probs: np.ndarray,
    support: np.ndarray,
    env_support: np.ndarray | None = None,
) -> np.ndarray:
    """
    Cap calibrated tornado probabilities by observed/model storm support.

    This keeps the model from producing operationally impossible high
    probabilities where the environment is favorable but convection is absent.
    """
    probs = np.clip(probs.astype(np.float64), 0.0, 1.0)
    if env_support is None:
        env_support_arr = np.ones_like(probs, dtype=np.float64)
    else:
        env_support_arr = np.clip(env_support.astype(np.float64), 0.0, 1.0)
    effective_support = np.clip(support.astype(np.float64), 0.0, 1.0) * np.clip(
        0.35 + 0.80 * env_support_arr,
        0.0,
        1.0,
    )

    # Keep a broad low-end environment envelope for SPC/Nadocast-style 2%
    # outlook areas, but only around modeled convection. The raw environment
    # model can be right about ingredients yet wrong operationally when storms
    # never form, so unsupported environment risk stays below display threshold.
    smoothed_probs = gaussian_filter(probs, sigma=8.0)
    near_storm = gaussian_filter(effective_support, sigma=6.0)
    envelope_cap = np.interp(
        env_support_arr,
        [0.12, 0.35, 0.65, 1.00],
        [0.025, 0.040, 0.065, 0.090],
    )
    environment_envelope = np.where(
        (near_storm >= 0.045) & (env_support_arr >= 0.12) & (smoothed_probs >= 0.032),
        np.minimum(np.clip((smoothed_probs - 0.014) * 0.72, 0.0, None), envelope_cap),
        0.0,
    )

    if np.nanmax(effective_support) <= 0.0:
        return environment_envelope.astype(np.float32)

    caps = np.interp(
        effective_support,
        [0.00, 0.20, 0.45, 0.70, 0.90, 1.00],
        [0.001, 0.010, 0.038, 0.085, 0.155, 0.240],
    )
    storm_supported = np.minimum(probs, caps)
    storm_supported *= np.clip(0.35 + 0.70 * effective_support, 0.0, 1.0)

    # Above the 2% outlook envelope, probabilities must be attached to modeled
    # storms. This keeps broad environmental risk from becoming false 5-15% blobs.
    combined = np.maximum(environment_envelope, storm_supported)
    return np.clip(combined, 0.0, 0.30).astype(np.float32)


class TornadoProbabilityPredictor:
    """Wraps a trained LightGBM + isotonic calibration bundle."""

    def __init__(self, model_path: Path | str | None = None):
        model_path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run `python -m src.training.build_dataset` then "
                "`python -m src.training.train` first."
            )
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self._lgbm       = bundle["lgbm"]
        self._calibrator = bundle["calibrator"]
        self._features   = bundle["features"]
        trained_on       = bundle.get("trained_on", "unknown")
        logger.info(
            "Loaded tornado probability model (trained %s, val AUC-ROC=%.4f, AUC-PR=%.4f)",
            trained_on,
            bundle.get("val_auc_roc", float("nan")),
            bundle.get("val_auc_pr",  float("nan")),
        )

    def predict(
        self,
        fields: dict[str, xr.DataArray],
        valid_dt: Optional[datetime] = None,
        fhour: int = 1,
    ) -> Optional[np.ndarray]:
        """
        Compute calibrated tornado probability for each grid point.

        Parameters
        ----------
        fields    : dict of xarray DataArrays (same dict passed to compute_tornado_composite)
        valid_dt  : datetime of the forecast valid time (for temporal features)

        Returns
        -------
        float32 ndarray, shape (H, W), values in [0, 1], or None if required
        fields are missing.
        """
        required = {"cape_ml", "cin_ml", "hlcy_3km"}
        if not required.issubset(fields):
            logger.warning("calibrated_predictor: missing required fields %s", required - fields.keys())
            return None

        def v(name: str, default: float = 0.0) -> np.ndarray:
            da = fields.get(name)
            if da is None:
                ref = next(iter(fields.values()))
                return np.full(np.asarray(ref.values).shape, default, dtype=np.float64)
            return np.asarray(da.values, dtype=np.float64)

        def optional_v(name: str) -> Optional[np.ndarray]:
            da = fields.get(name)
            if da is None:
                return None
            return np.asarray(da.values, dtype=np.float64)

        cape_ml    = v("cape_ml")
        original_shape = cape_ml.shape

        cin_ml     = v("cin_ml")
        hlcy_3km   = v("hlcy_3km")
        hlcy_1km   = v("hlcy_1km", 0.0)
        vwsh_arr   = optional_v("vwsh_0_6km")
        tmp_2m     = v("tmp_2m")
        dpt_2m     = v("dpt_2m")
        cape_sfc   = v("cape_surface")
        cape_mu    = v("cape_mu")
        cin_sfc    = v("cin_surface")
        ugrd_10m   = v("ugrd_10m")
        vgrd_10m   = v("vgrd_10m")
        ugrd_850   = optional_v("ugrd_850")
        vgrd_850   = optional_v("vgrd_850")
        ugrd_500   = optional_v("ugrd_500")
        vgrd_500   = optional_v("vgrd_500")
        ugrd_hi_proxy = optional_v("ugrd_pbl")
        vgrd_hi_proxy = optional_v("vgrd_pbl")
        if ugrd_hi_proxy is None:
            ugrd_hi_proxy = optional_v("ugrd_80m")
        if vgrd_hi_proxy is None:
            vgrd_hi_proxy = optional_v("vgrd_80m")

        # vwsh for ML model features: must match training distribution.
        # Training used HRRR which lacks VWSH, so vwsh_0_6km was 0.0 in all training shards.
        if vwsh_arr is not None:
            vwsh = vwsh_arr
        else:
            vwsh = np.zeros(original_shape, dtype=np.float64)

        # gate_vwsh for storm-gate env_support only — uses best available shear estimate
        # without affecting the ML model's feature space.
        if vwsh_arr is not None:
            gate_vwsh = vwsh_arr
        elif ugrd_500 is not None and vgrd_500 is not None:
            # 500mb (~5.5km AGL) minus 10m is a good 0-6km bulk shear proxy
            gate_vwsh = np.clip(np.hypot(ugrd_500 - ugrd_10m, vgrd_500 - vgrd_10m), 0.0, 40.0)
        elif ugrd_hi_proxy is not None and vgrd_hi_proxy is not None:
            shallow_shear = np.hypot(ugrd_hi_proxy - ugrd_10m, vgrd_hi_proxy - vgrd_10m)
            gate_vwsh = np.clip(shallow_shear * 2.0, 0.0, 35.0)
        else:
            logger.warning("calibrated_predictor: no usable bulk-shear field for storm gate; using 6 m/s")
            gate_vwsh = np.full(original_shape, 6.0, dtype=np.float64)

        if ugrd_850 is not None and vgrd_850 is not None:
            bwd_850_sfc = np.hypot(ugrd_850 - ugrd_10m, vgrd_850 - vgrd_10m)
        elif ugrd_hi_proxy is not None and vgrd_hi_proxy is not None:
            shallow_shear = np.hypot(ugrd_hi_proxy - ugrd_10m, vgrd_hi_proxy - vgrd_10m)
            bwd_850_sfc = np.clip(shallow_shear * 2.0, 0.0, 35.0)
        else:
            bwd_850_sfc = np.full(original_shape, 6.0, dtype=np.float64)

        # TC component terms
        lcl_m    = 122.0 * np.clip(tmp_2m - dpt_2m, 0.0, None)
        lcl_term = np.clip((2000.0 - lcl_m) / 1000.0, 0.0, 1.0)
        cin_term = np.where(cin_ml < -50.0, 0.0, np.clip((200.0 + cin_ml) / 150.0, 0.0, 1.0))
        bwd_term = np.clip(vwsh, 0.0, None) / 12.0
        srh_term = np.clip(hlcy_3km, 0.0, None) / 150.0

        tc = (
            np.clip(cape_ml, 0, None) / 1500.0
            * lcl_term * cin_term * srh_term * bwd_term
        )
        tc = np.clip(tc, 0.0, 10.0)

        # Storm-physics composites using 1km SRH
        ehi_1km = np.clip(cape_ml, 0.0, None) * np.clip(hlcy_1km, 0.0, None) / 160000.0
        stp_fixed = (
            np.clip(cape_ml, 0.0, None) / 1500.0
            * lcl_term
            * cin_term
            * np.clip(hlcy_1km, 0.0, None) / 150.0
            * np.clip(vwsh, 0.0, None) / 20.0
        )

        # ── Spatial gradient features ────────────────────────────────────────
        cape_grad_25 = _grad_mag(cape_ml,  25.0)
        cape_grad_50 = _grad_mag(cape_ml,  50.0)
        hlcy_grad_25 = _grad_mag(hlcy_3km, 25.0)
        hlcy_grad_50 = _grad_mag(hlcy_3km, 50.0)
        cin_grad_25  = _grad_mag(cin_ml,   25.0)
        tmp_grad_25  = _grad_mag(tmp_2m,   25.0)
        dpt_grad_25  = _grad_mag(dpt_2m,   25.0)

        # ── Neighborhood mean features ───────────────────────────────────────
        cape_mean_25  = _nbr_mean(cape_ml,  25.0)
        cape_mean_50  = _nbr_mean(cape_ml,  50.0)
        cape_mean_100 = _nbr_mean(cape_ml, 100.0)
        hlcy_mean_25  = _nbr_mean(hlcy_3km, 25.0)
        hlcy_mean_50  = _nbr_mean(hlcy_3km, 50.0)
        cin_mean_25   = _nbr_mean(cin_ml,   25.0)
        tmp_mean_25   = _nbr_mean(tmp_2m,   25.0)
        dpt_mean_25   = _nbr_mean(dpt_2m,   25.0)
        vwsh_mean_25  = _nbr_mean(vwsh,     25.0)

        # ── Geographic position ──────────────────────────────────────────────
        ref_da = next(iter(fields.values()))
        if "latitude" in ref_da.coords and "longitude" in ref_da.coords:
            lat_grid = np.asarray(ref_da.coords["latitude"].values,  dtype=np.float32)
            lon_grid = np.asarray(ref_da.coords["longitude"].values, dtype=np.float32)
            lon_grid = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
        else:
            lat_grid = np.zeros(original_shape, dtype=np.float32)
            lon_grid = np.zeros(original_shape, dtype=np.float32)

        # ── Temporal features ────────────────────────────────────────────────
        if valid_dt is not None:
            hour_utc = float(valid_dt.hour)
            doy      = float(valid_dt.timetuple().tm_yday)
        else:
            hour_utc = 18.0
            doy      = 150.0
        doy_sin = np.sin(2.0 * np.pi * doy / 365.0)
        doy_cos = np.cos(2.0 * np.pi * doy / 365.0)

        feature_map = {
            "cape_ml":              cape_ml,
            "cin_ml":               cin_ml,
            "hlcy_3km":             hlcy_3km,
            "vwsh_0_6km":           vwsh,
            "tmp_2m":               tmp_2m,
            "dpt_2m":               dpt_2m,
            "cape_surface":         cape_sfc,
            "cape_mu":              cape_mu,
            "cin_surface":          cin_sfc,
            "ugrd_10m":             ugrd_10m,
            "vgrd_10m":             vgrd_10m,
            "bwd_850_sfc":          bwd_850_sfc,
            "hlcy_1km":             hlcy_1km,
            "tc_cape_term":         np.clip(cape_ml, 0, None) / 1500.0,
            "tc_lcl_term":          lcl_term,
            "tc_cin_term":          cin_term,
            "tc_srh_term":          srh_term,
            "tc_bwd_term":          bwd_term,
            "tc_composite":         tc,
            "ehi_1km":              ehi_1km,
            "stp_fixed":            stp_fixed,
            "cape_ml_grad_25km":    cape_grad_25,
            "cape_ml_grad_50km":    cape_grad_50,
            "hlcy_3km_grad_25km":   hlcy_grad_25,
            "hlcy_3km_grad_50km":   hlcy_grad_50,
            "cin_ml_grad_25km":     cin_grad_25,
            "tmp_2m_grad_25km":     tmp_grad_25,
            "dpt_2m_grad_25km":     dpt_grad_25,
            "cape_ml_mean_25km":    cape_mean_25,
            "cape_ml_mean_50km":    cape_mean_50,
            "cape_ml_mean_100km":   cape_mean_100,
            "hlcy_3km_mean_25km":   hlcy_mean_25,
            "hlcy_3km_mean_50km":   hlcy_mean_50,
            "cin_ml_mean_25km":     cin_mean_25,
            "tmp_2m_mean_25km":     tmp_mean_25,
            "dpt_2m_mean_25km":     dpt_mean_25,
            "vwsh_mean_25km":       vwsh_mean_25,
            "lat":                  lat_grid,
            "lon":                  lon_grid,
            "hour_utc":             np.full(original_shape, hour_utc),
            "doy_sin":              np.full(original_shape, doy_sin),
            "doy_cos":              np.full(original_shape, doy_cos),
            "climo_freq":           get_climo_freq(
                                        lat_grid.astype(np.float32),
                                        lon_grid.astype(np.float32),
                                        valid_dt.month if valid_dt is not None else 5,
                                    ).astype(np.float32),
            "fhour":                np.full(original_shape, float(fhour), dtype=np.float32),
        }

        try:
            expanded_fields = {**fields, **build_first_pass_derived_fields(fields)}
        except Exception:
            expanded_fields = dict(fields)
        feature_map = add_nadocast_style_features(
            feature_map,
            expanded_fields,
            original_shape,
            valid_dt=valid_dt,
        )

        import pandas as pd
        X = pd.DataFrame(
            {
                c: feature_map.get(c, np.zeros(original_shape, dtype=np.float32)).ravel().astype(np.float32)
                for c in self._features
            }
        )

        raw_scores = self._lgbm.predict_proba(X)[:, 1]
        cal_probs  = self._calibrator.predict(raw_scores).astype(np.float32).reshape(original_shape)
        support = _convective_support(fields, original_shape)
        env_support = _tornado_environment_support(cape_ml, cin_ml, hlcy_3km, gate_vwsh, lcl_m)
        gated_probs = _apply_storm_gate(cal_probs, support, env_support)
        if np.nanmax(cal_probs) > 0.25 and np.nanmax(gated_probs) < np.nanmax(cal_probs) * 0.5:
            logger.info(
                "Storm gate reduced max ML probability from %.1f%% to %.1f%%",
                float(np.nanmax(cal_probs)) * 100.0,
                float(np.nanmax(gated_probs)) * 100.0,
            )
        return gated_probs


# Module-level singleton — loaded once, reused across forecast hours
_predictor: Optional[TornadoProbabilityPredictor] = None


def get_predictor(model_path: Path | str | None = None) -> TornadoProbabilityPredictor:
    global _predictor
    resolved = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
    if _predictor is None or getattr(_predictor, "_model_path", None) != resolved:
        _predictor = TornadoProbabilityPredictor(model_path)
        _predictor._model_path = resolved
    return _predictor
