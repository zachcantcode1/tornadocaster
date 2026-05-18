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


class TornadoProbabilityPredictor:
    """Wraps a trained LightGBM + isotonic calibration bundle."""

    def __init__(self, model_path: Path = _DEFAULT_MODEL_PATH):
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

        cape_ml    = v("cape_ml")
        original_shape = cape_ml.shape

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
        ugrd_850   = v("ugrd_850", 0.0)
        vgrd_850   = v("vgrd_850", 0.0)
        bwd_850_sfc = np.hypot(ugrd_850 - ugrd_10m, vgrd_850 - vgrd_10m)

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
            "tc_cape_term":         np.clip(cape_ml, 0, None) / 1500.0,
            "tc_lcl_term":          lcl_term,
            "tc_cin_term":          cin_term,
            "tc_srh_term":          srh_term,
            "tc_bwd_term":          bwd_term,
            "tc_composite":         tc,
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
        }

        import pandas as pd
        X = pd.DataFrame(
            {c: feature_map[c].ravel().astype(np.float32) for c in self._features}
        )

        raw_scores = self._lgbm.predict_proba(X)[:, 1]
        cal_probs  = self._calibrator.predict(raw_scores).astype(np.float32)
        return cal_probs.reshape(original_shape)


# Module-level singleton — loaded once, reused across forecast hours
_predictor: Optional[TornadoProbabilityPredictor] = None


def get_predictor(model_path: Path = _DEFAULT_MODEL_PATH) -> TornadoProbabilityPredictor:
    global _predictor
    if _predictor is None:
        _predictor = TornadoProbabilityPredictor(model_path)
    return _predictor
