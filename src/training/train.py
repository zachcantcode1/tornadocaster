"""
train.py — Train the calibrated tornado probability classifier.

Usage:
    python -m src.training.train --data data/training --out models/tornado_lgbm.pkl

Reads all Parquet shards from --data, trains a LightGBM binary classifier,
applies isotonic-regression calibration, and saves the pipeline to --out.

The saved model is a dict:
  {
      "lgbm":      LGBMClassifier (raw scores),
      "calibrator": IsotonicRegression (maps raw score → calibrated probability),
      "features":  list[str] (feature column order),
      "trained_on": "YYYY-MM-DD",
  }
"""
from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(
    data_dir: Path,
    include_latlon: bool = False,
    include_seasonal_priors: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load all Parquet shards into (X, y) arrays."""
    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found in {data_dir}")
    logger.info("Loading %d shards …", len(shards))

    frames = [pd.read_parquet(p) for p in shards]
    df = pd.concat(frames, ignore_index=True)
    logger.info("Total rows: %d  (positives: %d = %.2f%%)",
                len(df), df["label"].sum(), 100 * df["label"].mean())

    from src.training.historical_sampler import FEATURE_COLS
    features = list(FEATURE_COLS)
    if not include_latlon:
        features = [c for c in features if c not in {"lat", "lon"}]
    if not include_seasonal_priors:
        features = [c for c in features if c not in {"doy_sin", "doy_cos", "climo_freq"}]
    X = df[features].astype(np.float32)   # keep as DataFrame — preserves feature names
    y = df["label"].values.astype(np.int8)
    if "date" in df:
        groups = df["date"].astype(str).values
    else:
        logger.warning("Dataset has no date column; falling back to row-level grouping.")
        groups = np.arange(len(df)).astype(str)
    return X, y, groups, features


def train(
    data_dir: Path,
    out_path: Path,
    include_latlon: bool = False,
    include_seasonal_priors: bool = False,
) -> None:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, average_precision_score

    X, y, groups, features = load_dataset(
        data_dir,
        include_latlon=include_latlon,
        include_seasonal_priors=include_seasonal_priors,
    )

    # Split by forecast date so the same event/grid cannot leak across train
    # and validation. Random row splits make storm-scale models look much
    # better than they actually are operationally.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    val_groups = groups[val_idx]

    cal_eval_splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
    cal_rel_idx, eval_rel_idx = next(
        cal_eval_splitter.split(X_val, y_val, groups=val_groups)
    )
    X_cal, y_cal = X_val.iloc[cal_rel_idx], y_val[cal_rel_idx]
    X_eval, y_eval = X_val.iloc[eval_rel_idx], y_val[eval_rel_idx]
    logger.info(
        "Date-grouped split: train rows=%d dates=%d | cal rows=%d dates=%d | eval rows=%d dates=%d",
        len(X_train), len(set(groups[train_idx])),
        len(X_cal), len(set(val_groups[cal_rel_idx])),
        len(X_eval), len(set(val_groups[eval_rel_idx])),
    )
    if not include_latlon:
        logger.info("Excluded raw lat/lon features.")
    if not include_seasonal_priors:
        logger.info("Excluded direct season/climatology priors; forecast-hour timing remains available.")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    pos_weight_capped = min(pos_weight, 10.0)
    logger.info("Training LightGBM  (raw pos_weight=%.1f  capped=%.1f) …", pos_weight, pos_weight_capped)

    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=pos_weight_capped,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )

    raw_probs = model.predict_proba(X_eval)[:, 1]

    auc_roc = roc_auc_score(y_eval, raw_probs)
    auc_pr  = average_precision_score(y_eval, raw_probs)
    logger.info("Val AUC-ROC: %.4f   AUC-PR: %.4f", auc_roc, auc_pr)

    # ── Isotonic calibration ─────────────────────────────────────────────────
    calibrator = IsotonicRegression(out_of_bounds="clip")
    cal_raw_probs = model.predict_proba(X_cal)[:, 1]
    calibrator.fit(cal_raw_probs, y_cal)

    cal_probs = calibrator.predict(raw_probs)
    auc_roc_cal = roc_auc_score(y_eval, cal_probs)
    auc_pr_cal  = average_precision_score(y_eval, cal_probs)
    logger.info("After calibration — AUC-ROC: %.4f   AUC-PR: %.4f", auc_roc_cal, auc_pr_cal)

    # ── Reliability check (binned) ───────────────────────────────────────────
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(cal_probs, bins) - 1
    logger.info("Reliability (predicted → observed):")
    for i in range(len(bins) - 1):
        mask = bin_idx == i
        if mask.sum() > 0:
            obs = y_eval[mask].mean()
            logger.info("  %.1f–%.1f%%  →  observed %.1f%%  (n=%d)",
                        bins[i]*100, bins[i+1]*100, obs*100, mask.sum())

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "lgbm":       model,
        "calibrator": calibrator,
        "features":   features,
        "trained_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "val_auc_roc": round(auc_roc_cal, 4),
        "val_auc_pr":  round(auc_pr_cal, 4),
        "validation": "date_grouped",
        "include_latlon": include_latlon,
        "include_seasonal_priors": include_seasonal_priors,
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Model saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/training"))
    parser.add_argument("--out",  type=Path, default=Path("models/tornado_lgbm.pkl"))
    parser.add_argument(
        "--include-latlon",
        action="store_true",
        help="Allow raw lat/lon as model features. Default excludes them to reduce climatology overfit.",
    )
    parser.add_argument(
        "--include-seasonal-priors",
        action="store_true",
        help="Allow doy_sin/doy_cos/climo_freq as model features. Default excludes them to reduce broad climatology blobs.",
    )
    args = parser.parse_args()
    train(
        args.data,
        args.out,
        include_latlon=args.include_latlon,
        include_seasonal_priors=args.include_seasonal_priors,
    )
