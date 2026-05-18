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


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
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
    X = df[FEATURE_COLS].astype(np.float32)   # keep as DataFrame — preserves feature names
    y = df["label"].values.astype(np.int8)
    return X, y


def train(data_dir: Path, out_path: Path) -> None:
    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score

    X, y = load_dataset(data_dir)

    # Stratified split — keep 20% for calibration + evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Convert val to numpy for calibration/metrics (model stays DataFrame-aware)
    X_val_np = X_val.values if hasattr(X_val, "values") else X_val

    # ── LightGBM ─────────────────────────────────────────────────────────────
    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    logger.info("Training LightGBM  (pos_weight=%.1f) …", pos_weight)

    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )

    raw_probs = model.predict_proba(X_val)[:, 1]

    auc_roc = roc_auc_score(y_val, raw_probs)
    auc_pr  = average_precision_score(y_val, raw_probs)
    logger.info("Val AUC-ROC: %.4f   AUC-PR: %.4f", auc_roc, auc_pr)

    # ── Isotonic calibration ─────────────────────────────────────────────────
    mid = len(X_val_np) // 2
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs[:mid], y_val[:mid])

    cal_probs = calibrator.predict(raw_probs[mid:])
    auc_roc_cal = roc_auc_score(y_val[mid:], cal_probs)
    auc_pr_cal  = average_precision_score(y_val[mid:], cal_probs)
    logger.info("After calibration — AUC-ROC: %.4f   AUC-PR: %.4f", auc_roc_cal, auc_pr_cal)

    # ── Reliability check (binned) ───────────────────────────────────────────
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(cal_probs, bins) - 1
    logger.info("Reliability (predicted → observed):")
    for i in range(len(bins) - 1):
        mask = bin_idx == i
        if mask.sum() > 0:
            obs = y_val[mid:][mask].mean()
            logger.info("  %.1f–%.1f%%  →  observed %.1f%%  (n=%d)",
                        bins[i]*100, bins[i+1]*100, obs*100, mask.sum())

    # ── Save ─────────────────────────────────────────────────────────────────
    from src.training.historical_sampler import FEATURE_COLS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "lgbm":       model,
        "calibrator": calibrator,
        "features":   FEATURE_COLS,
        "trained_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "val_auc_roc": round(auc_roc_cal, 4),
        "val_auc_pr":  round(auc_pr_cal, 4),
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Model saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/training"))
    parser.add_argument("--out",  type=Path, default=Path("models/tornado_lgbm.pkl"))
    args = parser.parse_args()
    train(args.data, args.out)
