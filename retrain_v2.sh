#!/usr/bin/env bash
# retrain_v2.sh — wait for build_v2.log to finish, then retrain.
# Run this in a separate terminal if needed, or it will auto-run after build.

PYTHON="/c/Users/zachm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
WORKDIR="/c/Users/zachm/Documents/Tornado Caster/tornadocaster"
LOGFILE="$WORKDIR/logs/train_v2.log"

cd "$WORKDIR"

echo "$(date) — Starting LightGBM training on data/training_v2 ..." | tee -a "$LOGFILE"
"$PYTHON" -m src.training.train \
    --data data/training_v2 \
    --out  models/tornado_lgbm_v2.pkl \
    2>&1 | tee -a "$LOGFILE"

echo "$(date) — Training complete." | tee -a "$LOGFILE"
echo "Model saved to models/tornado_lgbm_v2.pkl"
