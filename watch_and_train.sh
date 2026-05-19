#!/usr/bin/env bash
# watch_and_train.sh — polls build_v2.log until the build finishes,
# then automatically runs training. Run this once and leave it.

PYTHON="/c/Users/zachm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
WORKDIR="/c/Users/zachm/Documents/Tornado Caster/tornadocaster"
BUILD_LOG="$WORKDIR/logs/build_v2.log"
TRAIN_LOG="$WORKDIR/logs/train_v2.log"

cd "$WORKDIR"

echo "$(date) — Watching build log for completion..."

while true; do
    # Build writes "Done — N shards written" on completion
    if grep -q "Done —" "$BUILD_LOG" 2>/dev/null; then
        echo "$(date) — Build finished. Starting training..."
        break
    fi
    sleep 60
done

echo "$(date) — Training LightGBM on data/training_v2 ..." | tee "$TRAIN_LOG"
"$PYTHON" -m src.training.train \
    --data data/training_v2 \
    --out  models/tornado_lgbm_v2.pkl \
    2>&1 | tee -a "$TRAIN_LOG"

echo ""
echo "=========================================="
echo "Training complete. Model: models/tornado_lgbm_v2.pkl"
echo "Test with:"
echo "  python forecast_now.py --model rrfs --ml --fstart 1 --fend 18 --output forecast_v2.png"
echo "=========================================="
