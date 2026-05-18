#!/usr/bin/env bash
# remote_bootstrap.sh — Run this on a fresh DigitalOcean Ubuntu droplet.
# Clones tornadocaster, installs deps, and kicks off the dataset build
# inside a tmux session so it survives SSH disconnects.
#
# Usage (from your local machine after SSH-ing in):
#   bash <(curl -fsSL https://raw.githubusercontent.com/zachcantcode1/tornadocaster/main/scripts/remote_bootstrap.sh)
#
# Or after uploading:
#   bash scripts/remote_bootstrap.sh [--years "2021 2022 2023"] [--out data/training]

set -euo pipefail

REPO_URL="https://github.com/zachcantcode1/tornadocaster.git"
REPO_DIR="$HOME/tornadocaster"
YEARS="${YEARS:-2021 2022 2023}"
OUT_DIR="${OUT_DIR:-data/training}"

# ── Parse optional flags ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --years) YEARS="$2"; shift 2 ;;
        --out)   OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "======================================================"
echo "  Tornadocaster Dataset Build Bootstrap"
echo "  Years : $YEARS"
echo "  Output: $REPO_DIR/$OUT_DIR"
echo "======================================================"

# ── System packages ──────────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv \
    git tmux \
    libeccodes-dev libeccodes-tools \
    build-essential \
    2>&1 | tail -5

# ── Clone repo ───────────────────────────────────────────────────────────────
echo "[2/5] Cloning repo..."
if [[ -d "$REPO_DIR/.git" ]]; then
    echo "  Repo already present — pulling latest..."
    git -C "$REPO_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# ── Python virtual environment ───────────────────────────────────────────────
echo "[3/5] Setting up Python venv..."
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip -q
pip install -q \
    numpy pandas scipy xarray pyarrow \
    cfgrib eccodes \
    aiohttp \
    lightgbm scikit-learn \
    matplotlib cartopy \
    2>&1 | tail -5

# Install any extras from requirements.txt that aren't covered above
pip install -q -r requirements.txt 2>&1 | tail -3 || true

# ── Verify key import ────────────────────────────────────────────────────────
echo "[4/5] Verifying imports..."
python3 -c "
import cfgrib, xarray, numpy, pandas, aiohttp, lightgbm, pyarrow
print('  All key packages OK')
"

# ── Launch build in tmux ─────────────────────────────────────────────────────
echo "[5/5] Launching dataset build in tmux session 'build'..."
SESSION="build"

# Convert space-separated years to --years args
YEAR_ARGS=$(echo "$YEARS" | tr ' ' '\n' | xargs -I{} echo -n "{} ")

tmux new-session -d -s "$SESSION" -x 220 -y 50 \
    "source $REPO_DIR/.venv/bin/activate && \
     cd $REPO_DIR && \
     python -m src.training.build_dataset --years $YEAR_ARGS --out $OUT_DIR \
     2>&1 | tee build.log; \
     echo '=== BUILD COMPLETE ===' >> build.log; \
     echo 'Done. Check build.log for results.'"

echo ""
echo "======================================================"
echo "  Build is running in tmux session '$SESSION'"
echo ""
echo "  To watch progress:"
echo "    tmux attach -t $SESSION"
echo "  To detach without stopping:"
echo "    Ctrl+B then D"
echo ""
echo "  To tail logs without attaching:"
echo "    tail -f $REPO_DIR/build.log"
echo ""
echo "  When done, run this from your LOCAL machine to download shards:"
echo "    rsync -avz --progress root@<DROPLET_IP>:$REPO_DIR/$OUT_DIR/ data/training/"
echo "======================================================"
