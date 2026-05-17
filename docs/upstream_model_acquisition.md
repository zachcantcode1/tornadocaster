# Upstream Model + Feature Contract Acquisition

This project can derive model paths and canonical feature order directly from the Nadocast upstream repo without pulling the full large model tree.

## 1) Minimal sparse upstream checkout

```bash
cd /Users/zach/Documents/tornadocaster
mkdir -p nadocast_upstream
cd nadocast_upstream
git init
git remote add origin https://github.com/brianhempel/nadocast.git
git fetch --depth=1 --filter=blob:none origin master
git checkout -b master --track origin/master
git sparse-checkout init --no-cone
cat > .git/info/sparse-checkout <<'EOF'
README.md
lib/Grid130.jl
models/href_prediction/HREFPrediction.jl
models/href_mid_2018_forward/HREF.jl
models/href_mid_2018_forward/features_2021v2_mean_prob_computed_climatology_blurs_grads_n=2005.txt
EOF
git checkout master
```

## 2) Extract contract files

```bash
cd /Users/zach/Documents/tornadocaster
python3 tools/extract_upstream_model_contract.py
```

Outputs:
- `artifacts/upstream/model_contract.json`
- `artifacts/upstream/features_order_2005.txt`

## 3) Optionally export one model file only

This pulls just the selected model blob (not the full `models/` tree):

```bash
python3 tools/extract_upstream_model_contract.py --export-model tornado:f13-24
```

Allowed windows:
- `f2-13`
- `f13-24`
- `f24-35`

Examples:
- `--export-model hail:f2-13`
- `--export-model sig_tornado:f24-35`

