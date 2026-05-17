#!/usr/bin/env python3
"""
Extract model and feature-order contract from a sparse Nadocast checkout.

Outputs:
  - artifacts/upstream/model_contract.json
  - artifacts/upstream/features_order_2005.txt

Optional:
  - export one selected model blob from upstream git object database
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path


def parse_models_block(href_prediction_jl: Path) -> list[dict]:
    text = href_prediction_jl.read_text()
    m = re.search(r"models\s*=\s*\[(.*?)\n\]", text, flags=re.DOTALL)
    if not m:
        raise RuntimeError("Could not find `models = [...]` block in HREFPrediction.jl")

    block = m.group(1)
    rows = []
    for tup in re.findall(r"\((.*?)\)\s*,?", block, flags=re.DOTALL):
        strings = re.findall(r'"([^"]+)"', tup)
        if len(strings) < 5:
            continue
        event, grib2_var, f2_13, f13_24, f24_35 = strings[:5]
        rows.append(
            {
                "event": event,
                "grib2_var": grib2_var,
                "f2_13_model_path": f"models/href_mid_2018_forward/{f2_13}",
                "f13_24_model_path": f"models/href_mid_2018_forward/{f13_24}",
                "f24_35_model_path": f"models/href_mid_2018_forward/{f24_35}",
            }
        )
    if not rows:
        raise RuntimeError("Parsed zero model rows from HREFPrediction.jl")
    return rows


def read_feature_order(feature_file: Path) -> list[str]:
    lines = [ln.strip() for ln in feature_file.read_text().splitlines() if ln.strip()]
    # Canonical file format in this repo uses trailing ":" on each feature.
    return [ln[:-1] if ln.endswith(":") else ln for ln in lines]


def git_show_file(repo: Path, ref_path: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "show", f"origin/master:{ref_path}"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git show failed for {ref_path}: {result.stderr.decode(errors='ignore')}"
        )
    out_path.write_bytes(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-repo",
        default="nadocast_upstream",
        help="Path to sparse nadocast upstream checkout.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/upstream",
        help="Output directory for extracted contract files.",
    )
    parser.add_argument(
        "--export-model",
        default="",
        help=(
            "Optional model selector in form <event>:<window>, e.g. tornado:f13-24 "
            "or hail:f2-13. Exports that model file into out-dir/models/."
        ),
    )
    args = parser.parse_args()

    root = Path.cwd()
    upstream = (root / args.upstream_repo).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    href_prediction = upstream / "models/href_prediction/HREFPrediction.jl"
    features_path = (
        upstream
        / "models/href_mid_2018_forward/features_2021v2_mean_prob_computed_climatology_blurs_grads_n=2005.txt"
    )
    if not href_prediction.exists():
        raise FileNotFoundError(f"Missing {href_prediction}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}")

    models = parse_models_block(href_prediction)
    feature_order = read_feature_order(features_path)

    contract = {
        "source_repo": "https://github.com/brianhempel/nadocast",
        "source_ref": "origin/master",
        "feature_count": len(feature_order),
        "feature_file": str(features_path.relative_to(upstream)),
        "models": models,
    }
    (out_dir / "model_contract.json").write_text(json.dumps(contract, indent=2))
    (out_dir / "features_order_2005.txt").write_text("\n".join(feature_order) + "\n")

    if args.export_model:
        try:
            event, window = args.export_model.split(":", 1)
        except ValueError as exc:
            raise ValueError("--export-model must be <event>:<window>") from exc

        window_key = {
            "f2-13": "f2_13_model_path",
            "f13-24": "f13_24_model_path",
            "f24-35": "f24_35_model_path",
        }.get(window)
        if window_key is None:
            raise ValueError("Window must be one of f2-13, f13-24, f24-35")

        row = next((r for r in models if r["event"] == event), None)
        if row is None:
            raise ValueError(f"No event named `{event}` in contract")

        model_relpath = row[window_key]
        out_model = out_dir / "models" / Path(model_relpath).name
        git_show_file(upstream, model_relpath, out_model)

        contract["exported_model"] = {
            "selector": args.export_model,
            "source_path": model_relpath,
            "output_path": str(out_model.relative_to(root)),
        }
        (out_dir / "model_contract.json").write_text(json.dumps(contract, indent=2))

    print(f"Wrote {(out_dir / 'model_contract.json')}")
    print(f"Wrote {(out_dir / 'features_order_2005.txt')}")
    if args.export_model:
        print(f"Exported model for selector {args.export_model}")


if __name__ == "__main__":
    main()
