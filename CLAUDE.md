# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tornadocaster is a Python rebuild of the original Nadocast (Ruby/Julia) tornado forecasting system. It ingests real-time NOAA atmospheric model data (HRRR and RRFS), engineers spatial/temporal features, runs LightGBM ML inference to produce per-grid-cell tornado probabilities, and generates SPC-style forecast discussions via GPT-4.

## Commands

```bash
# Run full pipeline in mock mode (no network)
python main.py

# Generate multi-hour forecast map + PNG
python forecast_now.py [--output IMAGE] [--fstart H] [--fend H] [--model rrfs|hrrr] [--ml]

# Train LightGBM classifier on labeled Parquet dataset
python -m src.training.train --data DIR --out MODEL.pkl

# Debug ingestion
python inspect_nado.py

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_feature_engineering.py -v

# Memory bounds test (full pipeline, <350 MiB overhead target)
pytest tests/test_end_to_end_orchestration.py
```

## Architecture: 5-Phase Pipeline

**Ingestion (`src/ingestion/noaa_fetcher.py`)** — `NOAAIndexFetcher` parses NOAA `.idx` index files to extract byte-range offsets, then fetches only the needed variables from GRIB2 files via HTTP `Range` headers. This avoids downloading multi-gigabyte files. Field catalogs (`FIRST_PASS_FIELD_CATALOG` for HRRR, `RRFS_FIELD_CATALOG` for RRFS) are defined in `main.py`.

**Feature Engineering (`src/features/`)** — Three modules:
- `spatial.py`: Spatial gradient magnitude and rolling-mean windows (25/50/100 mi radii) using numpy/dask with chunked tiling to stay under 16 GB RAM
- `derived.py`: ~40 derived fields replicating original Nadocast Julia math (lapse rates, CAPE×HLCY wind products, mixing ratios)
- `contract_alignment.py`: Maps computed features to the canonical JSON model contract, filling gaps with zeros or temporal proxies

**ML Inference (`src/models/`)** — `lgb_bridge.py` (`JuliaToLightGBMBridge`) loads LightGBM `.txt` model files (converted from Julia). It rejects Julia BSON binaries and falls back to a mock model for testing. `calibrated_predictor.py` wraps LightGBM scores with isotonic regression calibration. Model routing across event types and forecast windows (f2-13, f13-24, f24-35) is driven by a JSON model contract in `model_contract.py`.

**LLM Reporting (`src/llm/llm_report.py`)** — `ForecastDiscussionGenerator` calls OpenAI GPT-4o (temperature=0.2) with peak threat coordinates + LightGBM feature importances to produce deterministic SPC-style meteorological text.

**Training (`src/training/`)** — `build_dataset.py` combines historical NOAA grid data + SPC storm reports into labeled Parquet shards. `train.py` trains the LightGBM binary classifier (tornado/no-tornado) and exports a pickle with model + calibrator + feature names. `historical_sampler.py` handles fetching historical data.

## Key Constraints

- **Memory budget**: Strict <16 GB RAM. Dask chunking tiles large grids. Memory-profiling tests enforce <350 MiB pipeline overhead.
- **Zero-footprint ingestion**: HTTP byte-range requests only — never download full GRIB2 files.
- **Async throughout**: All I/O (ingestion, LLM calls) uses `asyncio` + `httpx`. No blocking I/O.
- **Feature parity with Nadocast**: `test_nadocast_comparison.py` validates math against the original Julia/Ruby outputs. Preserve this when changing derived features.
- **Model portability**: Real models may be unavailable; the system falls back to `build_mock_model()` for development/testing. Julia BSON files are explicitly rejected with an error.

## Test Coverage Map

| Test file | What it guards |
|---|---|
| `test_end_to_end_orchestration.py` | Full pipeline memory bounds |
| `test_feature_engineering.py` | Gradient/mean computations on chunked grids |
| `test_model_inference.py` | LightGBM prediction correctness |
| `test_ingestion_pull.py` | NOAA fetcher (mocked HTTP) |
| `test_binary_model_guard.py` | Julia BSON rejection logic |
| `test_nadocast_comparison.py` | Math parity with legacy code |
| `test_real_data_pipeline.py` | End-to-end with live NOAA data |

## Data Flow

```
NOAA S3 (HRRR/RRFS GRIB2)
  → [Ingestion] .idx parsing + byte-range fetch → xarray Dataset
  → [Feature Engineering] spatial means/gradients + derived fields → feature matrix (~17k features)
  → [ML Inference] LightGBM → per-grid-cell tornado probabilities
  → [Argmax] peak threat coordinates + feature importances
  → [LLM] OpenAI GPT-4o → SPC-style forecast discussion
  → [Output] PNG map + coordinates + probabilities + text
```

## Key Files

- `main.py` — Pipeline orchestration, field catalogs, daemon loop
- `forecast_now.py` — CLI entry point for forecast map generation
- `src/ingestion/noaa_fetcher.py` — NOAA data fetching
- `src/features/derived.py` — Nadocast-parity derived feature math
- `src/features/contract_alignment.py` — Feature→model contract mapping
- `src/models/lgb_bridge.py` — LightGBM model loading and mock fallback
- `src/llm/llm_report.py` — GPT-4 forecast discussion generation
- `src/training/train.py` — Model training pipeline
- `src/visualization/plot_forecast.py` — Cartopy map rendering
