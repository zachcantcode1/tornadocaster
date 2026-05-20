# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Tornado Caster is a NADOCast-first severe weather guidance viewer. The project
does not try to out-model NADOCast. It fetches published NADOCast GRIB2
probability grids, decodes them, and adds product-layer value through custom
maps, summaries, local lookups, overlays, and run-to-run interpretation.

## Commands

```bash
# Fetch latest matching NADOCast grid and render forecast.png
python forecast_now.py

# Fetch a specific historical run
python forecast_now.py --date 20240516 --cycle 0 --hazard tornado --model-set 2022

# Decode and summarize without rendering
python forecast_now.py --date 20240516 --cycle 0 --hazard tornado --model-set 2022 --summary-only

# Run tests
pytest tests/
```

## Architecture

**NADOCast Source (`src/sources/nadocast.py`)** resolves public NADOCast archive
paths, lists run files, downloads GRIB2 files, decodes them with `cfgrib`, and
normalizes percent grids into `0.0-1.0` probability arrays.

**Analysis (`src/analysis/probability.py`)** contains small helpers for summary
statistics and nearest-point probability sampling.

**Visualization (`src/visualization/plot_forecast.py`)** renders CONUS maps with
SPC/NADOCast-style probability thresholds.

**NOAA Ingestion (`src/ingestion/noaa_fetcher.py`)** is retained as a useful
future overlay/source utility, but it is no longer the primary forecast engine.

## Product Direction

Prefer changes that improve the NADOCast viewer workflow:

- custom overlays for SPC outlooks, watches/warnings, radar/MRMS, reports, cities, counties, and roads
- county/city/point probability extraction
- run-to-run trend detection
- threshold polygons and alerting
- archive browsing and verification

Avoid rebuilding a local ML training or inference stack unless it is clearly
scoped as a separate research/fallback adapter.
