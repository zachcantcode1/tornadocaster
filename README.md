# Tornado Caster

Tornado Caster is now a NADOCast-first severe weather guidance viewer.

Instead of trying to recreate the full NADOCast machine-learning pipeline, this
project treats published NADOCast GRIB2 probability grids as the primary source
of truth and focuses on the product layer around them: custom maps, summaries,
local lookups, overlays, and run-to-run interpretation.

## Current Scope

- Fetch public NADOCast GRIB2 files from `data.nadocast.com`
- Decode probability grids with `cfgrib` and `xarray`
- Normalize NADOCast percent grids into `0.0-1.0` probabilities
- Render CONUS probability maps using the existing SPC/NADOCast-style color ramp
- Provide small analysis helpers for probability summaries and nearest-point lookup

## Usage

Fetch the latest matching NADOCast tornado grid and render `forecast.png`:

```powershell
python forecast_now.py
```

Dark mode is the default map style. Use `--map-style light` for the lighter
basemap.

Fetch a specific run:

```powershell
python forecast_now.py --date 20240516 --cycle 0 --hazard tornado
```

Print a summary without rendering:

```powershell
python forecast_now.py --date 20240516 --cycle 0 --hazard tornado --summary-only
```

Common hazard tokens include `tornado`, `sig_tornado`, `hail`, `sig_hail`,
`wind`, `sig_wind`, and `wind_adj`.

## Architecture

```text
src/
  sources/
    nadocast.py       # NADOCast directory resolution, GRIB2 download, decode
  analysis/
    probability.py    # probability summaries and point sampling
  visualization/
    plot_forecast.py  # CONUS probability map rendering
```

The old model-building and training stack has been removed from the active code
path. Future work should add source adapters and overlays around this simpler
core rather than rebuilding severe-weather ML from scratch.
