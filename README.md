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

Render the current official SPC Day 1 categorical outlook on the same custom
CONUS basemap:

```powershell
python forecast_now.py --spc-day1 cat --output spc_day1.png
```

Supported SPC Day 1 products are `cat`, `tornado`, `hail`, and `wind`. These
render as standalone SPC maps rather than overlays on NADOCast.

## IFTTT/Twitter Automation

Generate a map and trigger an IFTTT Webhooks applet:

```powershell
$env:IFTTT_EVENT="your_event_name"
$env:IFTTT_WEBHOOK_KEY="your_ifttt_key"
python post_ifttt.py --product spc-tornado
```

You can also provide the full webhook URL instead of event/key:

```powershell
$env:IFTTT_WEBHOOK_URL="https://maker.ifttt.com/trigger/your_event_name/with/key/your_ifttt_key"
python post_ifttt.py --product nadocast
```

The webhook sends:

- `value1`: post text
- `value2`: public image URL, when provided
- `value3`: local generated image path

Important: IFTTT cannot fetch a PNG from your local filesystem. If your Twitter
applet posts an image, host/upload the generated PNG somewhere public first and
pass that URL:

```powershell
python post_ifttt.py --product spc-tornado --image-url "https://example.com/spc_day1_tornado.png"
```

To upload automatically, create a Cloudinary unsigned upload preset, then set:

```powershell
$env:CLOUDINARY_CLOUD_NAME="your_cloud_name"
$env:CLOUDINARY_UPLOAD_PRESET="your_unsigned_upload_preset"
$env:CLOUDINARY_FOLDER="tornado-caster"
python post_ifttt.py --product nadocast --upload cloudinary
```

The script uploads the generated PNG to Cloudinary, reads the returned public
`secure_url`, and sends that URL to IFTTT as `value2`.

### Autopost Only When New

Use `autopost.py` for scheduled automation. It checks the latest NADOCast run
and SPC Day 1 tornado outlook, compares them to a local state file, and only
posts products that have not already been posted.

```powershell
$env:CLOUDINARY_CLOUD_NAME="dbnaes5op"
$env:CLOUDINARY_UPLOAD_PRESET="tornado_caster_unsigned"
$env:CLOUDINARY_FOLDER="tornado-caster"
$env:IFTTT_NADOCAST_WEBHOOK_URL="https://maker.ifttt.com/trigger/nadocast_post/with/key/..."
$env:IFTTT_SPC_TORNADO_WEBHOOK_URL="https://maker.ifttt.com/trigger/spc_tor_posting/with/key/..."

python autopost.py
```

Safe check without posting:

```powershell
python autopost.py --dry-run
```

Force a repost of the latest products:

```powershell
python autopost.py --force
```

Recommended Windows Task Scheduler setup:

- Trigger: every 10-15 minutes
- Program:
  `C:\Users\zachm\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
- Arguments:
  `autopost.py`
- Start in:
  `C:\Users\zachm\Documents\Tornado Caster\tornadocaster`

Use `--dry-run` to generate the image without posting:

```powershell
python post_ifttt.py --product spc-tornado --dry-run
```

## Architecture

```text
src/
  sources/
    nadocast.py       # NADOCast directory resolution, GRIB2 download, decode
    spc_outlook.py    # SPC Day 1 GeoJSON download and polygon parsing
  analysis/
    probability.py    # probability summaries and point sampling
  visualization/
    plot_forecast.py  # CONUS probability map rendering
```

The old model-building and training stack has been removed from the active code
path. Future work should add source adapters and overlays around this simpler
core rather than rebuilding severe-weather ML from scratch.
