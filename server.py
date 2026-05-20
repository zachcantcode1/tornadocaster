"""FastAPI server for the Tora Weather NADOCast viewer."""
from __future__ import annotations

import asyncio
import csv
import logging
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.analysis.probability import nearest_probability, probability_summary
from src.sources.nadocast import NadocastRequest, NadocastSource
from src.visualization.plot_forecast import plot_conus_forecast

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Tora Weather")

_MAP_PATH = Path("cache/forecast.png")
_ZCTA_PATH = Path("data/2024_Gaz_zcta_national.txt")
_ZCTA_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_zcta_national.zip"
_CACHE_TTL = 3600  # seconds
_WEB_MAP_MIN_PROBABILITY = 0.01
_WEB_CONTOUR_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60, 1.0]
_cache: dict = {}
_cache_lock = asyncio.Lock()
_zcta_cache: dict[str, dict] | None = None


async def _build_forecast() -> dict:
    source = NadocastSource()
    request = await source.find_latest(NadocastRequest(hazard="tornado", model_set="2022"))
    grid = await source.fetch_grid(request)
    summary = probability_summary(grid.probability)

    _MAP_PATH.parent.mkdir(exist_ok=True)
    plot_conus_forecast(
        grid.latitude,
        grid.longitude,
        grid.probability,
        title=f"NADOCast {grid.hazard_label}",
        subtitle=grid.run_label,
        output_path=str(_MAP_PATH),
        map_style="dark",
    )

    updated = datetime.now(timezone.utc)
    cycle = grid.request.cycle
    run_date = grid.request.run_date
    return {
        "run_cycle": f"{cycle:02d}Z {run_date.strftime('%m/%d')}" if cycle is not None and run_date else "—",
        "run_label": grid.run_label,
        "max_probability": f"{summary.max_probability:.1%}",
        "risk_area": summary.cells_ge_2pct,
        "window": grid.request.window or "—",
        "data_source": "NADOCast",
        "source_url": grid.url,
        "hazard": grid.request.hazard,
        "updated_iso": updated.isoformat(),
        "map_url": f"/api/map?t={int(updated.timestamp())}",
        "probability_areas": _probability_areas(grid.latitude, grid.longitude, grid.probability),
        "probability_points": _probability_points(grid.latitude, grid.longitude, grid.probability),
        "probability_threshold": _WEB_MAP_MIN_PROBABILITY,
        "_latitude": grid.latitude,
        "_longitude": grid.longitude,
        "_probability": grid.probability,
        "_fetched_at": updated.timestamp(),
    }


def _probability_points(latitude: np.ndarray, longitude: np.ndarray, probability: np.ndarray) -> list[dict]:
    """Return web-map probability cells as compact lat/lon point data."""
    prob = np.asarray(probability, dtype=np.float32)
    mask = np.isfinite(prob) & (prob >= _WEB_MAP_MIN_PROBABILITY)
    if not np.any(mask):
        return []

    lat = np.asarray(latitude, dtype=np.float32)
    lon = np.asarray(longitude, dtype=np.float32)
    lon = np.where(lon > 180, lon - 360, lon)
    rows, cols = np.where(mask)

    return [
        {
            "lat": round(float(lat[row, col]), 4),
            "lon": round(float(lon[row, col]), 4),
            "p": round(float(prob[row, col]), 4),
        }
        for row, col in zip(rows, cols)
    ]


def _probability_areas(latitude: np.ndarray, longitude: np.ndarray, probability: np.ndarray) -> dict:
    """Return filled probability contours as GeoJSON for web map rendering."""
    prob = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(prob).any() or float(np.nanmax(prob)) < _WEB_MAP_MIN_PROBABILITY:
        return {"type": "FeatureCollection", "features": []}

    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    lon = np.where(lon > 180, lon - 360, lon)
    z = np.ma.masked_invalid(prob)
    generator = contour_generator(x=lon, y=lat, z=z)
    features = []

    for lower, upper in zip(_WEB_CONTOUR_LEVELS[:-1], _WEB_CONTOUR_LEVELS[1:]):
        if float(np.nanmax(prob)) < lower:
            break
        points_chunks, offsets_chunks = generator.filled(lower, upper)
        for points, offsets in zip(points_chunks, offsets_chunks):
            for start, end in zip(offsets[:-1], offsets[1:]):
                ring = points[start:end]
                if len(ring) < 4:
                    continue
                coords = _geojson_ring(ring)
                if len(coords) < 4:
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "lower": lower,
                            "upper": upper,
                            "label": _probability_label(lower, upper),
                            "color": _probability_color(lower),
                            "opacity": _probability_opacity(lower),
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords],
                        },
                    }
                )

    return {"type": "FeatureCollection", "features": features}


def _geojson_ring(points: np.ndarray) -> list[list[float]]:
    coords = [[round(float(lon), 4), round(float(lat), 4)] for lon, lat in points]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _probability_label(lower: float, upper: float) -> str:
    return f"{lower:.0%}-{upper:.0%}"


def _probability_color(probability: float) -> str:
    if probability >= 0.30:
        return "#c040c8"
    if probability >= 0.15:
        return "#c83228"
    if probability >= 0.10:
        return "#e87820"
    if probability >= 0.05:
        return "#f0c040"
    if probability >= 0.03:
        return "#32cd32"
    if probability >= 0.02:
        return "#76b041"
    return "#8b949e"


def _probability_opacity(probability: float) -> float:
    if probability >= 0.10:
        return 0.38
    if probability >= 0.03:
        return 0.34
    if probability >= 0.02:
        return 0.30
    return 0.18


async def get_forecast() -> dict:
    async with _cache_lock:
        now = datetime.now(timezone.utc).timestamp()
        age = now - _cache.get("_fetched_at", 0.0)
        if _cache and age < _CACHE_TTL:
            return _cache
        logger.info("Refreshing NADOCast forecast cache")
        data = await _build_forecast()
        _cache.clear()
        _cache.update(data)
    return _cache


def get_zcta_lookup() -> dict[str, dict]:
    global _zcta_cache
    if _zcta_cache is not None:
        return _zcta_cache
    if not _ZCTA_PATH.exists():
        _download_zcta_gazetteer()

    lookup: dict[str, dict] = {}
    with _ZCTA_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            normalized = {key.strip(): value.strip() for key, value in row.items() if key is not None}
            zcta = normalized.get("GEOID", "")
            lat = normalized.get("INTPTLAT")
            lon = normalized.get("INTPTLONG")
            if len(zcta) == 5 and lat and lon:
                lookup[zcta] = {
                    "zip": zcta,
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                }
    _zcta_cache = lookup
    return lookup


def _download_zcta_gazetteer() -> None:
    logger.info("Downloading Census ZCTA gazetteer to %s", _ZCTA_PATH)
    _ZCTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    archive_path = _ZCTA_PATH.with_suffix(".zip")
    urllib.request.urlretrieve(_ZCTA_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extract(_ZCTA_PATH.name, _ZCTA_PATH.parent)


@app.get("/api/forecast")
async def forecast_endpoint() -> dict:
    data = await get_forecast()
    return {k: v for k, v in data.items() if not k.startswith("_")}


@app.get("/api/zipcode/{zipcode}")
async def zipcode_endpoint(zipcode: str) -> dict:
    zcta = zipcode.strip()
    if not (len(zcta) == 5 and zcta.isdigit()):
        raise HTTPException(status_code=400, detail="ZIP code must be 5 digits")

    try:
        location = get_zcta_lookup()[zcta]
    except KeyError:
        raise HTTPException(status_code=404, detail="ZIP code not found in Census ZCTA gazetteer")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    forecast = await get_forecast()
    probability = nearest_probability(
        forecast["_probability"],
        forecast["_latitude"],
        forecast["_longitude"],
        location["lat"],
        location["lon"],
    )

    return {
        **location,
        "probability": round(probability, 4),
        "probability_label": f"{probability:.1%}",
        "run_label": forecast["run_label"],
    }


@app.get("/api/map")
async def map_endpoint():
    await get_forecast()
    if not _MAP_PATH.exists():
        raise HTTPException(status_code=503, detail="Map not yet generated")
    return FileResponse(_MAP_PATH, media_type="image/png")


app.mount("/", StaticFiles(directory="website", html=True), name="static")
