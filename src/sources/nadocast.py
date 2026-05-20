"""NADOCast GRIB2 source adapter."""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Iterable
from urllib.parse import unquote

import httpx
import numpy as np
import xarray as xr

_MODEL_PREFIXES = {
    "2022": "nadocast_2022_models_conus",
    "2024": "nadocast_2024_preliminary_models_conus",
}

_DEFAULT_WINDOWS = {
    0: "f12-35",
    12: "f02-23",
    18: "f01-17",
}


@dataclass(frozen=True)
class NadocastRequest:
    run_date: date | None = None
    cycle: int | None = None
    hazard: str = "tornado"
    model_set: str = "2022"
    calibrated: bool = False
    window: str | None = None
    filename: str | None = None


@dataclass(frozen=True)
class NadocastGrid:
    request: NadocastRequest
    url: str
    variable_name: str
    units: str
    latitude: np.ndarray
    longitude: np.ndarray
    probability: np.ndarray
    raw_values: np.ndarray
    attrs: dict

    @property
    def run_label(self) -> str:
        cycle = "" if self.request.cycle is None else f" {self.request.cycle:02d}Z"
        window = f" {self.request.window}" if self.request.window else ""
        date_label = self.request.run_date.isoformat() if self.request.run_date else "unknown date"
        model = f"NADOCast {self.request.model_set}"
        calib = " abs_calib" if self.request.calibrated else ""
        return f"{model}{calib} | {date_label}{cycle}{window}"

    @property
    def hazard_label(self) -> str:
        return self.request.hazard.replace("_", " ").title()


class NadocastSource:
    """Fetch and decode public NADOCast GRIB2 grids."""

    def __init__(self, base_url: str = "http://data.nadocast.com", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def list_run_files(self, run_date: date, cycle: int) -> list[str]:
        url = self.directory_url(run_date, cycle)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
        links = re.findall(r'href=["\']([^"\']+\.grib2)["\']', response.text)
        return sorted({os.path.basename(unquote(link)) for link in links})

    async def find_latest(self, request: NadocastRequest, max_days: int = 14) -> NadocastRequest:
        cycles = [18, 12, 0] if request.cycle is None else [request.cycle]
        start = request.run_date or date.today()
        for offset in range(max_days + 1):
            candidate_date = start - timedelta(days=offset)
            for cycle in cycles:
                candidate = replace(request, run_date=candidate_date, cycle=cycle)
                try:
                    filename = await self.resolve_filename(candidate)
                except (httpx.HTTPError, FileNotFoundError):
                    continue
                return replace(candidate, filename=filename)
        raise FileNotFoundError(
            f"No NADOCast {request.hazard!r} GRIB2 found in the last {max_days} days."
        )

    async def fetch_grid(self, request: NadocastRequest) -> NadocastGrid:
        resolved = await self.resolve_request(request)
        url = self.file_url(resolved)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
        ds = _decode_grib2(response.content)
        try:
            if not ds.data_vars:
                raise ValueError(f"NADOCast GRIB2 has no data variables: {url}")
            variable_name = next(iter(ds.data_vars))
            da = ds[variable_name]
            raw_values = np.asarray(da.values, dtype=np.float32)
            probability = _to_probability_fraction(raw_values, da.attrs.get("units", ""))
            lon = np.asarray(ds.longitude.values, dtype=np.float32)
            lon = np.where(lon > 180, lon - 360, lon)
            lat = np.asarray(ds.latitude.values, dtype=np.float32)
            return NadocastGrid(
                request=resolved,
                url=url,
                variable_name=variable_name,
                units=str(da.attrs.get("units", "")),
                latitude=lat,
                longitude=lon,
                probability=probability,
                raw_values=raw_values,
                attrs=dict(da.attrs),
            )
        finally:
            ds.close()

    async def resolve_request(self, request: NadocastRequest) -> NadocastRequest:
        if request.run_date is None or request.cycle is None:
            return await self.find_latest(request)
        filename = request.filename or await self.resolve_filename(request)
        window = request.window or _window_from_filename(filename)
        return replace(request, filename=filename, window=window)

    async def resolve_filename(self, request: NadocastRequest) -> str:
        if request.filename:
            return request.filename
        if request.run_date is None or request.cycle is None:
            raise ValueError("run_date and cycle are required to resolve a NADOCast filename")

        files = await self.list_run_files(request.run_date, request.cycle)
        matches = list(_matching_files(files, request))
        if not matches:
            expected = self.expected_filename(request)
            if expected in files:
                return expected
            raise FileNotFoundError(
                f"No NADOCast file matched hazard={request.hazard!r}, "
                f"model_set={request.model_set!r}, calibrated={request.calibrated} "
                f"at {self.directory_url(request.run_date, request.cycle)}"
            )
        return matches[0]

    def expected_filename(self, request: NadocastRequest) -> str:
        if request.run_date is None or request.cycle is None:
            raise ValueError("run_date and cycle are required to build a NADOCast filename")
        prefix = _MODEL_PREFIXES[request.model_set]
        run = request.run_date.strftime("%Y%m%d")
        cycle = f"t{request.cycle:02d}z"
        window = request.window or _DEFAULT_WINDOWS[request.cycle]
        calib = "_abs_calib" if request.calibrated else ""
        return f"{prefix}_{request.hazard}{calib}_{run}_{cycle}_{window}.grib2"

    def directory_url(self, run_date: date, cycle: int) -> str:
        yyyymm = run_date.strftime("%Y%m")
        yyyymmdd = run_date.strftime("%Y%m%d")
        return f"{self.base_url}/{yyyymm}/{yyyymmdd}/t{cycle}z/"

    def file_url(self, request: NadocastRequest) -> str:
        if request.run_date is None or request.cycle is None or request.filename is None:
            raise ValueError("run_date, cycle, and filename are required to build file URL")
        return f"{self.directory_url(request.run_date, request.cycle)}{request.filename}"


def _matching_files(files: Iterable[str], request: NadocastRequest) -> Iterable[str]:
    prefix = _MODEL_PREFIXES[request.model_set]
    hazard_prefix = f"{prefix}_{request.hazard}_"
    calib_prefix = f"{prefix}_{request.hazard}_abs_calib_"
    window = request.window

    for filename in files:
        if not filename.endswith(".grib2"):
            continue
        if request.calibrated:
            if not filename.startswith(calib_prefix):
                continue
        elif not filename.startswith(hazard_prefix) or "_abs_calib_" in filename:
            continue
        if window and f"_{window}.grib2" not in filename:
            continue
        yield filename


def _window_from_filename(filename: str) -> str | None:
    match = re.search(r"_(f\d{2}-\d{2})\.grib2$", filename)
    return match.group(1) if match else None


def _decode_grib2(content: bytes) -> xr.Dataset:
    fd, path = tempfile.mkstemp(suffix=".grib2")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        ds = xr.open_dataset(path, engine="cfgrib")
        ds.load()
        return ds
    finally:
        if os.path.exists(path):
            os.remove(path)


def _to_probability_fraction(values: np.ndarray, units: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if units == "%" or np.nanmax(arr) > 1.0:
        arr = arr / 100.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)
