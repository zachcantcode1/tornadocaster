import re
import httpx
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GribIndexRecord:
    index: int
    offset: int
    date_str: str
    variable: str
    level: str
    forecast: str
    next_offset: Optional[int] = None

class NOAAIndexFetcher:
    """Fetches and parses NOAA .idx files for zero-footprint GRIB2 byte-range pulls."""

    def __init__(self, timeout: float = 10.0):
        self.client = httpx.AsyncClient(timeout=timeout)

    async def fetch_idx_file(self, idx_url: str) -> str:
        """Fetch the raw text of a .idx file."""
        logger.info(f"Fetching IDX file: {idx_url}")
        response = await self.client.get(idx_url)
        response.raise_for_status()
        return response.text

    def parse_idx_text(self, idx_text: str) -> List[GribIndexRecord]:
        """
        Parse NOAA .idx text into records. 
        Format example: 1:0:d=2024051700:CAPE:surface:anl:
        """
        records: List[GribIndexRecord] = []
        lines = [line.strip() for line in idx_text.strip().split("\n") if line.strip()]

        for line in lines:
            # NOAA idx lines are colon-delimited, but some fields can include colons.
            # We only need the first six fields for routing byte ranges.
            parts = line.split(":")
            if len(parts) < 6:
                logger.warning("Skipping malformed idx line: %s", line)
                continue

            try:
                index = int(parts[0])
                offset = int(parts[1])
            except ValueError:
                logger.warning("Skipping idx line with non-integer index/offset: %s", line)
                continue

            date_str = parts[2]
            variable = parts[3]
            level = parts[4]
            forecast = parts[5]
            records.append(GribIndexRecord(index, offset, date_str, variable, level, forecast))

        # Calculate next_offset for byte-range targeting.
        for i in range(len(records) - 1):
            records[i].next_offset = records[i + 1].offset

        return records

    def find_record(
        self,
        records: List[GribIndexRecord],
        variable: str,
        level: Optional[str] = None,
    ) -> GribIndexRecord:
        """Return the first idx record matching variable and optional level."""
        for record in records:
            if record.variable != variable:
                continue
            if level is not None and record.level != level:
                continue
            return record
        level_desc = f" and level={level}" if level is not None else ""
        raise ValueError(f"Could not find idx record for variable={variable}{level_desc}")

    async def fetch_byte_range(self, grib_url: str, start: int, end: Optional[int]) -> bytes:
        """Fetch a specific byte range from a remote GRIB2 file."""
        if end is not None and end <= start:
            raise ValueError(f"Invalid byte range requested: start={start}, end={end}")

        headers = {}
        if end is not None:
            headers["Range"] = f"bytes={start}-{end - 1}"
        else:
            headers["Range"] = f"bytes={start}-"

        logger.info(f"Fetching byte range {headers['Range']} from {grib_url}")
        response = await self.client.get(grib_url, headers=headers)
        response.raise_for_status()
        return response.content

    async def fetch_record_dataset(self, grib_url: str, record: GribIndexRecord) -> xr.Dataset:
        """Fetch one idx record as an in-memory byte-range decode via temporary file."""
        grib_bytes = await self.fetch_byte_range(grib_url, record.offset, record.next_offset)
        fd, temp_path = tempfile.mkstemp(suffix=".grib2")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(grib_bytes)
            return xr.open_dataset(temp_path, engine="cfgrib")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def fetch_named_fields(
        self,
        grib_url: str,
        records: List[GribIndexRecord],
        specs: list[tuple[str, str, str]],
    ) -> dict[str, xr.DataArray]:
        """
        Fetch multiple fields by (output_name, variable, level).
        Returns dict of output_name -> DataArray.
        """
        out: dict[str, xr.DataArray] = {}
        for output_name, variable, level in specs:
            try:
                record = self.find_record(records, variable=variable, level=level)
            except ValueError:
                continue
            ds = await self.fetch_record_dataset(grib_url, record)
            if not ds.data_vars:
                continue
            first_var = list(ds.data_vars.values())[0]
            out[output_name] = first_var
        return out

    async def close(self):
        await self.client.aclose()
