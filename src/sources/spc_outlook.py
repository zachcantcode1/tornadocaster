"""SPC convective outlook GeoJSON source adapter."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


_DAY1_PRODUCTS = {
    "cat": "day1otlk_cat.nolyr.geojson",
    "tornado": "day1otlk_torn.nolyr.geojson",
    "hail": "day1otlk_hail.nolyr.geojson",
    "wind": "day1otlk_wind.nolyr.geojson",
}


@dataclass(frozen=True)
class SpcOutlookPolygon:
    dn: int
    label: str
    label2: str
    fill: str
    stroke: str
    geometry: BaseGeometry


@dataclass(frozen=True)
class SpcOutlook:
    product: str
    url: str
    valid_iso: str | None
    expire_iso: str | None
    issue_iso: str | None
    forecaster: str | None
    polygons: tuple[SpcOutlookPolygon, ...]

    @property
    def product_label(self) -> str:
        if self.product == "cat":
            return "Categorical"
        return self.product.title()

    @property
    def run_label(self) -> str:
        valid = f" valid {_format_chicago_time(self.valid_iso)}" if self.valid_iso else ""
        return f"SPC Day 1 {self.product_label}{valid}"

    @property
    def valid_period_label(self) -> str:
        if not self.valid_iso:
            return self.run_label
        valid = _format_chicago_time(self.valid_iso)
        if not self.expire_iso:
            return f"SPC Day 1 {self.product_label} valid {valid}"
        expire = _format_chicago_time(self.expire_iso)
        return f"SPC Day 1 {self.product_label} valid {valid} - {expire}"


class SpcOutlookSource:
    """Fetch current SPC Day 1 outlook polygons."""

    def __init__(self, base_url: str = "https://www.spc.noaa.gov/products/outlook", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def day1_url(self, product: str) -> str:
        try:
            filename = _DAY1_PRODUCTS[product]
        except KeyError as exc:
            valid = ", ".join(sorted(_DAY1_PRODUCTS))
            raise ValueError(f"Unsupported SPC Day 1 product {product!r}. Expected one of: {valid}") from exc
        return f"{self.base_url}/{filename}"

    async def fetch_day1(self, product: str = "cat") -> SpcOutlook:
        url = self.day1_url(product)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
        return parse_spc_outlook_geojson(response.json(), product=product, url=url)


def parse_spc_outlook_geojson(data: dict[str, Any], product: str, url: str = "") -> SpcOutlook:
    polygons: list[SpcOutlookPolygon] = []
    metadata: dict[str, Any] = {}

    for feature in data.get("features", []):
        properties = feature.get("properties") or {}
        geometry_data = feature.get("geometry")
        if not geometry_data:
            continue
        if not metadata:
            metadata = properties
        polygons.append(
            SpcOutlookPolygon(
                dn=int(properties.get("DN", 0)),
                label=str(properties.get("LABEL", "")),
                label2=str(properties.get("LABEL2", "")),
                fill=str(properties.get("fill", "#000000")),
                stroke=str(properties.get("stroke", "#000000")),
                geometry=shape(geometry_data),
            )
        )

    return SpcOutlook(
        product=product,
        url=url,
        valid_iso=_optional_str(metadata.get("VALID_ISO")),
        expire_iso=_optional_str(metadata.get("EXPIRE_ISO")),
        issue_iso=_optional_str(metadata.get("ISSUE_ISO")),
        forecaster=_optional_str(metadata.get("FORECASTER")),
        polygons=tuple(sorted(polygons, key=lambda item: item.dn)),
    )


def _optional_str(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _format_chicago_time(value: str) -> str:
    dt = datetime.fromisoformat(value)
    local = dt.astimezone(ZoneInfo("America/Chicago"))
    return f"{local.strftime('%I').lstrip('0')}:{local:%M %p %Z %b} {local.day}"
