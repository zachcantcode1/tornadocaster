"""Probability-grid analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbabilitySummary:
    max_probability: float
    mean_probability: float
    cells_ge_2pct: int
    cells_ge_5pct: int
    cells_ge_10pct: int
    cells_ge_15pct: int
    cells_ge_30pct: int


def probability_summary(probability: np.ndarray) -> ProbabilitySummary:
    arr = np.asarray(probability, dtype=np.float32)
    return ProbabilitySummary(
        max_probability=float(np.nanmax(arr)),
        mean_probability=float(np.nanmean(arr)),
        cells_ge_2pct=int(np.count_nonzero(arr >= 0.02)),
        cells_ge_5pct=int(np.count_nonzero(arr >= 0.05)),
        cells_ge_10pct=int(np.count_nonzero(arr >= 0.10)),
        cells_ge_15pct=int(np.count_nonzero(arr >= 0.15)),
        cells_ge_30pct=int(np.count_nonzero(arr >= 0.30)),
    )


def nearest_probability(
    probability: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    point_lat: float,
    point_lon: float,
) -> float:
    """Return the nearest grid-cell probability for a point."""
    lon = np.where(longitude > 180, longitude - 360, longitude)
    distances = (latitude - point_lat) ** 2 + (lon - point_lon) ** 2
    y, x = np.unravel_index(np.nanargmin(distances), distances.shape)
    return float(probability[y, x])
