from datetime import date

import numpy as np

from src.analysis.probability import nearest_probability, probability_summary
from src.sources.nadocast import NadocastRequest, NadocastSource
from src.visualization.plot_forecast import _legend_entries


def test_expected_filename_for_2022_tornado_run():
    source = NadocastSource()
    request = NadocastRequest(
        run_date=date(2024, 5, 16),
        cycle=0,
        hazard="tornado",
        model_set="2022",
    )

    assert source.expected_filename(request) == (
        "nadocast_2022_models_conus_tornado_20240516_t00z_f12-35.grib2"
    )


def test_expected_filename_for_calibrated_2024_run():
    source = NadocastSource()
    request = NadocastRequest(
        run_date=date(2025, 10, 2),
        cycle=0,
        hazard="wind_adj",
        model_set="2024",
        calibrated=True,
    )

    assert source.expected_filename(request) == (
        "nadocast_2024_preliminary_models_conus_wind_adj_abs_calib_"
        "20251002_t00z_f12-35.grib2"
    )


def test_probability_summary_counts_thresholds():
    grid = np.array([[0.001, 0.02, 0.051], [0.11, 0.16, 0.31]], dtype=np.float32)

    summary = probability_summary(grid)

    assert summary.max_probability == float(np.max(grid))
    assert summary.cells_ge_2pct == 5
    assert summary.cells_ge_5pct == 4
    assert summary.cells_ge_10pct == 3
    assert summary.cells_ge_15pct == 2
    assert summary.cells_ge_30pct == 1


def test_nearest_probability_normalizes_longitude_domain():
    probability = np.array([[0.02, 0.05], [0.10, 0.15]], dtype=np.float32)
    latitude = np.array([[35.0, 35.0], [36.0, 36.0]], dtype=np.float32)
    longitude = np.array([[260.0, 261.0], [260.0, 261.0]], dtype=np.float32)

    assert np.isclose(nearest_probability(probability, latitude, longitude, 36.1, -99.1), 0.15)


def test_legend_entries_stop_at_highest_plotted_bin():
    colors, labels = _legend_entries(0.035)

    assert len(colors) == 3
    assert labels == ["1%", "2%", "3%"]
