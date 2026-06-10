from datetime import date

import numpy as np

from src.analysis.probability import nearest_probability, probability_summary
from src.sources.nadocast import NadocastRequest, NadocastSource
from src.sources.spc_outlook import parse_spc_outlook_geojson
from src.visualization.plot_forecast import _is_spc_intensity_polygon, _legend_entries, _spc_legend_label


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


def test_nadocast_run_label_uses_chicago_time():
    request = NadocastRequest(
        run_date=date(2024, 5, 16),
        cycle=0,
        hazard="tornado",
        model_set="2022",
        window="f12-35",
    )

    from src.sources.nadocast import NadocastGrid

    grid = NadocastGrid(
        request=request,
        url="https://example.test/grid.grib2",
        variable_name="torprob",
        units="%",
        latitude=np.zeros((1, 1), dtype=np.float32),
        longitude=np.zeros((1, 1), dtype=np.float32),
        probability=np.zeros((1, 1), dtype=np.float32),
        raw_values=np.zeros((1, 1), dtype=np.float32),
        attrs={},
    )

    assert grid.run_label == "NADOCast 2022 | 7:00 PM CDT May 15 f12-35"


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


def test_parse_spc_day1_geojson_polygon_metadata():
    outlook = parse_spc_outlook_geojson(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "DN": 3,
                        "VALID_ISO": "2026-06-10T13:00:00+00:00",
                        "EXPIRE_ISO": "2026-06-11T12:00:00+00:00",
                        "ISSUE_ISO": "2026-06-10T12:54:00+00:00",
                        "FORECASTER": "Guyer/Wendt",
                        "LABEL": "MRGL",
                        "LABEL2": "Marginal Risk",
                        "stroke": "#005500",
                        "fill": "#66A366",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-100.0, 35.0],
                                [-99.0, 35.0],
                                [-99.0, 36.0],
                                [-100.0, 36.0],
                                [-100.0, 35.0],
                            ]
                        ],
                    },
                }
            ],
        },
        product="cat",
        url="https://example.test/day1otlk_cat.nolyr.geojson",
    )

    assert outlook.product_label == "Categorical"
    assert outlook.valid_iso == "2026-06-10T13:00:00+00:00"
    assert outlook.valid_period_label == (
        "SPC Day 1 Categorical valid 8:00 AM CDT Jun 10 - 7:00 AM CDT Jun 11"
    )
    assert outlook.forecaster == "Guyer/Wendt"
    assert len(outlook.polygons) == 1
    assert outlook.polygons[0].dn == 3
    assert outlook.polygons[0].label == "MRGL"
    assert outlook.polygons[0].geometry.area > 0


def test_spc_probability_and_intensity_labels_are_distinct():
    outlook = parse_spc_outlook_geojson(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "DN": 10,
                        "LABEL": "0.10",
                        "LABEL2": "10% Tornado Risk",
                        "stroke": "#FD8A2B",
                        "fill": "#FFE481",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-100.0, 35.0],
                                [-99.0, 35.0],
                                [-99.0, 36.0],
                                [-100.0, 36.0],
                                [-100.0, 35.0],
                            ]
                        ],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {
                        "DN": 2,
                        "LABEL": "CIG1",
                        "LABEL2": "Tornado Conditional Intensity Group 1 Risk",
                        "stroke": "#000000",
                        "fill": "#888888",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-99.8, 35.2],
                                [-99.2, 35.2],
                                [-99.2, 35.8],
                                [-99.8, 35.8],
                                [-99.8, 35.2],
                            ]
                        ],
                    },
                },
            ],
        },
        product="tornado",
    )

    probability_polygon = next(p for p in outlook.polygons if p.label == "0.10")
    intensity_polygon = next(p for p in outlook.polygons if p.label == "CIG1")

    assert _spc_legend_label(probability_polygon.label) == "10%"
    assert not _is_spc_intensity_polygon(probability_polygon)
    assert _is_spc_intensity_polygon(intensity_polygon)
