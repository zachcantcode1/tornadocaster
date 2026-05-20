import numpy as np

from src.features.contract_alignment import (
    align_feature_matrix,
    add_first_pass_legacy_aliases,
    add_dynamic_legacy_aliases,
    add_temporal_proxy_aliases,
    account_for_ordered_features,
)


def test_align_feature_matrix_zero_fills_missing():
    arrays = {
        "a": np.array([1.0, 2.0], dtype=np.float32),
        "c": np.array([5.0, 6.0], dtype=np.float32),
    }
    mat, names, missing = align_feature_matrix(arrays, ["a", "b", "c"], grid_size=2)
    assert names == ["a", "b", "c"]
    assert missing == ["b"]
    np.testing.assert_allclose(mat[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(mat[:, 1], [0.0, 0.0])
    np.testing.assert_allclose(mat[:, 2], [5.0, 6.0])


def test_first_pass_legacy_aliases_add_cape_fields():
    arrays = {
        "raw": np.array([1.0, 2.0], dtype=np.float32),
        "25mi_mean": np.array([3.0, 4.0], dtype=np.float32),
        "50mi_mean": np.array([5.0, 6.0], dtype=np.float32),
        "100mi_mean": np.array([7.0, 8.0], dtype=np.float32),
        "50mi_gradient_x": np.array([9.0, 10.0], dtype=np.float32),
        "50mi_gradient_y": np.array([11.0, 12.0], dtype=np.float32),
        "100mi_gradient_x": np.array([13.0, 14.0], dtype=np.float32),
        "100mi_gradient_y": np.array([15.0, 16.0], dtype=np.float32),
        "100mi_gradient_mag": np.array([17.0, 18.0], dtype=np.float32),
    }
    aliased = add_first_pass_legacy_aliases(arrays)
    assert "CAPE:surface:hour fcst:wt ens mean" in aliased
    assert "CAPE:surface:hour fcst:wt ens mean:25mi mean" in aliased
    assert "CAPE:surface:hour fcst:wt ens mean:100mi linestraddling grad" in aliased


def test_dynamic_legacy_aliases_maps_generic_field_blocks():
    arrays = {
        "foo__raw": np.array([1.0, 2.0], dtype=np.float32),
        "foo__25mi_mean": np.array([3.0, 4.0], dtype=np.float32),
        "foo__50mi_gradient_x": np.array([5.0, 6.0], dtype=np.float32),
        "foo__100mi_gradient_mag": np.array([7.0, 8.0], dtype=np.float32),
    }
    mapped = add_dynamic_legacy_aliases(arrays, {"foo": "BAR:surface:hour fcst:wt ens mean"})
    assert "BAR:surface:hour fcst:wt ens mean" in mapped
    assert "BAR:surface:hour fcst:wt ens mean:25mi mean" in mapped
    assert "BAR:surface:hour fcst:wt ens mean:50mi forward grad" in mapped
    assert "BAR:surface:hour fcst:wt ens mean:100mi linestraddling grad" in mapped


def test_temporal_proxy_aliases_backfill_hour_and_window_suffixes():
    arrays = {
        "CAPE:surface:hour fcst:wt ens mean:25mi mean": np.array([1.0, 2.0], dtype=np.float32)
    }
    ordered = [
        "CAPE:surface:hour fcst:wt ens mean:25mi mean",
        "CAPE:surface:hour fcst:wt ens mean:25mi mean:+1hr",
        "CAPE:surface:hour fcst:wt ens mean:25mi mean:3hr mean",
        "CAPE:surface:hour fcst:wt ens mean:25mi mean:3hr delta",
    ]
    proxied = add_temporal_proxy_aliases(arrays, ordered)
    assert "CAPE:surface:hour fcst:wt ens mean:25mi mean:+1hr" in proxied
    assert "CAPE:surface:hour fcst:wt ens mean:25mi mean:3hr mean" in proxied
    assert "CAPE:surface:hour fcst:wt ens mean:25mi mean:3hr delta" in proxied


def test_account_for_ordered_features_classifies_sources():
    arrays = {
        "A": np.array([1.0, 2.0], dtype=np.float32),
        "B": np.array([3.0, 4.0], dtype=np.float32),
    }
    ordered = ["A", "B:+1hr", "C"]
    matrix, names, accounting = account_for_ordered_features(arrays, ordered, grid_size=2)
    assert names == ordered
    assert matrix.shape == (2, 3)
    assert accounting["total"] == 3
    assert accounting["direct_count"] == 1
    assert accounting["proxy_count"] == 1
    assert accounting["zero_fill_count"] == 1
