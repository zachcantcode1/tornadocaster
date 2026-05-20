from datetime import datetime, timezone

import numpy as np
import xarray as xr

from src.features.derived import build_first_pass_derived_fields
from src.features.nadocast_style import (
    NADOCAST_STYLE_FEATURE_COLS,
    add_nadocast_style_features,
    stack_feature_columns,
)


def test_nadocast_style_features_expand_motion_and_upstream_terms():
    y, x = np.mgrid[0:6, 0:6].astype(np.float32)
    base = xr.DataArray(x + y, dims=("y", "x"))
    fields = {
        "cape_ml": base * 100 + 500,
        "cin_ml": base * -5,
        "cin_mu": base * -4,
        "hlcy_3km": base * 10 + 100,
        "vwsh_0_6km": base + 15,
        "tmp_2m": base * 0 + 300,
        "dpt_2m": base * 0 + 292,
        "cape_surface": base * 120 + 600,
        "cape_mu": base * 140 + 700,
        "cin_surface": base * -4,
        "ugrd_10m": base * 0 + 8,
        "vgrd_10m": base * 0 + 4,
        "ugrd_925": base * 0 + 12,
        "vgrd_925": base * 0 + 5,
        "ugrd_850": base * 0 + 18,
        "vgrd_850": base * 0 + 8,
        "ugrd_700": base * 0 + 22,
        "vgrd_700": base * 0 + 9,
        "ugrd_500": base * 0 + 30,
        "vgrd_500": base * 0 + 14,
        "ugrd_250": base * 0 + 40,
        "vgrd_250": base * 0 + 20,
        "tmp_700": base * 0 + 275,
        "tmp_500": base * 0 + 255,
        "tmp_925": base * 0 + 295,
        "dpt_925": base * 0 + 289,
        "dpt_850": base * 0 + 285,
        "refc_atm": base * 3 + 20,
        "mxuphl_25km": base * 5,
        "pwat": base + 30,
    }
    derived = build_first_pass_derived_fields(fields)
    feature_map = {
        "cape_ml": fields["cape_ml"].values,
        "cin_ml": fields["cin_ml"].values,
        "hlcy_3km": fields["hlcy_3km"].values,
        "vwsh_0_6km": fields["vwsh_0_6km"].values,
        "dpt_2m": fields["dpt_2m"].values,
        "tmp_2m": fields["tmp_2m"].values,
    }
    expanded = add_nadocast_style_features(
        feature_map,
        {**fields, **derived},
        (6, 6),
        valid_dt=datetime(2026, 5, 18, 21, tzinfo=timezone.utc),
    )

    assert len(NADOCAST_STYLE_FEATURE_COLS) > 200
    assert "dpt_2m_50km_forward_grad" in expanded
    assert "dpt_2m_50km_leftward_grad" in expanded
    assert "storm_upstream_conv_925_3hr_gated_scp" in expanded
    assert "refc_atm_prob_gt_40p0_25km_mean" in expanded

    mat = stack_feature_columns(
        expanded,
        ["dpt_2m_50km_forward_grad", "storm_upstream_conv_925_3hr_gated_scp"],
        (6, 6),
    )
    assert mat.shape == (36, 2)
    assert np.isfinite(mat).all()
