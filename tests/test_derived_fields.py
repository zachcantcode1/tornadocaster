import numpy as np
import xarray as xr

from src.features.derived import build_first_pass_derived_fields


def test_build_first_pass_derived_fields_smoke():
    data = np.ones((4, 4), dtype=np.float32)
    da = xr.DataArray(data, dims=("y", "x"))
    fields = {
        "cape_surface": da * 1000,
        "cape_ml": da * 800,
        "cape_mu": da * 1200,
        "cin_surface": da * -50,
        "cin_ml": da * -40,
        "hlcy_3km": da * 200,
        "vwsh_0_6km": da * 20,
        "tmp_700": da * 280,
        "tmp_500": da * 260,
        "tmp_850": da * 290,
        "tmp_925": da * 295,
        "dpt_850": da * 285,
        "dpt_925": da * 288,
        "ugrd_500": da * 10,
        "vgrd_500": da * 10,
        "ugrd_700": da * 8,
        "vgrd_700": da * 8,
        "ugrd_850": da * 6,
        "vgrd_850": da * 6,
        "ugrd_925": da * 5,
        "vgrd_925": da * 5,
        "wind_10m": da * 4,
        "wind_80m": da * 5,
        "wind_850": da * 9,
        "wind_925": da * 7,
    }
    out = build_first_pass_derived_fields(fields)
    assert "calc_700_500_lapse" in out
    assert "calc_sbcape_hlcy" in out
    assert "calc_scpish_rm" in out
    assert "calc_max_wind_le_850" in out
    assert "calc_ushear" in out
    assert "calc_vshear" in out
    assert "calc_ustm" in out
    assert "calc_div_925" in out
    assert "calc_abs_vort_500" in out
