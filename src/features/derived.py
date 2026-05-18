import numpy as np
import xarray as xr


def _sqrt_pos(x: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.sqrt, xr.where(x > 0, x, 0))


def _dx_dy(a: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    dy, dx = np.gradient(a.values.astype(np.float32), axis=(0, 1))
    return xr.DataArray(dx, coords=a.coords, dims=a.dims), xr.DataArray(dy, coords=a.coords, dims=a.dims)


def _mixing_ratio(dpt_k: xr.DataArray, pressure_hpa: float) -> xr.DataArray:
    """Approximation of mixing ratio (g/kg) from dewpoint (K) and pressure (hPa)."""
    td_c = xr.where(dpt_k > 173.15, dpt_k, 173.15) - 273.15  # clamp to -100°C min
    td_c = xr.where(td_c < 60.0, td_c, 60.0)  # clamp to 60°C max
    e = 6.112 * xr.apply_ufunc(np.exp, 17.67 * td_c / (td_c + 243.5))
    e = xr.where(e < pressure_hpa * 0.99, e, pressure_hpa * 0.99)  # avoid divide-by-zero
    return 622.0 * e / (pressure_hpa - e)


def build_first_pass_derived_fields(
    fields: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    """
    First-pass derived feature set inspired by Nadocast extra_features.
    """
    out: dict[str, xr.DataArray] = {}

    cape_sfc = fields.get("cape_surface")
    cape_ml = fields.get("cape_ml")
    cape_mu = fields.get("cape_mu")
    cin_sfc = fields.get("cin_surface")
    cin_ml = fields.get("cin_ml")
    hlcy = fields.get("hlcy_3km")
    vwsh = fields.get("vwsh_0_6km")
    tmp_700 = fields.get("tmp_700")
    tmp_500 = fields.get("tmp_500")
    tmp_850 = fields.get("tmp_850")
    tmp_925 = fields.get("tmp_925")
    dpt_850 = fields.get("dpt_850")
    dpt_925 = fields.get("dpt_925")
    ugrd_500 = fields.get("ugrd_500")
    vgrd_500 = fields.get("vgrd_500")
    ugrd_700 = fields.get("ugrd_700")
    vgrd_700 = fields.get("vgrd_700")
    ugrd_850 = fields.get("ugrd_850")
    vgrd_850 = fields.get("vgrd_850")
    ugrd_925 = fields.get("ugrd_925")
    vgrd_925 = fields.get("vgrd_925")
    wind_10m = fields.get("wind_10m")
    wind_80m = fields.get("wind_80m")
    wind_850 = fields.get("wind_850")
    wind_925 = fields.get("wind_925")

    if tmp_700 is not None and tmp_500 is not None:
        out["calc_700_500_lapse"] = tmp_700 - tmp_500
    if tmp_925 is not None and tmp_700 is not None:
        out["calc_925_700_lapse"] = tmp_925 - tmp_700
    if ugrd_700 is not None and vgrd_700 is not None:
        out["calc_wind_700"] = np.hypot(ugrd_700, vgrd_700)
    if ugrd_500 is not None and vgrd_500 is not None:
        out["calc_wind_500"] = np.hypot(ugrd_500, vgrd_500)

    if cape_sfc is not None and hlcy is not None:
        out["calc_sbcape_hlcy"] = cape_sfc * hlcy
        out["calc_sqrt_sbcape_hlcy"] = _sqrt_pos(cape_sfc) * hlcy
    if cape_ml is not None and hlcy is not None:
        out["calc_mlcape_hlcy"] = cape_ml * hlcy
        out["calc_sqrt_mlcape_hlcy"] = _sqrt_pos(cape_ml) * hlcy
    if cape_sfc is not None and vwsh is not None:
        out["calc_sbcape_bwd"] = cape_sfc * vwsh
        out["calc_sqrt_sbcape_bwd"] = _sqrt_pos(cape_sfc) * vwsh
    if cape_ml is not None and vwsh is not None:
        out["calc_mlcape_bwd"] = cape_ml * vwsh
        out["calc_sqrt_mlcape_bwd"] = _sqrt_pos(cape_ml) * vwsh

    if cape_sfc is not None and cin_sfc is not None:
        out["calc_sbcape_200_plus_sbcin"] = cape_sfc * (200.0 + cin_sfc)
        out["calc_sqrt_sbcape_200_plus_sbcin"] = _sqrt_pos(cape_sfc) * (200.0 + cin_sfc)
    if cape_ml is not None and cin_ml is not None:
        out["calc_mlcape_200_plus_mlcin"] = cape_ml * (200.0 + cin_ml)
        out["calc_sqrt_mlcape_200_plus_mlcin"] = _sqrt_pos(cape_ml) * (200.0 + cin_ml)

    if cape_mu is not None and hlcy is not None and vwsh is not None:
        out["calc_scpish_rm"] = (cape_mu * hlcy * vwsh) * (1.0 / (1000.0 * 50.0 * 20.0))
    if wind_10m is not None and wind_80m is not None and wind_925 is not None and wind_850 is not None:
        out["calc_max_wind_le_850"] = xr.apply_ufunc(np.maximum, xr.apply_ufunc(np.maximum, wind_10m, wind_80m), xr.apply_ufunc(np.maximum, wind_925, wind_850))
        out["calc_sum_wind_le_850"] = wind_10m + wind_80m + wind_925 + wind_850
    if "calc_max_wind_le_850" in out and "calc_wind_700" in out:
        out["calc_max_wind_le_700"] = xr.apply_ufunc(np.maximum, out["calc_max_wind_le_850"], out["calc_wind_700"])
    if "calc_sum_wind_le_850" in out and "calc_wind_700" in out:
        out["calc_sum_wind_le_700"] = out["calc_sum_wind_le_850"] + out["calc_wind_700"]

    # Simple bunkers-style bulk components (first-pass approximation).
    if all(v is not None for v in [ugrd_925, ugrd_850, ugrd_700, ugrd_500]):
        out["calc_umean"] = (ugrd_925 + ugrd_850 + 0.5 * ugrd_700 + ugrd_500) / 3.5
    if all(v is not None for v in [vgrd_925, vgrd_850, vgrd_700, vgrd_500]):
        out["calc_vmean"] = (vgrd_925 + vgrd_850 + 0.5 * vgrd_700 + vgrd_500) / 3.5
    if all(v is not None for v in [ugrd_500, ugrd_925]):
        out["calc_ushear"] = 0.95 * ugrd_500 - 0.93 * ugrd_925
        if fields.get("ugrd_250") is not None:
            out["calc_ushear"] = out["calc_ushear"] + 0.05 * fields["ugrd_250"]
    if all(v is not None for v in [vgrd_500, vgrd_925]):
        out["calc_vshear"] = 0.95 * vgrd_500 - 0.93 * vgrd_925
        if fields.get("vgrd_250") is not None:
            out["calc_vshear"] = out["calc_vshear"] + 0.05 * fields["vgrd_250"]
    if "calc_ushear" in out and "calc_vshear" in out:
        out["calc_shear"] = np.hypot(out["calc_ushear"], out["calc_vshear"]) + 1e-6
    if "calc_umean" in out and "calc_vshear" in out and "calc_shear" in out:
        out["calc_ustm"] = out["calc_umean"] + 7.5 * out["calc_vshear"] / (out["calc_shear"] + 0.25)
    if "calc_vmean" in out and "calc_ushear" in out and "calc_shear" in out:
        out["calc_vstm"] = out["calc_vmean"] - 7.5 * out["calc_ushear"] / (out["calc_shear"] + 0.25)

    # Simple first-pass divergence/vorticity approximations (grid-unit derivatives).
    if ugrd_925 is not None and vgrd_925 is not None:
        dudx, dudy = _dx_dy(ugrd_925)
        dvdx, dvdy = _dx_dy(vgrd_925)
        out["calc_div_925"] = (dudx + dvdy) * 1e5
        out["calc_abs_vort_925"] = (dvdx - dudy) * 1e5
    if ugrd_850 is not None and vgrd_850 is not None:
        dudx, dudy = _dx_dy(ugrd_850)
        dvdx, dvdy = _dx_dy(vgrd_850)
        out["calc_div_850"] = (dudx + dvdy) * 1e5
        out["calc_abs_vort_850"] = (dvdx - dudy) * 1e5
    if ugrd_700 is not None and vgrd_700 is not None:
        dudx, dudy = _dx_dy(ugrd_700)
        dvdx, dvdy = _dx_dy(vgrd_700)
        out["calc_abs_vort_700"] = (dvdx - dudy) * 1e5
    if ugrd_500 is not None and vgrd_500 is not None:
        dudx, dudy = _dx_dy(ugrd_500)
        dvdx, dvdy = _dx_dy(vgrd_500)
        out["calc_abs_vort_500"] = (dvdx - dudy) * 1e5
    if fields.get("ugrd_250") is not None and fields.get("vgrd_250") is not None:
        dudx, dudy = _dx_dy(fields["ugrd_250"])
        dvdx, dvdy = _dx_dy(fields["vgrd_250"])
        out["calc_div_250"] = (dudx + dvdy) * 1e5
        out["calc_abs_vort_250"] = (dvdx - dudy) * 1e5

    if "calc_div_250" in out and "calc_div_925" in out:
        out["calc_diff_div_250_925"] = out["calc_div_250"] - out["calc_div_925"]
    if "calc_div_925" in out:
        out["calc_conv_only_925"] = xr.where(out["calc_div_925"] < 0, -out["calc_div_925"], 0)
    if "calc_div_850" in out:
        out["calc_conv_only_850"] = xr.where(out["calc_div_850"] < 0, -out["calc_div_850"], 0)

    # --- Triple / quad compound features ---

    # CAPE × HLCY × CIN
    if cape_sfc is not None and hlcy is not None and cin_sfc is not None:
        cin_term_sfc = 200.0 + cin_sfc
        out["calc_sbcape_hlcy_cin"] = out["calc_sbcape_hlcy"] * cin_term_sfc
        out["calc_sqrt_sbcape_hlcy_cin"] = out["calc_sqrt_sbcape_hlcy"] * cin_term_sfc
    if cape_ml is not None and hlcy is not None and cin_ml is not None:
        cin_term_ml = 200.0 + cin_ml
        out["calc_mlcape_hlcy_cin"] = out["calc_mlcape_hlcy"] * cin_term_ml
        out["calc_sqrt_mlcape_hlcy_cin"] = out["calc_sqrt_mlcape_hlcy"] * cin_term_ml

    # CAPE × BWD × HLCY
    if "calc_sbcape_bwd" in out and hlcy is not None:
        out["calc_sbcape_bwd_hlcy"] = out["calc_sbcape_bwd"] * hlcy
        out["calc_sqrt_sbcape_bwd_hlcy"] = out["calc_sqrt_sbcape_bwd"] * hlcy
    if "calc_mlcape_bwd" in out and hlcy is not None:
        out["calc_mlcape_bwd_hlcy"] = out["calc_mlcape_bwd"] * hlcy
        out["calc_sqrt_mlcape_bwd_hlcy"] = out["calc_sqrt_mlcape_bwd"] * hlcy

    # CAPE × BWD × HLCY × CIN
    if "calc_sbcape_bwd_hlcy" in out and cin_sfc is not None:
        out["calc_sbcape_bwd_hlcy_cin"] = out["calc_sbcape_bwd_hlcy"] * (200.0 + cin_sfc)
        out["calc_sqrt_sbcape_bwd_hlcy_cin"] = out["calc_sqrt_sbcape_bwd_hlcy"] * (200.0 + cin_sfc)
    if "calc_mlcape_bwd_hlcy" in out and cin_ml is not None:
        out["calc_mlcape_bwd_hlcy_cin"] = out["calc_mlcape_bwd_hlcy"] * (200.0 + cin_ml)
        out["calc_sqrt_mlcape_bwd_hlcy_cin"] = out["calc_sqrt_mlcape_bwd_hlcy"] * (200.0 + cin_ml)

    # Lapse rate × BWD and lapse rate × cold-500mb × BWD
    if "calc_700_500_lapse" in out and vwsh is not None:
        out["calc_lapse_bwd"] = out["calc_700_500_lapse"] * vwsh
        if tmp_500 is not None:
            cold500 = 273.15 - tmp_500  # -(Celsius at 500mb)
            out["calc_lapse_cold500_bwd"] = out["calc_700_500_lapse"] * cold500 * vwsh

    # MUCAPE compound features (MUCAPE = cape_mu in our naming)
    if cape_mu is not None and vwsh is not None:
        out["calc_mucape_bwd"] = cape_mu * vwsh
        if "calc_700_500_lapse" in out:
            out["calc_mucape_lapse_bwd"] = cape_mu * out["calc_700_500_lapse"] * vwsh
            if tmp_500 is not None:
                cold500 = 273.15 - tmp_500
                out["calc_mucape_lapse_cold500_bwd"] = cape_mu * out["calc_700_500_lapse"] * cold500 * vwsh
                dpt_925 = fields.get("dpt_925")
                dpt_850 = fields.get("dpt_850")
                if dpt_925 is not None:
                    mixr925 = _mixing_ratio(dpt_925, 925.0)
                    out["calc_mucape_mixr925_lapse_cold500_bwd"] = (
                        cape_mu * mixr925 * out["calc_700_500_lapse"] * cold500 * vwsh
                    )
                if dpt_850 is not None:
                    mixr850 = _mixing_ratio(dpt_850, 850.0)
                    out["calc_mucape_mixr850_lapse_cold500_bwd"] = (
                        cape_mu * mixr850 * out["calc_700_500_lapse"] * cold500 * vwsh
                    )

    # SCP binary threshold
    if "calc_scpish_rm" in out:
        out["calc_scpish_gt1"] = xr.where(out["calc_scpish_rm"] > 1.0, 1.0, 0.0)

    # Storm-relative mid-level wind components (mean of storm motion and 500mb wind)
    ugrd_500 = fields.get("ugrd_500")
    vgrd_500 = fields.get("vgrd_500")
    if "calc_ustm" in out and ugrd_500 is not None:
        out["calc_ustm_500_mean"] = 0.5 * (out["calc_ustm"] + ugrd_500)
    if "calc_vstm" in out and vgrd_500 is not None:
        out["calc_vstm_500_mean"] = 0.5 * (out["calc_vstm"] + vgrd_500)

    return out
