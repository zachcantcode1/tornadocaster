import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone
from src.ingestion.noaa_fetcher import NOAAIndexFetcher
from src.features.spatial import generate_legacy_feature_blocks, generate_legacy_feature_blocks_for_fields
from src.features.derived import build_first_pass_derived_fields
from src.features.contract_alignment import (
    align_feature_matrix,
    build_feature_coverage_summary,
    add_first_pass_legacy_aliases,
    add_dynamic_legacy_aliases,
    add_temporal_proxy_aliases,
    account_for_ordered_features,
)
from src.models.lgb_bridge import JuliaToLightGBMBridge
from src.models.model_contract import load_contract, select_model_path
from src.models.julia_sidecar import JuliaSidecarRunner
from src.llm.llm_report import ForecastDiscussionGenerator
import xarray as xr
import numpy as np
import tempfile
import os

logger = logging.getLogger(__name__)

FIRST_PASS_FIELD_CATALOG = [
    # Instability
    ("cape_surface", "CAPE", "surface", "CAPE:surface:hour fcst:wt ens mean"),
    ("cape_ml", "CAPE", "90-0 mb above ground", "CAPE:90-0 mb above ground:hour fcst:wt ens mean"),
    ("cape_mu", "CAPE", "180-0 mb above ground", "CAPE:180-0 mb above ground:hour fcst:wt ens mean"),
    ("cin_surface", "CIN", "surface", "CIN:surface:hour fcst:wt ens mean"),
    ("cin_ml", "CIN", "90-0 mb above ground", "CIN:90-0 mb above ground:hour fcst:wt ens mean"),
    ("cin_mu", "CIN", "180-0 mb above ground", "CIN:180-0 mb above ground:hour fcst:wt ens mean"),
    # Shear / rotation
    ("hlcy_3km", "HLCY", "3000-0 m above ground", "HLCY:3000-0 m above ground:hour fcst:wt ens mean"),
    ("hlcy_1km", "HLCY", "1000-0 m above ground", "HLCY:1000-0 m above ground:hour fcst:wt ens mean"),
    ("vwsh_0_6km", "VWSH", "6000-0 m above ground", "VWSH:6000-0 m above ground:hour fcst:wt ens mean"),
    ("vwsh_surface", "VWSH", "surface", "VWSH:surface:hour fcst:wt ens mean"),
    # Geopotential heights
    ("hgt_250", "HGT", "250 mb", "HGT:250 mb:hour fcst:wt ens mean"),
    ("hgt_500", "HGT", "500 mb", "HGT:500 mb:hour fcst:wt ens mean"),
    ("hgt_700", "HGT", "700 mb", "HGT:700 mb:hour fcst:wt ens mean"),
    ("hgt_850", "HGT", "850 mb", "HGT:850 mb:hour fcst:wt ens mean"),
    ("hgt_925", "HGT", "925 mb", "HGT:925 mb:hour fcst:wt ens mean"),
    ("hgt_cloud_base", "HGT", "cloud base", "HGT:cloud base:hour fcst:wt ens mean"),
    ("hgt_cloud_ceiling", "HGT", "cloud ceiling", "HGT:cloud ceiling:hour fcst:wt ens mean"),
    # U-component winds
    ("ugrd_10m", "UGRD", "10 m above ground", "UGRD:10 m above ground:hour fcst:wt ens mean"),
    ("ugrd_80m", "UGRD", "80 m above ground", "UGRD:80 m above ground:hour fcst:wt ens mean"),
    ("ugrd_250", "UGRD", "250 mb", "UGRD:250 mb:hour fcst:wt ens mean"),
    ("ugrd_500", "UGRD", "500 mb", "UGRD:500 mb:hour fcst:wt ens mean"),
    ("ugrd_700", "UGRD", "700 mb", "UGRD:700 mb:hour fcst:wt ens mean"),
    ("ugrd_850", "UGRD", "850 mb", "UGRD:850 mb:hour fcst:wt ens mean"),
    ("ugrd_925", "UGRD", "925 mb", "UGRD:925 mb:hour fcst:wt ens mean"),
    # V-component winds
    ("vgrd_10m", "VGRD", "10 m above ground", "VGRD:10 m above ground:hour fcst:wt ens mean"),
    ("vgrd_80m", "VGRD", "80 m above ground", "VGRD:80 m above ground:hour fcst:wt ens mean"),
    ("vgrd_250", "VGRD", "250 mb", "VGRD:250 mb:hour fcst:wt ens mean"),
    ("vgrd_500", "VGRD", "500 mb", "VGRD:500 mb:hour fcst:wt ens mean"),
    ("vgrd_700", "VGRD", "700 mb", "VGRD:700 mb:hour fcst:wt ens mean"),
    ("vgrd_850", "VGRD", "850 mb", "VGRD:850 mb:hour fcst:wt ens mean"),
    ("vgrd_925", "VGRD", "925 mb", "VGRD:925 mb:hour fcst:wt ens mean"),
    # Wind speed
    ("wind_10m", "WIND", "10 m above ground", "WIND:10 m above ground:hour fcst:wt ens mean"),
    ("wind_80m", "WIND", "80 m above ground", "WIND:80 m above ground:hour fcst:wt ens mean"),
    ("wind_250", "WIND", "250 mb", "WIND:250 mb:hour fcst:wt ens mean"),
    ("wind_850", "WIND", "850 mb", "WIND:850 mb:hour fcst:wt ens mean"),
    ("wind_925", "WIND", "925 mb", "WIND:925 mb:hour fcst:wt ens mean"),
    # Temperature
    ("tmp_2m", "TMP", "2 m above ground", "TMP:2 m above ground:hour fcst:wt ens mean"),
    ("tmp_250", "TMP", "250 mb", "TMP:250 mb:hour fcst:wt ens mean"),
    ("tmp_500", "TMP", "500 mb", "TMP:500 mb:hour fcst:wt ens mean"),
    ("tmp_700", "TMP", "700 mb", "TMP:700 mb:hour fcst:wt ens mean"),
    ("tmp_850", "TMP", "850 mb", "TMP:850 mb:hour fcst:wt ens mean"),
    ("tmp_925", "TMP", "925 mb", "TMP:925 mb:hour fcst:wt ens mean"),
    # Dewpoint
    ("dpt_2m", "DPT", "2 m above ground", "DPT:2 m above ground:hour fcst:wt ens mean"),
    ("dpt_500", "DPT", "500 mb", "DPT:500 mb:hour fcst:wt ens mean"),
    ("dpt_700", "DPT", "700 mb", "DPT:700 mb:hour fcst:wt ens mean"),
    ("dpt_850", "DPT", "850 mb", "DPT:850 mb:hour fcst:wt ens mean"),
    ("dpt_925", "DPT", "925 mb", "DPT:925 mb:hour fcst:wt ens mean"),
    # Vertical motion
    ("vvel_700", "VVEL", "700 mb", "VVEL:700 mb:hour fcst:wt ens mean"),
    ("vvel_700_500", "VVEL", "700-500 mb", "VVEL:700-500 mb:hour fcst:wt ens mean"),
    # Moisture / precip / clouds
    ("rh_700", "RH", "700 mb", "RH:700 mb:hour fcst:wt ens mean"),
    ("pwat", "PWAT", "entire atmosphere", "PWAT:entire atmosphere:hour fcst:wt ens mean"),
    ("vis_surface", "VIS", "surface", "VIS:surface:hour fcst:wt ens mean"),
    ("crain_surface", "CRAIN", "surface", "CRAIN:surface:hour fcst:wt ens mean"),
    ("cfrzr_surface", "CFRZR", "surface", "CFRZR:surface:hour fcst:wt ens mean"),
    ("cicep_surface", "CICEP", "surface", "CICEP:surface:hour fcst:wt ens mean"),
    ("csnow_surface", "CSNOW", "surface", "CSNOW:surface:hour fcst:wt ens mean"),
    ("lcdc", "LCDC", "low cloud layer", "LCDC:low cloud layer:hour fcst:wt ens mean"),
    ("mcdc", "MCDC", "middle cloud layer", "MCDC:middle cloud layer:hour fcst:wt ens mean"),
    ("hcdc", "HCDC", "high cloud layer", "HCDC:high cloud layer:hour fcst:wt ens mean"),
    ("tcdc", "TCDC", "entire atmosphere", "TCDC:entire atmosphere:hour fcst:wt ens mean"),
    # Soil
    ("soilw_0_10cm", "SOILW", "0-0.1 m below ground", "SOILW:0-0.1 m below ground:hour fcst:wt ens mean"),
    ("tsoil_0_10cm", "TSOIL", "0-0.1 m below ground", "TSOIL:0-0.1 m below ground:hour fcst:wt ens mean"),
    # Other
    ("hindex_surface", "HINDEX", "surface", "HINDEX:surface:hour fcst:wt ens mean"),
    # Convective indicators — storm gate in calibrated_predictor requires these to
    # distinguish active convection from a favorable-but-storm-free environment.
    ("refc_atm",    "REFC",   "entire atmosphere", "REFC:entire atmosphere:hour fcst:wt ens mean"),
    ("mxuphl_03km", "MXUPHL", "3000-0 m above ground",    "MXUPHL:3000-0 m above ground:hour fcst:wt ens mean"),
    ("mxuphl_25km", "MXUPHL", "5000-2000 m above ground", "MXUPHL:5000-2000 m above ground:hour fcst:wt ens mean"),
    ("gust_surface", "GUST",  "surface",                  "GUST:surface:hour fcst:wt ens mean"),
]

FIRST_PASS_DERIVED_LEGACY_BASE = {
    # Lapse rates
    "calc_700_500_lapse": "700-500mbLapseRate:calculated:hour fcst:",
    "calc_925_700_lapse": "925-700mbLapseRate:calculated:hour fcst:",
    # Wind magnitudes
    "calc_wind_700": "Wind700mb:calculated:hour fcst:",
    "calc_wind_500": "Wind500mb:calculated:hour fcst:",
    # CAPE × HLCY products
    "calc_sbcape_hlcy": "SBCAPE*HLCY3000-0m:calculated:hour fcst:",
    "calc_mlcape_hlcy": "MLCAPE*HLCY3000-0m:calculated:hour fcst:",
    "calc_sqrt_sbcape_hlcy": "sqrtSBCAPE*HLCY3000-0m:calculated:hour fcst:",
    "calc_sqrt_mlcape_hlcy": "sqrtMLCAPE*HLCY3000-0m:calculated:hour fcst:",
    # CAPE × BWD products
    "calc_sbcape_bwd": "SBCAPE*BWD0-6km:calculated:hour fcst:",
    "calc_mlcape_bwd": "MLCAPE*BWD0-6km:calculated:hour fcst:",
    "calc_sqrt_sbcape_bwd": "sqrtSBCAPE*BWD0-6km:calculated:hour fcst:",
    "calc_sqrt_mlcape_bwd": "sqrtMLCAPE*BWD0-6km:calculated:hour fcst:",
    # CAPE × CIN products
    "calc_sbcape_200_plus_sbcin": "SBCAPE*(200+SBCIN):calculated:hour fcst:",
    "calc_mlcape_200_plus_mlcin": "MLCAPE*(200+MLCIN):calculated:hour fcst:",
    "calc_sqrt_sbcape_200_plus_sbcin": "sqrtSBCAPE*(200+SBCIN):calculated:hour fcst:",
    "calc_sqrt_mlcape_200_plus_mlcin": "sqrtMLCAPE*(200+MLCIN):calculated:hour fcst:",
    # CAPE × HLCY × CIN triple products
    "calc_sbcape_hlcy_cin": "SBCAPE*HLCY3000-0m*(200+SBCIN):calculated:hour fcst:",
    "calc_mlcape_hlcy_cin": "MLCAPE*HLCY3000-0m*(200+MLCIN):calculated:hour fcst:",
    "calc_sqrt_sbcape_hlcy_cin": "sqrtSBCAPE*HLCY3000-0m*(200+SBCIN):calculated:hour fcst:",
    "calc_sqrt_mlcape_hlcy_cin": "sqrtMLCAPE*HLCY3000-0m*(200+MLCIN):calculated:hour fcst:",
    # CAPE × BWD × HLCY triple products
    "calc_sbcape_bwd_hlcy": "SBCAPE*BWD0-6km*HLCY3000-0m:calculated:hour fcst:",
    "calc_mlcape_bwd_hlcy": "MLCAPE*BWD0-6km*HLCY3000-0m:calculated:hour fcst:",
    "calc_sqrt_sbcape_bwd_hlcy": "sqrtSBCAPE*BWD0-6km*HLCY3000-0m:calculated:hour fcst:",
    "calc_sqrt_mlcape_bwd_hlcy": "sqrtMLCAPE*BWD0-6km*HLCY3000-0m:calculated:hour fcst:",
    # CAPE × BWD × HLCY × CIN quad products
    "calc_sbcape_bwd_hlcy_cin": "SBCAPE*BWD0-6km*HLCY3000-0m*(200+SBCIN):calculated:hour fcst:",
    "calc_mlcape_bwd_hlcy_cin": "MLCAPE*BWD0-6km*HLCY3000-0m*(200+MLCIN):calculated:hour fcst:",
    "calc_sqrt_sbcape_bwd_hlcy_cin": "sqrtSBCAPE*BWD0-6km*HLCY3000-0m*(200+SBCIN):calculated:hour fcst:",
    "calc_sqrt_mlcape_bwd_hlcy_cin": "sqrtMLCAPE*BWD0-6km*HLCY3000-0m*(200+MLCIN):calculated:hour fcst:",
    # Lapse rate compound products
    "calc_lapse_bwd": "700-500mbLapseRate*BWD0-6km:calculated:hour fcst:",
    "calc_lapse_cold500_bwd": "700-500mbLapseRate*-Celcius500mb*BWD0-6km:calculated:hour fcst:",
    # MUCAPE compound products
    "calc_mucape_bwd": "MUCAPE*BWD0-6km:calculated:hour fcst:",
    "calc_mucape_lapse_bwd": "MUCAPE*700-500mbLapseRate*BWD0-6km:calculated:hour fcst:",
    "calc_mucape_lapse_cold500_bwd": "MUCAPE*700-500mbLapseRate*-Celcius500mb*BWD0-6km:calculated:hour fcst:",
    "calc_mucape_mixr925_lapse_cold500_bwd": "MUCAPE*MixingRatio925mb*700-500mbLapseRate*-Celcius500mb*BWD0-6km:calculated:hour fcst:",
    "calc_mucape_mixr850_lapse_cold500_bwd": "MUCAPE*MixingRatio850mb*700-500mbLapseRate*-Celcius500mb*BWD0-6km:calculated:hour fcst:",
    # SCP and derived storm-motion
    "calc_scpish_rm": "SCPish(RM):calculated:hour fcst:",
    "calc_scpish_gt1": "SCPish(RM)>1:calculated:hour fcst:",
    "calc_ustm_500_mean": "U½STM½500mb:calculated:hour fcst:",
    "calc_vstm_500_mean": "V½STM½500mb:calculated:hour fcst:",
    # Wind extremes
    "calc_max_wind_le_850": "MaxWind<=850mb:calculated:hour fcst:",
    "calc_sum_wind_le_850": "SumWind<=850mb:calculated:hour fcst:",
    "calc_max_wind_le_700": "MaxWind<=700mb:calculated:hour fcst:",
    "calc_sum_wind_le_700": "SumWind<=700mb:calculated:hour fcst:",
    # Mean and shear components
    "calc_umean": "UMEAN:calculated:hour fcst:",
    "calc_vmean": "VMEAN:calculated:hour fcst:",
    "calc_ushear": "USHEAR:calculated:hour fcst:",
    "calc_vshear": "VSHEAR:calculated:hour fcst:",
    "calc_shear": "SHEAR:calculated:hour fcst:",
    "calc_ustm": "USTM:calculated:hour fcst:",
    "calc_vstm": "VSTM:calculated:hour fcst:",
    # Divergence and vorticity
    "calc_div_925": "Divergence925mb*10^5:calculated:hour fcst:",
    "calc_div_850": "Divergence850mb*10^5:calculated:hour fcst:",
    "calc_div_250": "Divergence250mb*10^5:calculated:hour fcst:",
    "calc_diff_div_250_925": "DifferentialDivergence250-925mb*10^5:calculated:hour fcst:",
    "calc_conv_only_925": "ConvergenceOnly925mb*10^5:calculated:hour fcst:",
    "calc_conv_only_850": "ConvergenceOnly850mb*10^5:calculated:hour fcst:",
    "calc_abs_vort_925": "AbsVorticity925mb*10^5:calculated:hour fcst:",
    "calc_abs_vort_850": "AbsVorticity850mb*10^5:calculated:hour fcst:",
    "calc_abs_vort_700": "AbsVorticity700mb*10^5:calculated:hour fcst:",
    "calc_abs_vort_500": "AbsVorticity500mb*10^5:calculated:hour fcst:",
    "calc_abs_vort_250": "AbsVorticity250mb*10^5:calculated:hour fcst:",
}

HRRR_S3_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
RRFS_S3_BASE = "https://noaa-rrfs-pds.s3.amazonaws.com"

# RRFS-A field catalog.  Tuple layout: (internal_name, grib_variable, grib_level).
# RRFS uses height-AGL / special-level descriptors rather than pressure levels for
# most convective parameters — see rrfs.tHHz.2dfld.2p5km.fFFF.hi.grib2 .idx layout.
RRFS_FIELD_CATALOG = [
    # Instability
    ("cape_surface", "CAPE", "surface"),
    ("cape_ml",      "CAPE", "90-0 mb above ground"),
    ("cape_mu",      "CAPE", "180-0 mb above ground"),
    ("cape_deep_ml", "CAPE", "255-0 mb above ground"),
    ("cin_surface",  "CIN",  "surface"),
    ("cin_ml",       "CIN",  "90-0 mb above ground"),
    ("cin_mu",       "CIN",  "180-0 mb above ground"),
    # Shear / rotation (SRH available at two depths)
    ("hlcy_3km", "HLCY", "3000-0 m above ground"),
    ("hlcy_1km", "HLCY", "1000-0 m above ground"),
    # LCL height and pressure — direct model output, no T2m/Td2m approximation
    ("hgt_lcl",  "HGT",  "level of adiabatic condensation from sfc"),
    ("pres_lcl", "PRES", "level of adiabatic condensation from sfc"),
    # Composite / derived parameters computed in-model
    ("efhl_surface",  "EFHL",   "surface"),                   # effective helicity
    ("cangle_500m",   "CANGLE", "0-500 m above ground"),      # critical angle
    ("dcape_400mb",   "DCAPE",  "400-0 mb above ground"),     # downdraft CAPE
    # Max updraft helicity — key supercell / tornado indicator
    ("mxuphl_03km", "MXUPHL", "3000-0 m above ground"),
    ("mxuphl_25km", "MXUPHL", "5000-2000 m above ground"),
    # Min updraft helicity (negative rotation tracking)
    ("mnuphl_03km", "MNUPHL", "3000-0 m above ground"),
    # Low-level relative vorticity
    ("relv_2km", "RELV", "2000-0 m above ground"),
    ("relv_1km", "RELV", "1000-0 m above ground"),
    # Surface / near-surface
    ("tmp_2m",        "TMP",   "2 m above ground"),
    ("dpt_2m",        "DPT",   "2 m above ground"),
    ("gust_surface",  "GUST",  "surface"),
    ("hpbl_surface",  "HPBL",  "surface"),    # PBL height
    ("vis_surface",   "VIS",   "surface"),
    # Winds for 0-1km shear proxy (10 m vs PBL)
    ("ugrd_10m",  "UGRD", "10 m above ground"),
    ("vgrd_10m",  "VGRD", "10 m above ground"),
    ("ugrd_80m",  "UGRD", "80 m above ground"),
    ("vgrd_80m",  "VGRD", "80 m above ground"),
    ("ugrd_pbl",  "UGRD", "planetary boundary layer"),
    ("vgrd_pbl",  "VGRD", "planetary boundary layer"),
    # Misc
    ("hail_surface",  "HAIL",   "surface"),
    ("refc_atm",      "REFC",   "entire atmosphere (considered as a single layer)"),
]


def build_hrrr_url(date: str, cycle: int, fhour: int = 1) -> str:
    return (
        f"{HRRR_S3_BASE}/hrrr.{date}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsfcf{fhour:02d}.grib2"
    )


def build_rrfs_url(date: str, cycle: int, fhour: int = 1) -> str:
    """Return the S3 URL for a RRFS-A CONUS 2D-field forecast file.

    Args:
        date:  YYYYMMDD string
        cycle: model cycle hour (0-23)
        fhour: forecast hour (0-18 for hourly cycles)
    """
    return (
        f"{RRFS_S3_BASE}/rrfs_a/rrfs.{date}/{cycle:02d}/"
        f"rrfs.t{cycle:02d}z.2dfld.3km.f{fhour:03d}.conus.grib2"
    )


async def resolve_latest_hrrr_url(fetcher: "NOAAIndexFetcher", fhour: int = 1) -> str:
    """Probe recent HRRR cycles (newest first) and return the first available URL."""
    now = datetime.now(timezone.utc)
    for hours_back in range(2, 8):
        candidate = now - timedelta(hours=hours_back)
        url = build_hrrr_url(candidate.strftime("%Y%m%d"), candidate.hour, fhour)
        try:
            await fetcher.fetch_idx_file(f"{url}.idx")
            logger.info("Resolved HRRR URL: %s", url)
            return url
        except Exception:
            continue
    raise RuntimeError("Could not resolve any recent HRRR run after checking 6 cycles.")


async def resolve_latest_rrfs_url(fetcher: "NOAAIndexFetcher", fhour: int = 1) -> str:
    """Probe recent RRFS cycles (newest first) and return the first available URL.

    RRFS runs every hour.  Data typically appears ~60-90 min after cycle time.
    We step back through the last 6 cycles to find one with an index file.
    Falls back to HRRR if no RRFS cycle is available.
    """
    now = datetime.now(timezone.utc)
    for hours_back in range(1, 7):
        candidate = now - timedelta(hours=hours_back)
        url = build_rrfs_url(candidate.strftime("%Y%m%d"), candidate.hour, fhour)
        try:
            await fetcher.fetch_idx_file(f"{url}.idx")
            logger.info("Resolved RRFS URL: %s", url)
            return url
        except Exception:
            continue
    logger.warning("No RRFS run found in last 6 hours; falling back to HRRR.")
    return await resolve_latest_hrrr_url(fetcher, fhour=fhour)


def _extract_lat_lon(ds: xr.Dataset, y_idx: int, x_idx: int) -> tuple[float, float]:
    if "latitude" in ds and "longitude" in ds:
        lat_val = float(ds["latitude"].values[y_idx, x_idx])
        lon_val = float(ds["longitude"].values[y_idx, x_idx])
        if lon_val > 180:
            lon_val -= 360
        return lat_val, lon_val
    return 35.0 + (y_idx * 0.01), -90.0 + (x_idx * 0.01)


def _stack_feature_blocks(features: dict) -> tuple[np.ndarray, list[str], tuple[int, int], dict[str, np.ndarray]]:
    feature_names = list(features.keys())
    first_da = features[feature_names[0]]
    grid_shape = first_da.shape
    flattened = []
    arrays_by_name = {}
    for name in feature_names:
        arr = np.asarray(features[name].compute().values, dtype=np.float32)
        vec = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        arrays_by_name[name] = vec
        flattened.append(vec)
    # Rows are grid cells, columns are features.
    matrix = np.stack(flattened, axis=1)
    return matrix, feature_names, grid_shape, arrays_by_name


async def run_pipeline(
    mock_mode: bool = False,
    model_path: str = "",
    enable_llm: bool = False,
    contract_path: str = "",
    event: str = "tornado",
    window: str = "f13-24",
    inference_backend: str = "auto",  # auto | lightgbm | julia | mock
    coverage_report_path: str = "",
    base_url: str = "",  # override URL; resolved dynamically when empty
    fhour: int = 1,      # HRRR forecast hour used when resolving dynamically
):
    """
    Main orchestration pipeline: Ingestion -> Feature Engineering -> Inference -> LLM.
    """
    logger.info("Starting Tornadocaster Pipeline (event=%s window=%s)...", event, window)

    # === PHASE 1: INGESTION ===
    logger.info("Phase 1: Ingestion")
    fetcher = NOAAIndexFetcher()
    temp_path = None
    try:
        if mock_mode:
            logger.info("Mock mode enabled. Skipping HTTP download.")
            data = np.random.rand(100, 100).astype(np.float32)
            ds = xr.Dataset({"cape": (("y", "x"), data), "cin": (("y", "x"), -data)})
            fields_for_features = {}
            for i, (field_name, _, _, _) in enumerate(FIRST_PASS_FIELD_CATALOG):
                fields_for_features[field_name] = (ds["cape"] * (1.0 + i * 0.01) + (i * 0.001)).astype(np.float32)
        else:
            if not base_url:
                base_url = await resolve_latest_hrrr_url(fetcher, fhour=fhour)
            idx_url = f"{base_url}.idx"
            idx_text = await fetcher.fetch_idx_file(idx_url)
            records = fetcher.parse_idx_text(idx_text)
            # First-pass multi-field pull to increase real feature coverage.
            requested = [(name, var, level) for (name, var, level, _) in FIRST_PASS_FIELD_CATALOG]
            fields_for_features = await fetcher.fetch_named_fields(base_url, records, requested)
            if not fields_for_features:
                target_record = fetcher.find_record(records, variable="CAPE", level="surface")
                fallback_ds = await fetcher.fetch_record_dataset(base_url, target_record)
                first_var = list(fallback_ds.data_vars.values())[0]
                fields_for_features = {"cape_surface": first_var}
            # Build representative dataset for downstream coordinate extraction.
            first_da = list(fields_for_features.values())[0]
            ds = xr.Dataset({"field0": first_da})

        # === PHASE 2: SPATIAL FEATURE ENGINEERING ===
        logger.info("Phase 2: Feature Engineering")
        derived_fields = build_first_pass_derived_fields(fields_for_features)
        fields_for_features = {**fields_for_features, **derived_fields}
        if "cape_surface" in fields_for_features and len(fields_for_features) == 1:
            feature_blocks = generate_legacy_feature_blocks(fields_for_features["cape_surface"], tile_size=50)
        else:
            feature_blocks = generate_legacy_feature_blocks_for_fields(fields_for_features, tile_size=50)
        model_input, feature_names, grid_shape, arrays_by_name = _stack_feature_blocks(feature_blocks)
        field_to_legacy_base = {name: legacy for (name, _, _, legacy) in FIRST_PASS_FIELD_CATALOG}
        field_to_legacy_base.update(FIRST_PASS_DERIVED_LEGACY_BASE)
        arrays_by_name = add_dynamic_legacy_aliases(arrays_by_name, field_to_legacy_base)
        arrays_by_name = add_first_pass_legacy_aliases(arrays_by_name)
        implemented_feature_names = list(arrays_by_name.keys())
        coverage_summary = {
            "total_features": len(feature_names),
            "implemented_features": len(feature_names),
            "missing_features": 0,
            "coverage_fraction": 1.0,
            "implemented_feature_names": implemented_feature_names,
            "missing_feature_names_sample": [],
        }

        # === PHASE 3: ML INFERENCE ===
        logger.info("Phase 3: LightGBM Inference")
        resolved_model_path = model_path
        contract = None
        if not resolved_model_path and contract_path and os.path.exists(contract_path):
            contract = load_contract(contract_path)
            rel_path = select_model_path(contract, event=event, window=window)
            candidate_name = os.path.basename(rel_path)
            local_candidate = os.path.join(
                os.path.dirname(contract_path), "models", candidate_name
            )
            if os.path.exists(local_candidate):
                resolved_model_path = local_candidate
                logger.info("Resolved model from contract: %s", resolved_model_path)
            else:
                logger.info(
                    "Contract resolved %s but local export not found at %s; using mock model.",
                    rel_path,
                    local_candidate,
                )
        elif contract_path and os.path.exists(contract_path):
            contract = load_contract(contract_path)

        # If a contract is available, align to canonical upstream feature order.
        if contract is not None:
            features_file = os.path.join(os.path.dirname(contract_path), "features_order_2005.txt")
            if os.path.exists(features_file):
                ordered_feature_names = [ln.strip() for ln in open(features_file).readlines() if ln.strip()]
                arrays_by_name = add_temporal_proxy_aliases(arrays_by_name, ordered_feature_names)
                implemented_feature_names = list(arrays_by_name.keys())
                model_input, feature_names, accounting = account_for_ordered_features(
                    arrays_by_name, ordered_feature_names, grid_shape[0] * grid_shape[1]
                )
                missing_features = accounting["zero_fill_features_sample"]
                coverage_summary = build_feature_coverage_summary(
                    implemented_feature_names=implemented_feature_names,
                    ordered_feature_names=feature_names,
                    missing_feature_names=[],
                )
                coverage_summary["accounting"] = accounting
                if accounting["zero_fill_count"] > 0:
                    logger.info(
                        "Feature accounting: direct=%d proxy=%d zero_fill=%d out of %d.",
                        accounting["direct_count"],
                        accounting["proxy_count"],
                        accounting["zero_fill_count"],
                        len(feature_names),
                    )
                if coverage_report_path:
                    report_dir = os.path.dirname(coverage_report_path)
                    if report_dir:
                        os.makedirs(report_dir, exist_ok=True)
                    with open(coverage_report_path, "w") as f:
                        json.dump(
                            {
                                "event": event,
                                "window": window,
                                "coverage": coverage_summary,
                            },
                            f,
                            indent=2,
                        )

        if resolved_model_path and os.path.exists(resolved_model_path):
            use_julia = False
            if inference_backend == "julia":
                use_julia = True
            elif inference_backend == "auto" and JuliaToLightGBMBridge.is_memory_constrained_tree_model(
                resolved_model_path
            ):
                use_julia = True

            if use_julia:
                if not JuliaSidecarRunner.is_available():
                    logger.warning(
                        "Requested/auto-selected Julia backend, but Julia runtime is unavailable. "
                        "Falling back to mock model."
                    )
                    model = JuliaToLightGBMBridge.build_mock_model(n_features=len(feature_names))
                    predictions = model.predict(model_input)
                else:
                    logger.info("Running native Julia sidecar inference: %s", resolved_model_path)
                    try:
                        predictions = JuliaSidecarRunner().predict(resolved_model_path, model_input)
                    except Exception as exc:
                        logger.warning(
                            "Julia sidecar inference failed (%s). Falling back to mock model.",
                            exc,
                        )
                        model = JuliaToLightGBMBridge.build_mock_model(n_features=len(feature_names))
                        predictions = model.predict(model_input)
            else:
                logger.info("Loading model with LightGBM bridge: %s", resolved_model_path)
                model = JuliaToLightGBMBridge.load_model_from_file(resolved_model_path)
                predictions = model.predict(model_input)
        else:
            logger.info("No model file provided; building local mock model for %d features.", len(feature_names))
            model = JuliaToLightGBMBridge.build_mock_model(n_features=len(feature_names))
            predictions = model.predict(model_input)
        max_idx = np.argmax(predictions)
        flat_y = max_idx // grid_shape[1]
        flat_x = max_idx % grid_shape[1]

        # === PHASE 4: LLM REPORTING ===
        logger.info("Phase 4: LLM Report Generation")
        coord = _extract_lat_lon(ds, flat_y, flat_x)
        mock_coords = [coord]
        mock_importances = {
            name: 1.0 / len(feature_names)
            for name in feature_names[: min(5, len(feature_names))]
        }

        if mock_mode or not enable_llm:
            report = "MOCK REPORT GENERATED SUCCESSFULLY."
        else:
            generator = ForecastDiscussionGenerator()
            report = await generator.generate_discussion(mock_coords, mock_importances)

        logger.info("Pipeline Complete. Final Result:")
        logger.info(report)
        return {
            "status": "success",
            "max_threat_coord": mock_coords[0],
            "max_prediction": float(predictions[max_idx]),
            "feature_coverage": coverage_summary,
        }

    finally:
        await fetcher.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_pipeline(mock_mode=True))
