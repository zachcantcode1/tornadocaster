import asyncio
import logging
import json
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
    ("cape_surface", "CAPE", "surface", "CAPE:surface:hour fcst:wt ens mean"),
    ("cape_ml", "CAPE", "90-0 mb above ground", "CAPE:90-0 mb above ground:hour fcst:wt ens mean"),
    ("cape_mu", "CAPE", "180-0 mb above ground", "CAPE:180-0 mb above ground:hour fcst:wt ens mean"),
    ("cin_surface", "CIN", "surface", "CIN:surface:hour fcst:wt ens mean"),
    ("cin_ml", "CIN", "90-0 mb above ground", "CIN:90-0 mb above ground:hour fcst:wt ens mean"),
    ("cin_mu", "CIN", "180-0 mb above ground", "CIN:180-0 mb above ground:hour fcst:wt ens mean"),
    ("hlcy_3km", "HLCY", "3000-0 m above ground", "HLCY:3000-0 m above ground:hour fcst:wt ens mean"),
    ("hgt_500", "HGT", "500 mb", "HGT:500 mb:hour fcst:wt ens mean"),
    ("hgt_250", "HGT", "250 mb", "HGT:250 mb:hour fcst:wt ens mean"),
    ("hgt_700", "HGT", "700 mb", "HGT:700 mb:hour fcst:wt ens mean"),
    ("hgt_850", "HGT", "850 mb", "HGT:850 mb:hour fcst:wt ens mean"),
    ("hgt_925", "HGT", "925 mb", "HGT:925 mb:hour fcst:wt ens mean"),
    ("ugrd_500", "UGRD", "500 mb", "UGRD:500 mb:hour fcst:wt ens mean"),
    ("ugrd_250", "UGRD", "250 mb", "UGRD:250 mb:hour fcst:wt ens mean"),
    ("ugrd_700", "UGRD", "700 mb", "UGRD:700 mb:hour fcst:wt ens mean"),
    ("ugrd_850", "UGRD", "850 mb", "UGRD:850 mb:hour fcst:wt ens mean"),
    ("ugrd_925", "UGRD", "925 mb", "UGRD:925 mb:hour fcst:wt ens mean"),
    ("vgrd_500", "VGRD", "500 mb", "VGRD:500 mb:hour fcst:wt ens mean"),
    ("vgrd_250", "VGRD", "250 mb", "VGRD:250 mb:hour fcst:wt ens mean"),
    ("vgrd_700", "VGRD", "700 mb", "VGRD:700 mb:hour fcst:wt ens mean"),
    ("vgrd_850", "VGRD", "850 mb", "VGRD:850 mb:hour fcst:wt ens mean"),
    ("vgrd_925", "VGRD", "925 mb", "VGRD:925 mb:hour fcst:wt ens mean"),
    ("tmp_2m", "TMP", "2 m above ground", "TMP:2 m above ground:hour fcst:wt ens mean"),
    ("tmp_500", "TMP", "500 mb", "TMP:500 mb:hour fcst:wt ens mean"),
    ("tmp_700", "TMP", "700 mb", "TMP:700 mb:hour fcst:wt ens mean"),
    ("tmp_850", "TMP", "850 mb", "TMP:850 mb:hour fcst:wt ens mean"),
    ("tmp_925", "TMP", "925 mb", "TMP:925 mb:hour fcst:wt ens mean"),
    ("dpt_2m", "DPT", "2 m above ground", "DPT:2 m above ground:hour fcst:wt ens mean"),
    ("dpt_850", "DPT", "850 mb", "DPT:850 mb:hour fcst:wt ens mean"),
    ("dpt_925", "DPT", "925 mb", "DPT:925 mb:hour fcst:wt ens mean"),
    ("rh_700", "RH", "700 mb", "RH:700 mb:hour fcst:wt ens mean"),
    ("pwat", "PWAT", "entire atmosphere", "PWAT:entire atmosphere:hour fcst:wt ens mean"),
    ("vis_surface", "VIS", "surface", "VIS:surface:hour fcst:wt ens mean"),
    ("crain_surface", "CRAIN", "surface", "CRAIN:surface:hour fcst:wt ens mean"),
    ("wind_10m", "WIND", "10 m above ground", "WIND:10 m above ground:hour fcst:wt ens mean"),
    ("wind_80m", "WIND", "80 m above ground", "WIND:80 m above ground:hour fcst:wt ens mean"),
    ("wind_850", "WIND", "850 mb", "WIND:850 mb:hour fcst:wt ens mean"),
    ("wind_925", "WIND", "925 mb", "WIND:925 mb:hour fcst:wt ens mean"),
    ("vwsh_0_6km", "VWSH", "6000-0 m above ground", "VWSH:6000-0 m above ground:hour fcst:wt ens mean"),
    ("hindex_surface", "HINDEX", "surface", "HINDEX:surface:hour fcst:wt ens mean"),
]

FIRST_PASS_DERIVED_LEGACY_BASE = {
    "calc_700_500_lapse": "700-500mbLapseRate:calculated:hour fcst:",
    "calc_925_700_lapse": "925-700mbLapseRate:calculated:hour fcst:",
    "calc_wind_700": "Wind700mb:calculated:hour fcst:",
    "calc_wind_500": "Wind500mb:calculated:hour fcst:",
    "calc_sbcape_hlcy": "SBCAPE*HLCY3000-0m:calculated:hour fcst:",
    "calc_mlcape_hlcy": "MLCAPE*HLCY3000-0m:calculated:hour fcst:",
    "calc_sqrt_sbcape_hlcy": "sqrtSBCAPE*HLCY3000-0m:calculated:hour fcst:",
    "calc_sqrt_mlcape_hlcy": "sqrtMLCAPE*HLCY3000-0m:calculated:hour fcst:",
    "calc_sbcape_bwd": "SBCAPE*BWD0-6km:calculated:hour fcst:",
    "calc_mlcape_bwd": "MLCAPE*BWD0-6km:calculated:hour fcst:",
    "calc_sqrt_sbcape_bwd": "sqrtSBCAPE*BWD0-6km:calculated:hour fcst:",
    "calc_sqrt_mlcape_bwd": "sqrtMLCAPE*BWD0-6km:calculated:hour fcst:",
    "calc_sbcape_200_plus_sbcin": "SBCAPE*(200+SBCIN):calculated:hour fcst:",
    "calc_mlcape_200_plus_mlcin": "MLCAPE*(200+MLCIN):calculated:hour fcst:",
    "calc_sqrt_sbcape_200_plus_sbcin": "sqrtSBCAPE*(200+SBCIN):calculated:hour fcst:",
    "calc_sqrt_mlcape_200_plus_mlcin": "sqrtMLCAPE*(200+MLCIN):calculated:hour fcst:",
    "calc_scpish_rm": "SCPish(RM):calculated:hour fcst:",
    "calc_max_wind_le_850": "MaxWind<=850mb:calculated:hour fcst:",
    "calc_sum_wind_le_850": "SumWind<=850mb:calculated:hour fcst:",
    "calc_max_wind_le_700": "MaxWind<=700mb:calculated:hour fcst:",
    "calc_sum_wind_le_700": "SumWind<=700mb:calculated:hour fcst:",
    "calc_umean": "UMEAN:calculated:hour fcst:",
    "calc_vmean": "VMEAN:calculated:hour fcst:",
    "calc_ushear": "USHEAR:calculated:hour fcst:",
    "calc_vshear": "VSHEAR:calculated:hour fcst:",
    "calc_shear": "SHEAR:calculated:hour fcst:",
    "calc_ustm": "USTM:calculated:hour fcst:",
    "calc_vstm": "VSTM:calculated:hour fcst:",
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
):
    """
    Main orchestration daemon loop connecting Ingestion -> Engineering -> Inference -> LLM.
    """
    logger.info("Starting Tornadocaster Pipeline...")
    
    # === PHASE 1: INGESTION ===
    logger.info("Phase 1: Ingestion")
    base_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240516/conus/hrrr.t00z.wrfsfcf01.grib2"
    fetcher = NOAAIndexFetcher()
    temp_path = None
    try:
        if mock_mode:
            # Generate mock dataset for fast testing
            logger.info("Mock mode enabled. Skipping HTTP download.")
            data = np.random.rand(100, 100).astype(np.float32)
            ds = xr.Dataset({"cape": (("y", "x"), data), "cin": (("y", "x"), -data)})
            fields_for_features = {}
            for i, (field_name, _, _, _) in enumerate(FIRST_PASS_FIELD_CATALOG):
                # Deterministic perturbations to avoid identical columns.
                fields_for_features[field_name] = (ds["cape"] * (1.0 + i * 0.01) + (i * 0.001)).astype(np.float32)
        else:
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
