import asyncio
import os
import tempfile
import numpy as np
import xarray as xr
import lightgbm as lgb
from memory_profiler import memory_usage

from src.ingestion.noaa_fetcher import NOAAIndexFetcher
from src.features.spatial import calculate_spatial_gradient
from src.models.lgb_bridge import JuliaToLightGBMBridge
from src.llm.llm_report import ForecastDiscussionGenerator

async def run_real_pipeline_without_llm():
    print("Starting Real Data Pipeline Test...")
    
    # === PHASE 1: INGESTION ===
    print("Phase 1: Ingestion (Real Data)")
    base_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240516/conus/hrrr.t00z.wrfsfcf01.grib2"
    fetcher = NOAAIndexFetcher()
    temp_path = None
    try:
        idx_url = f"{base_url}.idx"
        idx_text = await fetcher.fetch_idx_file(idx_url)
        records = fetcher.parse_idx_text(idx_text)
        
        target_record = next((r for r in records if r.variable == "CAPE" and r.level == "surface"), None)
        grib_bytes = await fetcher.fetch_byte_range(base_url, target_record.offset, target_record.next_offset)
        
        fd, temp_path = tempfile.mkstemp(suffix=".grib2")
        with os.fdopen(fd, 'wb') as f:
            f.write(grib_bytes)
        ds = xr.open_dataset(temp_path, engine="cfgrib")
        
        # === PHASE 2: SPATIAL FEATURE ENGINEERING ===
        print("Phase 2: Feature Engineering (Real Data)")
        cape_array = ds.cape if 'cape' in ds else list(ds.data_vars.values())[0]
        cape_grad = calculate_spatial_gradient(cape_array, tile_size=500)
        computed_features = cape_grad.compute()
        
        height, width = computed_features.shape
        print(f"Computed features shape: {height}x{width}")
        
        # === PHASE 3: ML INFERENCE ===
        print("Phase 3: LightGBM Inference (Dummy Model on Real Data shapes)")
        X_train = np.array([[0.0], [5000.0]], dtype=np.float32)
        y_train = np.array([0, 1], dtype=np.float32)
        params = {'objective': 'regression', 'min_data_in_leaf': 1, 'min_data_in_bin': 1}
        booster_orig = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=1)
        mock_lgbm_text = booster_orig.model_to_string()
        
        model = JuliaToLightGBMBridge.load_model_from_json(mock_lgbm_text)
        
        flat_features = computed_features.values.flatten().reshape(-1, 1)
        # Drop NaNs or fill NaNs because real data might have NaNs
        flat_features = np.nan_to_num(flat_features, 0)
        predictions = model.predict(flat_features)
        
        max_idx = np.argmax(predictions)
        flat_y = max_idx // width
        flat_x = max_idx % width
        
        print(f"Max threat index identified at local grid ({flat_y}, {flat_x}) with predicted value {predictions[max_idx]:.4f}")
        
        print("Real Data Pipeline ran successfully without LLM call.")
    finally:
        await fetcher.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def test_real_data_bounds():
    mem_profile = memory_usage(lambda: asyncio.run(run_real_pipeline_without_llm()))
    print(f"\nPeak Memory Consumed: {max(mem_profile) - mem_profile[0]:.2f} MiB")

if __name__ == "__main__":
    test_real_data_bounds()
