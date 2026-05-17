import asyncio
import os
import tempfile
import urllib.request
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.spatial import cKDTree

from src.ingestion.noaa_fetcher import NOAAIndexFetcher
from src.models.lgb_bridge import JuliaToLightGBMBridge
from tests.test_grid_alignment import download_nadocast_grid, download_hrrr_cape

async def run_bridging_and_plotting():
    print("Downloading Legacy Nadocast Output Grid (Grid130 / 226x387)...")
    ds_nado, nado_path = await download_nadocast_grid()
    nado_lat = ds_nado.latitude.values
    nado_lon = ds_nado.longitude.values
    nado_lon = np.where(nado_lon > 180, nado_lon - 360, nado_lon)
    
    # Actually extract the probability values of Nadocast
    # Variable is typically 'unknown' or similar in cfgrib if not in standard tables
    # For now, let's just grab the first data array
    data_var_name = list(ds_nado.data_vars.keys())[0]
    nado_prob = ds_nado[data_var_name].values
    
    print("Downloading New Pipeline HRRR CAPE Grid (1059x1799)...")
    ds_hrrr, hrrr_path = await download_hrrr_cape()
    hrrr_lat = ds_hrrr.latitude.values
    hrrr_lon = ds_hrrr.longitude.values
    hrrr_lon = np.where(hrrr_lon > 180, hrrr_lon - 360, hrrr_lon)
    hrrr_cape = ds_hrrr.cape.values
    
    print("Building KDTree for fast spatial resampling (HRRR -> Grid130)...")
    hrrr_points = np.column_stack([hrrr_lat.ravel(), hrrr_lon.ravel()])
    nado_points = np.column_stack([nado_lat.ravel(), nado_lon.ravel()])
    tree = cKDTree(hrrr_points)
    
    distances, indices = tree.query(nado_points, k=1)
    hrrr_cape_resampled = hrrr_cape.ravel()[indices].reshape(nado_lat.shape)
    
    print("Creating a mock LightGBM model scaled to CAPE...")
    # Map range 0->6000 CAPE to 0->10% probability linearly for visual proxy
    X_dummy = np.linspace(0, 6000, 100).reshape(-1, 1).astype(np.float32)
    y_dummy = np.linspace(0, 10.0, 100).astype(np.float32)
    dtrain = lgb.Dataset(X_dummy, label=y_dummy)
    params = {'objective': 'regression', 'max_depth': 3, 'num_leaves': 7, 'min_data_in_bin': 1, 'min_data_in_leaf': 1}
    mock_booster = lgb.train(params, dtrain, num_boost_round=10)
    
    print("Deserializing through our Python LightGBM Bridge...")
    booster_str = mock_booster.model_to_string()
    bridged_model = JuliaToLightGBMBridge.load_model_from_json(booster_str)
    
    print("Running inference on mapped Grid130 array...")
    predictions_flat = bridged_model.predict(hrrr_cape_resampled.ravel().reshape(-1, 1))
    predictions_grid = predictions_flat.reshape(nado_lat.shape)
    
    print("Plotting Side-by-Side Validation...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    im1 = ax1.imshow(nado_prob, cmap='gist_ncar_r', origin='lower', vmin=0, vmax=10.0)
    ax1.set_title("Legacy Nadocast Output (Julia - May 16, 2024)")
    ax1.set_xlabel("X (Grid130 Indices)")
    ax1.set_ylabel("Y (Grid130 Indices)")
    fig.colorbar(im1, ax=ax1, label="Probability (%)")
    
    ax2 = axes[1]
    # Restrict proxy output bounds for equivalent visual color ramp matching
    predictions_grid = np.clip(predictions_grid, 0, 10.0)
    im2 = ax2.imshow(predictions_grid, cmap='gist_ncar_r', origin='lower', vmin=0, vmax=10.0)
    ax2.set_title("New Pipeline Mock Inference (Proxy LightGBM)")
    ax2.set_xlabel("X (Grid130 Indices)")
    
    # Overlay the max point we found previously for reference
    idx_max = np.nanargmax(hrrr_cape_resampled)
    max_y, max_x = np.unravel_index(idx_max, hrrr_cape_resampled.shape)
    ax2.plot(max_x, max_y, 'ro', markersize=8, label="CAPE Peak")
    ax2.legend()
    fig.colorbar(im2, ax=ax2, label="Mock Probability")
    
    plt.tight_layout()
    plt.savefig("alignment_validation.png", dpi=150)
    print("Saved plot to 'alignment_validation.png'.")
    
    os.remove(nado_path)
    os.remove(hrrr_path)

if __name__ == "__main__":
    asyncio.run(run_bridging_and_plotting())
