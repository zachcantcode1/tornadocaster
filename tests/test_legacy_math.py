import numpy as np
import xarray as xr
from src.features.spatial import generate_legacy_feature_blocks

def test_legacy_blocks():
    print("Testing Legacy FeatureBlock generation...")
    # Simulate a 13km Grid130 shape (e.g., 226 x 387 from the legacy GRIB check)
    data = np.random.rand(226, 387).astype(np.float32)
    da = xr.DataArray(data, dims=("y", "x"), name="CAPE")
    
    features = generate_legacy_feature_blocks(da, tile_size=100)
    
    expected_keys = [
        "raw",
        "25mi_mean",
        "50mi_mean",
        "100mi_mean",
        "25mi_gradient_x",
        "25mi_gradient_y",
        "25mi_gradient_mag",
        "50mi_gradient_x",
        "50mi_gradient_y",
        "50mi_gradient_mag",
        "100mi_gradient_x",
        "100mi_gradient_y",
        "100mi_gradient_mag",
    ]
    
    for key in expected_keys:
        assert key in features
        # Trigger lazy dask evaluation to ensure no crashing
        computed = features[key].compute()
        assert computed.shape == (226, 387)

    assert len(features) == 13
        
    print(f"Sanity Check: Automatically generated {len(features)} matching math blocks (Means/Gradients) across Grid130 dimensions (226x387).")

if __name__ == "__main__":
    test_legacy_blocks()
