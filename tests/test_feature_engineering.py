import xarray as xr
import numpy as np
import dask
import pytest
from src.features.spatial import calculate_spatial_gradient, calculate_spatial_mean

def test_chunked_spatial_operations():
    # 1. Create a dummy grid matching HRRR CONUS scale (e.g. 1059 x 1799)
    # Give it a known analytical pattern (e.g., a simple linear slope) to verify gradients
    
    height, width = 1059, 1799
    
    # Linear slope in X direction: 0, 1, 2, ..., 1798
    # Constant in Y direction.
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    
    # Broadcast to 2D
    X, Y = np.meshgrid(x_coords, y_coords)
    data = X.astype(np.float32) * 2.0 # Gradient in X should be strictly 2.0
    
    da = xr.DataArray(
        data, 
        dims=("y", "x"), 
        coords={"y": y_coords, "x": x_coords},
        name="test_var"
    )
    
    # 2. Test Gradient Calculation
    # We tile at 500x500. Dask will compute the blocks seamlessly.
    grad = calculate_spatial_gradient(da, tile_size=500)
    
    # Confirm it is still a dask array (lazy evaluation)
    assert grad.chunks is not None, "Data was unexpectedly computed eagerly!"
    
    # Compute it into memory to assert values
    computed_grad = grad.compute()
    
    # Since it's a constant slope in X of 2.0, the gradient magnitude in X is 2, in Y is 0.
    # So magnitude = sqrt(2^2 + 0^2) = 2.0 across the interior.
    # At the edges, np.gradient uses forward/backward differences, which are also 2.0 for a linear sequence.
    np.testing.assert_allclose(computed_grad.values, 2.0, rtol=1e-5)
    
    print("Gradient test passed. Spatial chunking correctly handled boundaries.")

    # 3. Test Mean Calculation
    # We test a 3x3 window (radius=1)
    mean_da = calculate_spatial_mean(da, radius=1, tile_size=500)
    assert mean_da.chunks is not None
    
    computed_mean = mean_da.compute()
    
    # For a linear sequence (0, 2, 4...), the mean of a symmetric window is equal to the center value!
    # So the mean array should equal the original array (except maybe corners, but rolling with min_periods=1 does average available).
    # Since rolling average over [0,2] at the edge is 1.0 vs original 0.0, we assert just the interior.
    interior_original = da.values[1:-1, 1:-1]
    interior_mean = computed_mean.values[1:-1, 1:-1]
    np.testing.assert_allclose(interior_mean, interior_original, rtol=1e-5)
    
    print("Spatial mean test passed. Memory remained unbound until compute().")

if __name__ == "__main__":
    test_chunked_spatial_operations()
    print("Step 2 basic architecture checks out!")
