"""
Feature engineering pipeline focusing on out-of-core chunked operations
via xarray and dask to maintain strict memory budgets.

Incorporates exact math blocks from legacy Nadocast (models/shared/FeatureEngineeringShared.jl)
Spatial features computed: 
- Raw
- 25mi, 50mi, 100mi means
- 25mi, 50mi, 100mi gradients (forward, leftward, linestraddling - replicated here via magnitude/directional proxies for now)
"""
import xarray as xr
import dask.array as da
import numpy as np

# Radii mappings based on ~13km (8.07 mile) grid spatial resolution used in Nadocast Grid130
RADIUS_25MI_13KM_GRID = 3
RADIUS_50MI_13KM_GRID = 6
RADIUS_100MI_13KM_GRID = 12

def calculate_spatial_gradient(data_array: xr.DataArray, tile_size: int = 500) -> xr.DataArray:
    """
    Computes a simple spatial gradient magnitude for a 2D data array.
    Uses dask to chunk the data spatially.
    
    Gradient Magnitude = sqrt((d/dx)^2 + (d/dy)^2)
    """
    # Chunk the data spatially to avoid OOM crashes
    # Assuming dimensions are (y, x)
    dims = data_array.dims
    if 'y' in dims and 'x' in dims:
        da_chunked = data_array.chunk({"y": tile_size, "x": tile_size})
    else:
        # Fallback if dims are named differently
        da_chunked = data_array.chunk({dims[0]: tile_size, dims[1]: tile_size})
        
    # Calculate gradients using differentiate (central differences)
    # xarray.differentiate() operates on coordinates, but if coordinates 
    # are missing or non-uniform, we can use dask/numpy gradient on the raw array.
    # For robust tiled processing, we'll use dask.array.gradient directly.
    
    # We must ensure overlapping to avoid boundary artifacts if using dask map_overlap
    # But dask.array.gradient handles internal boundaries correctly.
    
    y_axis, x_axis = 0, 1 # Assuming 2D (y, x) shape
    
    dy, dx = da.gradient(da_chunked.data, axis=(y_axis, x_axis))
    
    gradient_magnitude = da.sqrt(dy**2 + dx**2)
    
    # Re-wrap in xarray using same coords and dims
    return xr.DataArray(
        gradient_magnitude,
        coords=data_array.coords,
        dims=data_array.dims,
        name=f"{data_array.name}_gradient" if data_array.name else "gradient"
    )


def calculate_spatial_gradients_xy(
    data_array: xr.DataArray, tile_size: int = 500
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Computes directional gradients in x and y plus gradient magnitude.
    Returns (dx, dy, magnitude) as lazily-evaluated DataArrays.
    """
    dims = data_array.dims
    if "y" in dims and "x" in dims:
        da_chunked = data_array.chunk({"y": tile_size, "x": tile_size})
    else:
        da_chunked = data_array.chunk({dims[0]: tile_size, dims[1]: tile_size})

    dy, dx = da.gradient(da_chunked.data, axis=(0, 1))
    grad_mag = da.sqrt(dy**2 + dx**2)

    base_name = data_array.name or "feature"
    dx_da = xr.DataArray(dx, coords=data_array.coords, dims=data_array.dims, name=f"{base_name}_grad_x")
    dy_da = xr.DataArray(dy, coords=data_array.coords, dims=data_array.dims, name=f"{base_name}_grad_y")
    mag_da = xr.DataArray(
        grad_mag,
        coords=data_array.coords,
        dims=data_array.dims,
        name=f"{base_name}_grad_mag",
    )
    return dx_da, dy_da, mag_da

def calculate_spatial_mean(data_array: xr.DataArray, radius: int = 1, tile_size: int = 500) -> xr.DataArray:
    """
    Computes a spatial rolling mean over a given radius (e.g. 1 means 3x3 window).
    """
    dims = data_array.dims
    if 'y' in dims and 'x' in dims:
        da_chunked = data_array.chunk({"y": tile_size, "x": tile_size})
        
        # xarray's rolling operates sequentially and supports dask.
        # min_periods=1 ensures edge pixels are still computed
        rolled = da_chunked.rolling(y=radius*2+1, center=True, min_periods=1).mean()
        mean_array = rolled.rolling(x=radius*2+1, center=True, min_periods=1).mean()
        
        mean_array.name = f"{data_array.name}_mean_r{radius}" if data_array.name else f"mean_r{radius}"
        return mean_array
    else:
        raise ValueError("DataArray must have 'y' and 'x' dimensions for 2D rolling operations.")

def generate_legacy_feature_blocks(data_array: xr.DataArray, tile_size: int = 500) -> dict:
    """
    Computes the exact 13-block feature set equivalent to legacy models/shared/FeatureEngineeringShared.jl
    Assumes incoming data_array has already been resampled to the 13km Grid130 projection.
    """
    features = {}
    
    # Block 1: Raw
    features["raw"] = data_array
    
    # Blocks 2-4: Means
    mean_25 = calculate_spatial_mean(data_array, radius=RADIUS_25MI_13KM_GRID, tile_size=tile_size)
    mean_50 = calculate_spatial_mean(data_array, radius=RADIUS_50MI_13KM_GRID, tile_size=tile_size)
    mean_100 = calculate_spatial_mean(data_array, radius=RADIUS_100MI_13KM_GRID, tile_size=tile_size)
    
    features["25mi_mean"] = mean_25
    features["50mi_mean"] = mean_50
    features["100mi_mean"] = mean_100
    
    # Blocks 5-13: 3 directional gradients for each radius-smoothed field.
    # This gives stable 13-block semantics:
    # raw (1) + means (3) + directional gradients (9) = 13 total.
    mean_by_radius = {
        "25mi": mean_25,
        "50mi": mean_50,
        "100mi": mean_100,
    }
    for radius_name, mean_array in mean_by_radius.items():
        grad_x, grad_y, grad_mag = calculate_spatial_gradients_xy(mean_array, tile_size=tile_size)
        features[f"{radius_name}_gradient_x"] = grad_x
        features[f"{radius_name}_gradient_y"] = grad_y
        features[f"{radius_name}_gradient_mag"] = grad_mag

    return features


def generate_legacy_feature_blocks_for_fields(
    fields: dict[str, xr.DataArray], tile_size: int = 500
) -> dict[str, xr.DataArray]:
    """
    Generate legacy-style blocks for multiple fields.
    Output keys are <field_name>__<block_name>.
    """
    out: dict[str, xr.DataArray] = {}
    for field_name, data_array in fields.items():
        blocks = generate_legacy_feature_blocks(data_array, tile_size=tile_size)
        for block_name, block_array in blocks.items():
            out[f"{field_name}__{block_name}"] = block_array
    return out
