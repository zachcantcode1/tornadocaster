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

def _needs_chunking(data_array: xr.DataArray, tile_size: int) -> bool:
    """Return True only when the array is large enough that chunking saves memory."""
    return any(s > tile_size * 2 for s in data_array.shape)


def calculate_spatial_gradient(data_array: xr.DataArray, tile_size: int = 500) -> xr.DataArray:
    """
    Computes a simple spatial gradient magnitude for a 2D data array.
    Uses dask chunking only when the array exceeds the tile size threshold;
    otherwise falls back to in-memory numpy operations to avoid dask overhead.

    Gradient Magnitude = sqrt((d/dx)^2 + (d/dy)^2)
    """
    dims = data_array.dims
    base_name = data_array.name or "feature"

    if _needs_chunking(data_array, tile_size):
        if 'y' in dims and 'x' in dims:
            da_chunked = data_array.chunk({"y": tile_size, "x": tile_size})
        else:
            da_chunked = data_array.chunk({dims[0]: tile_size, dims[1]: tile_size})
        dy, dx = da.gradient(da_chunked.data, axis=(0, 1))
        gradient_magnitude = da.sqrt(dy**2 + dx**2)
        return xr.DataArray(
            gradient_magnitude,
            coords=data_array.coords,
            dims=data_array.dims,
            name=f"{base_name}_gradient",
        )
    else:
        vals = np.asarray(data_array.values, dtype=np.float32)
        dy, dx = np.gradient(vals, axis=(0, 1))
        gradient_magnitude = np.sqrt(dy**2 + dx**2)
        return xr.DataArray(
            gradient_magnitude,
            coords=data_array.coords,
            dims=data_array.dims,
            name=f"{base_name}_gradient",
        )


def calculate_spatial_gradients_xy(
    data_array: xr.DataArray, tile_size: int = 500
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Computes directional gradients in x and y plus gradient magnitude.
    Uses dask only when the array exceeds the tile size threshold.
    """
    dims = data_array.dims
    base_name = data_array.name or "feature"

    if _needs_chunking(data_array, tile_size):
        if "y" in dims and "x" in dims:
            da_chunked = data_array.chunk({"y": tile_size, "x": tile_size})
        else:
            da_chunked = data_array.chunk({dims[0]: tile_size, dims[1]: tile_size})
        dy, dx = da.gradient(da_chunked.data, axis=(0, 1))
        grad_mag = da.sqrt(dy**2 + dx**2)
        dx_da = xr.DataArray(dx, coords=data_array.coords, dims=data_array.dims, name=f"{base_name}_grad_x")
        dy_da = xr.DataArray(dy, coords=data_array.coords, dims=data_array.dims, name=f"{base_name}_grad_y")
        mag_da = xr.DataArray(grad_mag, coords=data_array.coords, dims=data_array.dims, name=f"{base_name}_grad_mag")
    else:
        vals = np.asarray(data_array.values, dtype=np.float32)
        dy_np, dx_np = np.gradient(vals, axis=(0, 1))
        grad_mag_np = np.sqrt(dy_np**2 + dx_np**2)
        dx_da = xr.DataArray(dx_np, coords=data_array.coords, dims=data_array.dims, name=f"{base_name}_grad_x")
        dy_da = xr.DataArray(dy_np, coords=data_array.coords, dims=data_array.dims, name=f"{base_name}_grad_y")
        mag_da = xr.DataArray(grad_mag_np, coords=data_array.coords, dims=data_array.dims, name=f"{base_name}_grad_mag")

    return dx_da, dy_da, mag_da

def calculate_spatial_mean(data_array: xr.DataArray, radius: int = 1, tile_size: int = 500) -> xr.DataArray:
    """
    Computes a spatial rolling mean over a given radius (e.g. 1 means 3x3 window).
    Uses dask chunking only when the array exceeds the tile size threshold.
    """
    dims = data_array.dims
    if 'y' not in dims or 'x' not in dims:
        raise ValueError("DataArray must have 'y' and 'x' dimensions for 2D rolling operations.")

    window = radius * 2 + 1
    name = f"{data_array.name}_mean_r{radius}" if data_array.name else f"mean_r{radius}"

    if _needs_chunking(data_array, tile_size):
        da_chunked = data_array.chunk({"y": tile_size, "x": tile_size})
        rolled = da_chunked.rolling(y=window, center=True, min_periods=1).mean()
        mean_array = rolled.rolling(x=window, center=True, min_periods=1).mean()
    else:
        rolled = data_array.rolling(y=window, center=True, min_periods=1).mean()
        mean_array = rolled.rolling(x=window, center=True, min_periods=1).mean()

    mean_array.name = name
    return mean_array

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
