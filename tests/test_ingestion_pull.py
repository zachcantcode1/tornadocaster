import asyncio
import os
import tempfile
import xarray as xr
import numpy as np
from src.ingestion.noaa_fetcher import NOAAIndexFetcher

async def test_small_pull():
    base_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240516/conus/hrrr.t00z.wrfsfcf01.grib2"
    idx_url = f"{base_url}.idx"

    fetcher = NOAAIndexFetcher()
    temp_path = None
    try:
        # 1. Fetch the index file
        idx_text = await fetcher.fetch_idx_file(idx_url)
        records = fetcher.parse_idx_text(idx_text)
        
        print(f"Parsed {len(records)} records from index.")
        
        # 2. Find CAPE at surface
        target_record = fetcher.find_record(records, variable="CAPE", level="surface")
        
        # 3. Pull strictly the bytes for that single variable
        grib_bytes = await fetcher.fetch_byte_range(
            base_url, 
            target_record.offset, 
            target_record.next_offset
        )
        
        assert len(grib_bytes) > 0, "Downloaded bytes are empty"
        print(f"Successfully downloaded {len(grib_bytes)} bytes.")
        
        # Save to a temporary file for cfgrib to read
        fd, temp_path = tempfile.mkstemp(suffix=".grib2")
        with os.fdopen(fd, 'wb') as f:
            f.write(grib_bytes)
            
        print(f"Saved binary chunk to {temp_path}")
        
        # 4. Open with xarray and cfgrib to verify properties
        ds = xr.open_dataset(temp_path, engine="cfgrib")
        print("\nDataset loaded successfully:")
        print(ds)
        
        # 5. Assert spatial grid and float integrity
        # HRRR CONUS typical dimensions: ~1059 x 1799
        cape_values = ds.cape.values
        height, width = cape_values.shape
        print(f"\nCAPE shape: {height}x{width}")
        
        assert height > 1000 and width > 1700, f"Unexpected grid dimensions: {height}x{width}"
        assert not np.isnan(cape_values).all(), "All downloaded values are NaN!"
        max_cape = np.nanmax(cape_values)
        print(f"Maximum CAPE value in this chunk: {max_cape} J/kg")
        
        print("\nStep 1 Complete! Grib decode and verification successful.")
        
    finally:
        await fetcher.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    asyncio.run(test_small_pull())
