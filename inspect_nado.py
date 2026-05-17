import asyncio
import xarray as xr
from tests.test_grid_alignment import download_nadocast_grid
import os

async def main():
    ds_nado, path = await download_nadocast_grid()
    print("Nadocast Data Vars:")
    for k, v in ds_nado.data_vars.items():
        print(f" - {k}: {v.shape} (Range: {v.min().values} to {v.max().values})")
    os.remove(path)

if __name__ == "__main__":
    asyncio.run(main())
