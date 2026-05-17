import asyncio
import numpy as np
from tests.test_grid_alignment import download_nadocast_grid
import os

async def main():
    ds_nado, path = await download_nadocast_grid()
    lat = ds_nado.latitude.values
    lon = ds_nado.longitude.values
    print("Nadocast Lat Shape:", lat.shape)
    print("Nadocast Latitudes (First col):", lat[:, 0])
    print("Nadocast Longitudes (First row):", lon[0, :])
    os.remove(path)

if __name__ == "__main__":
    asyncio.run(main())
