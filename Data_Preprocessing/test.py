import xarray as xr
import numpy as np

file = r"G:\CNN_SpatialDownscaling\MODIS11\2018_mod11\output_hourly_data\MOD11_20171217_21.nc "
ds = xr.open_dataset(file)
arr = ds["LST_1km"].values

print("min:", np.nanmin(arr))
print("max:", np.nanmax(arr))
print("NaN比例:", np.isnan(arr).sum() / arr.size * 100)
