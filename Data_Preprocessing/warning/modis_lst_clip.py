import xarray as xr
import rioxarray
import geopandas as gpd
import numpy as np
import glob
import os

clip_shapefile = 'G:\\Agr-qu\\clip.shp'
input_dir = 'G:\\CNN_SpatialDownscaling\\MODIS11\\2018'
output_dir = 'G:\\CNN_SpatialDownscaling\\MODIS11\\2018_clip'
os.makedirs(output_dir, exist_ok=True)
clip_gdf = gpd.read_file(clip_shapefile).to_crs('EPSG:4326')

for nc_path in glob.glob(os.path.join(input_dir, '*.nc')):
    ds = xr.open_dataset(nc_path)
    for var in ds.data_vars:
        if set(['lat', 'lon']).issubset(ds[var].dims):
            da = ds[var].rio.write_crs('EPSG:4326', inplace=False)
            da_clipped = da.rio.clip(clip_gdf.geometry, clip_gdf.crs, drop=True, invert=False)
            if da_clipped.size == 0 or da_clipped.shape[0] == 0 or da_clipped.shape[1] == 0:
                # 完全不重叠，生成全NaN
                minx, miny, maxx, maxy = clip_gdf.total_bounds
                lat_res = abs(da['lat'].values[1] - da['lat'].values[0]) if da['lat'].size > 1 else 0.01
                lon_res = abs(da['lon'].values[1] - da['lon'].values[0]) if da['lon'].size > 1 else 0.01
                lat_grid = np.arange(miny, maxy + lat_res, lat_res)
                lon_grid = np.arange(minx, maxx + lon_res, lon_res)
                ds[var] = xr.DataArray(np.full((len(lat_grid), len(lon_grid)), np.nan), dims=('lat', 'lon'), coords={'lat': lat_grid, 'lon': lon_grid})
            else:
                ds[var] = da_clipped
    # 清理所有变量的 grid_mapping 属性，防止 xarray 写入冲突
    for var in ds.data_vars:
        if 'grid_mapping' in ds[var].attrs:
            del ds[var].attrs['grid_mapping']
    ds.to_netcdf(os.path.join(output_dir, os.path.basename(nc_path)))