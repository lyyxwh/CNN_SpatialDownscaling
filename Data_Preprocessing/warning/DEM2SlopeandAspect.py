import richdem as rd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import xarray as xr
import os

def derive_slope_aspect_from_dem(dem_path, slope_tif_path, aspect_tif_path, slope_nc_path, aspect_nc_path):
    # 1. 读取DEM数据
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_transform = src.transform
        dem_crs = src.crs
        dem_profile = src.profile

    # 2. 用 richdem 计算坡度和坡向（单位为度）
    dem_rd = rd.rdarray(dem_data, no_data=dem_profile['nodata'])
    dem_rd.projection = str(dem_crs)

    slope = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
    aspect = rd.TerrainAttribute(dem_rd, attrib='aspect')

    # 3. 将 slope 和 aspect 重投影为 WGS84
    dst_crs = 'EPSG:4326'
    transform, width, height = calculate_default_transform(
        dem_crs, dst_crs, dem_data.shape[1], dem_data.shape[0], *src.bounds
    )

    def reproject_array(src_array):
        dst_array = np.empty((height, width), dtype=np.float32)
        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=dem_transform,
            src_crs=dem_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        return dst_array

    slope_reproj = reproject_array(slope)
    aspect_reproj = reproject_array(aspect)

    # 4. 写入GeoTIFF文件（坡度）
    tif_profile = dem_profile.copy()
    tif_profile.update({
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'transform': transform,
        'crs': dst_crs,
        'count': 1,
        'dtype': 'float32'
    })

    with rasterio.open(slope_tif_path, 'w', **tif_profile) as dst:
        dst.write(slope_reproj, 1)

    with rasterio.open(aspect_tif_path, 'w', **tif_profile) as dst:
        dst.write(aspect_reproj, 1)

    # 5. 写入NetCDF文件（用xarray）
    y_coords = np.linspace(transform.f, transform.f + transform.e * height, height)
    x_coords = np.linspace(transform.c, transform.c + transform.a * width, width)

    slope_da = xr.DataArray(
        slope_reproj,
        dims=['y', 'x'],
        coords={'y': y_coords, 'x': x_coords},
        attrs={'units': 'degrees', 'description': 'Slope', 'crs': dst_crs}
    )
    aspect_da = xr.DataArray(
        aspect_reproj,
        dims=['y', 'x'],
        coords={'y': y_coords, 'x': x_coords},
        attrs={'units': 'degrees', 'description': 'Aspect', 'crs': dst_crs}
    )

    slope_da.to_netcdf(slope_nc_path)
    aspect_da.to_netcdf(aspect_nc_path)

    print("✔ 坡度与坡向计算完成并已输出：")
    print("GeoTIFF:")
    print(f"  - {slope_tif_path}")
    print(f"  - {aspect_tif_path}")
    print("NetCDF:")
    print(f"  - {slope_nc_path}")
    print(f"  - {aspect_nc_path}")

if __name__ == "__main__":
    dem_path = "G:\\CNN_SpatialDownscaling\\SRTM_DEM\\SRTM_DEM_clip.tif"
    slope_tif = "G:\\CNN_SpatialDownscaling\\SRTM_DEM\\SRTM_DEM_slope.tif"
    aspect_tif = "G:\\CNN_SpatialDownscaling\\SRTM_DEM\\SRTM_DEM_aspect.tif"
    slope_nc = "G:\\CNN_SpatialDownscaling\\SRTM_DEM\\SRTM_DEM_slope.nc"
    aspect_nc = "G:\\CNN_SpatialDownscaling\\SRTM_DEM\\SRTM_DEM_aspect.nc"

    derive_slope_aspect_from_dem(dem_path, slope_tif, aspect_tif, slope_nc, aspect_nc)
