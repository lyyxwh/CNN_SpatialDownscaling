import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, calculate_default_transform, Resampling
import xarray as xr
import numpy as np

def extract_and_merge_img(shp_path, tiff_path, output_nc_path, output_tif_path):
    # 读取矢量数据
    gdf = gpd.read_file(shp_path)

    with rasterio.open(tiff_path) as src:
        # 将shapefile投影为与raster一致的坐标系
        gdf = gdf.to_crs(src.crs)

        # 进行裁剪（原始投影）
        try:
            img_data, out_transform = mask(src, gdf.geometry, crop=True)
        except ValueError:
            print("ERROR: Shapefile and raster do not overlap.")
            print("Shapefile bounds:", gdf.total_bounds)
            print("Raster bounds:", src.bounds)
            return

        # 准备重投影到 WGS84
        dst_crs = 'EPSG:4326'
        dst_transform, width, height = calculate_default_transform(
            src.crs, dst_crs, img_data.shape[2], img_data.shape[1], *src.bounds
        )

        # 创建目标数组
        dst_data = np.empty((img_data.shape[0], height, width), dtype=img_data.dtype)

        for i in range(img_data.shape[0]):
            reproject(
                source=img_data[i],
                destination=dst_data[i],
                src_transform=out_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )

        # 创建经纬度坐标
        x_coords = np.linspace(dst_transform.c, dst_transform.c + dst_transform.a * width, width)
        y_coords = np.linspace(dst_transform.f, dst_transform.f + dst_transform.e * height, height)

        # 构建 xarray.DataArray
        data_array = xr.DataArray(
            dst_data,
            dims=["band", "y", "x"],
            coords={
                "band": np.arange(1, dst_data.shape[0] + 1),
                "y": y_coords,
                "x": x_coords,
            },
            attrs={
                "crs": dst_crs,
                "transform": dst_transform
            }
        )

        # 保存为 NetCDF 文件
        data_array.to_netcdf(output_nc_path)

        # 保存为 GeoTIFF 文件
        out_meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": dst_data.shape[0],
            "dtype": dst_data.dtype,
            "crs": dst_crs,
            "transform": dst_transform
        }

        with rasterio.open(output_tif_path, "w", **out_meta) as dest:
            dest.write(dst_data)

        print(f"Saved NetCDF: {output_nc_path}")
        print(f"Saved GeoTIFF: {output_tif_path}")


if __name__ == "__main__":
    clip_data = "G:\\Agr-qu\\clip.shp"
    input_img_path = "G:\\CNN_SpatialDownscaling\\CLCD（土地覆盖）\\CLCD_v01_2018_albert.tif"
    output_nc_path = "G:\\CNN_SpatialDownscaling\\CLCD_v01_2018_albert.nc"
    output_tif_path = "G:\\CNN_SpatialDownscaling\\CLCD_v01_2018_albert.tif"
    input_img_path1 = "G:\\CNN_SpatialDownscaling\\SRTM_DEM\\SRTM_DEM.img" 
    output_nc_path1 = "G:\\CNN_SpatialDownscaling\\SRTM_DEM\\SRTM_DEM_clip.nc"  
    output_tif_path1 = "G:\\CNN_SpatialDownscaling\\SRTM_DEM\\SRTM_DEM_clip.tif"

    extract_and_merge_img(clip_data, input_img_path, output_nc_path, output_tif_path)
    extract_and_merge_img(clip_data, input_img_path1, output_nc_path1, output_tif_path1)
