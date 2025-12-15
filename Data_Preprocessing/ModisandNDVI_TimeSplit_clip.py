import xarray as xr
import rioxarray  
import geopandas as gpd
from shapely.geometry import mapping
import os  
import netCDF4 as nc


def separate_modis_by_time(input_file, clip_data, output_folder,first_name):
    with nc.Dataset(input_file) as f:
        # time是f文件中的某个变量variables
        #f['Day_view_time'].set_collective(True)  #这是MOD11A1的变量
        f['time'].set_collective(True)
        ds = xr.open_dataset(
            input_file,
            decode_cf=True,
            engine="netcdf4"  # 强制使用netcdf4引擎，兼容性更好
            )                      

        # 提前准备 shapefile
        gdf = gpd.read_file(clip_data)

        # 循环每个时间点
        for idx in range(len(ds.time)):
            ds_slice = ds.isel(time=idx)
            ds_slice.attrs = ds.attrs  # 复制全局属性
            ds_slice["time"].attrs = ds["time"].attrs  # 保留时间属性

            # 按变量迭代裁剪
            out_vars = {}
            for var in ds_slice.data_vars:
                da_var = ds_slice[var]
                if "lon" in da_var.dims and "lat" in da_var.dims:
                    da_var = da_var.load()  # 关键：先加载到内存
                    da_var = da_var.rio.set_spatial_dims("lon", "lat")         
                    da_var = da_var.rio.write_crs("EPSG:4326")                
                    clipped_var = da_var.rio.clip(
                        gdf.geometry.apply(mapping),
                        gdf.crs,
                        drop=False
                    )                                                          
                    out_vars[var] = clipped_var
                else:
                    print(f"跳过变量 {var}，因为它不包含 'lon' 和 'lat' 维度")

            ds_out = xr.Dataset(out_vars, coords={"time": ds_slice.time})
            # 使用时间值命名输出文件
            time_str = str(ds_slice.time.values).replace(":", "").replace(" ", "_")
            out_name = os.path.join(output_folder, f"{first_name}_{time_str}.nc")
            os.makedirs(os.path.dirname(out_name), exist_ok=True)
            # 写入前如目标文件已存在则先删除，避免HDF error
            if os.path.exists(out_name):
                os.remove(out_name)
            ds_out.to_netcdf(
                out_name,
                format="NETCDF4_CLASSIC",
                encoding={v: {"zlib": True} for v in ds_out.data_vars},
                engine="netcdf4"
            )
            ds_out.close()

            ds.close()
            print(f"已处理时间点 {idx + 1}/{len(ds.time)}，输出文件：{out_name}")

if __name__ == '__main__':
    input_file = r'G:\\CNN_SpatialDownscaling\\201712and201901\\MOD13A2.061_1km_aid00012017.nc'
    input_file1 = r'G:\\CNN_SpatialDownscaling\\201712and201901\\MOD13A2.061_1km_aid0001 (1).nc'
    clip_data = r'G:\\Agr-qu\\clip.shp'
    output_folder = r'G:\\CNN_SpatialDownscaling\\201712and201901\\ndvi\\2017'
    output_folder1 = r'G:\\CNN_SpatialDownscaling\\201712and201901\\ndvi\\2019'

    separate_modis_by_time(input_file, clip_data, output_folder,"NDVI")
    separate_modis_by_time(input_file1, clip_data, output_folder1,"NDVI")
