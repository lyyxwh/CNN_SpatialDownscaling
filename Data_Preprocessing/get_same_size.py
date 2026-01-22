import os
import rasterio
import xarray as xr
import rioxarray
import numpy as np
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.transform import from_origin
from concurrent.futures import ThreadPoolExecutor, as_completed

# 该脚本用于批量处理栅格数据，包括重采样、投影和数据类型转换，
# 确保所有数据与基准栅格（modis LST）具有相同的空间参数和格式。

# 定义数据路径和变量名，请根据您的实际文件路径进行修改
data_paths = {
        #'dem': 'I:\\SRTM_DEM\\2018_1000\\SRTM_DEM_resample_clip.img',
        #'slope': 'I:\\SRTM_DEM\\2018_1000\\SRTM_DEM_resample_clip_slope.img', 
        #'aspect': 'I:\\SRTM_DEM\\2018_1000\\SRTM_DEM_resample_clip_aspect.img',
        #'clcd': 'I:\\CLCD\\2018_1000m\\CLCD_v01_2018__ProjectRaster_clip1.tif',
        'era5': r'G:\\CNN_SpatialDownscaling\\201712and201901\\ERA5_1000',
        #'smap': r'G:\CNN_SpatialDownscaling\SMAP\SPL4SMGP\SMAP\output_clip_resample_nc',
        'modis': r'G:\CNN_SpatialDownscaling\modis_lst\2018_mod\output_hourly_data',
        #'ndvi': 'G:\\CNN_SpatialDownscaling\\201712and201901\\ndvi\\2019'
}

# 变量名字典，用于从 NetCDF 文件中提取特定变量
var_names = {
    'era5': ['t2m', 'ssrd', 'strd', 'vpd', 'd2m', 'rh'],
    'smap': 'sm_surface_wetness',
    'cldas': 'TG',
    'ndvi': '_1_km_16_days_NDVI',
    'modis': 'LST_1km'  # LST 变量名未使用，因为LST文件夹中只有LST数据
}

def get_base_raster_info(data_path, var_name):
    """
    从指定的 NetCDF 文件中获取基准栅格信息，若缺失则手动设置。
    该信息（transform, crs, height, width）将用于统一所有其他栅格数据。
    """
    with xr.open_dataset(data_path) as ds:
        # 检查指定的变量名是否存在
        if var_name not in ds.variables:
            raise ValueError(f"变量 '{var_name}' 在文件中不存在。可用变量: {list(ds.variables)}")
        
        data_array = ds[var_name]
        # 自动重命名空间维度，支持大写LAT/LON
        dims = list(data_array.dims)
        rename_dict = {}
        if ('lat' in dims or 'LAT' in dims) and 'y' not in dims:
            rename_dict['lat' if 'lat' in dims else 'LAT'] = 'y'
        if ('lon' in dims or 'LON' in dims) and 'x' not in dims:
            rename_dict['lon' if 'lon' in dims else 'LON'] = 'x'
        if rename_dict:
            data_array = data_array.rename(rename_dict)
        # 强制设置空间维度
        try:
            data_array = data_array.rio.set_spatial_dims(y_dim="y", x_dim="x", inplace=False)
        except Exception:
            pass

        # 检查 CRS（坐标参考系统），如果缺失则手动设置
        if not data_array.rio.crs:
            print(f"文件 '{data_path}' 缺少 CRS，正在手动设置。")
            base_crs = CRS.from_epsg(4326) # 假设默认使用 WGS84
            data_array = data_array.rio.write_crs(base_crs)
        else:
            base_crs = data_array.rio.crs

        # 检查 transform（仿射变换），如果缺失则手动设置
        if not data_array.rio.transform().is_identity:
            base_transform = data_array.rio.transform()
        else:
            print(f"文件 '{data_path}' 缺少 transform，正在手动设置。")
            # 支持大写LON/LAT
            lon_name = 'lon' if 'lon' in data_array.coords else 'LON'
            lat_name = 'lat' if 'lat' in data_array.coords else 'LAT'
            lon_min = data_array[lon_name].min().item()
            lat_max = data_array[lat_name].max().item()
            # 根据经验设置默认的分辨率（1km）
            base_transform = from_origin(lon_min, lat_max, 0.008333333333, 0.008333333333)
            data_array = data_array.rio.write_transform(base_transform)

        # 获取基准栅格的高度和宽度
        base_height, base_width = data_array.rio.shape
    return base_transform, base_crs, base_height, base_width

def process_file(input_path, output_dir, base_info, key, var_names):
    """
    处理单个栅格文件，包括重采样和数据类型转换，并保留输出文件格式。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"警告: 文件不存在 {input_path}")
        return
    
    # 从基准信息中解包参数
    base_transform, base_crs, base_height, base_width = base_info
    base_name, ext = os.path.splitext(os.path.basename(input_path))
    
    print(f"正在处理 {input_path} ...")
    
    try:
        # 如果文件是 NetCDF 格式
        if ext == '.nc':
            with xr.open_dataset(input_path) as ds:
                # 创建一个空字典来存储重采样后的变量
                resampled_vars = {}
                var_list = []
                
                # 如果在 var_names 字典中找到了对应的 key
                if key in var_names:
                    # 将变量名转换为列表以方便遍历
                    var_list = var_names[key] if isinstance(var_names[key], list) else [var_names[key]]
                else:
                    # 如果没有指定变量名，则处理文件中的所有数据变量
                    var_list = list(ds.data_vars.keys())
                    print("未找到指定变量名，将尝试处理所有数据变量。")
                
                for var_name in var_list:
                    if var_name in ds.variables:
                        data_array = ds[var_name]
                        # 强制写入LST的CRS，保证一致
                        data_array = data_array.rio.write_crs(base_crs, inplace=False)
                        # 执行重投影和重采样
                        resampled_data = data_array.rio.reproject(
                            dst_crs=base_crs,
                            resampling=Resampling.bilinear,
                            shape=(base_height, base_width),
                            transform=base_transform
                        )
                        # 将重采样后的数据添加到字典中
                        resampled_vars[var_name] = resampled_data.astype(np.float32)
                    else:
                        print(f"警告: 文件 '{input_path}' 中未找到变量 '{var_name}'。")
                
                # 如果字典中包含数据，则将其保存到单个 .nc 文件
                if resampled_vars:
                    # 将字典转换为 xarray.Dataset
                    combined_ds = xr.Dataset(resampled_vars)
                    output_path = os.path.join(output_dir, f"{base_name}_resampled.nc")
                    # 保存为 NetCDF4 格式
                    combined_ds.to_netcdf(output_path, format="NETCDF4")
                    print(f"已将所有变量保存到 {output_path}")

        # 如果文件是 .img 或 .tif 格式
        elif ext in ['.img', '.tif']:
            with rasterio.open(input_path) as src:
                # 根据 'clcd' 键选择重采样方法（最近邻用于分类数据，双线性用于连续数据）
                resampling_method = Resampling.nearest if 'clcd' in key else Resampling.bilinear
                
                # 读取数据并将其转换为 float32 以确保兼容性
                data = src.read(
                    out_shape=(src.count, base_height, base_width),
                    resampling=resampling_method
                ).astype(np.float32)

                profile = src.profile
                profile.update({
                    'dtype': 'float32',  # 明确将数据类型设置为 float32，避免 'float16' 错误
                    'height': base_height,
                    'width': base_width,
                    'transform': base_transform,
                    'crs': base_crs
                })
                
                output_path = os.path.join(output_dir, f"{base_name}.tif")
                # 使用新的配置文件写入数据
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data)
                
                print(f"成功保存到 {output_path}")

    except Exception as e:
        # 捕获并打印处理过程中的错误
        print(f"处理文件 {input_path} 时出错: {e}")
        return

def process_files_in_parallel(input_dir, output_dir, base_info, key, var_names, max_workers=None):
    """
    使用线程池并行处理 input_dir 下的所有栅格文件。
    适用于 IO 密集型重采样任务，避免在 Windows 上多进程序列化问题。
    """
    exts = ('.nc', '.tif', '.img')
    file_list = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(exts):
                file_list.append(os.path.join(root, f))

    if not file_list:
        print(f"未找到可处理的文件: {input_dir}")
        return

    max_workers = max_workers or min(len(file_list), os.cpu_count() or 4)
    print(f"并行处理 {len(file_list)} 个文件，线程数 = {max_workers}")

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_file, p, output_dir, base_info, key, var_names): p for p in file_list}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"处理失败: {path} -> {e}")
                failures.append((path, str(e)))

    if failures:
        print(f"\n部分文件处理失败: {len(failures)}")
        for p, err in failures[:10]:
            print(f" - {p}: {err}")

def main():
    """主函数，负责执行数据处理流程。"""
    lst_path = None
    # 遍历MODIS文件夹，找到第一个 .nc 文件作为基准栅格
    for root, _, files in os.walk(data_paths['modis']):
        for file in files:
            if file.endswith('.nc'):
                lst_path = os.path.join(root, file)
                break
        if lst_path:
            break
            
    if not lst_path:
        print("未在 LST 文件夹中找到任何 .nc 文件。")
        return
        
    try:
        # 获取基准栅格信息
        base_info = get_base_raster_info(lst_path, var_names['modis'])
        print("\n--- 基准栅格信息获取成功 ---")
        print(f"分辨率: {base_info[0].a} x {abs(base_info[0].e)}")
        print(f"尺寸: {base_info[3]} x {base_info[2]}")
        print("------------------------------\n")
        
    except (ValueError, FileNotFoundError) as e:
        print(f"无法获取基准栅格信息: {e}")
        return

    # 创建输出目录
    output_base_dir = 'I:\\CNN_SpatialDownscaling\\Processed_Data1'
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 遍历所有数据路径并进行处理
    for key, path in data_paths.items():
        output_dir = os.path.join(output_base_dir, f'processed_{key}')
        os.makedirs(output_dir, exist_ok=True)
        
        if os.path.isdir(path):
            print(f"--- 并行处理文件夹 {path} ---")
            process_files_in_parallel(path, output_dir, base_info, key, var_names)
        else:
            print(f"--- 正在处理单个文件 {path} ---")
            process_file(path, output_dir, base_info, key, var_names)

    print("\n所有文件处理完毕！")

# 确保脚本作为主程序运行时才执行 main() 函数
if __name__ == "__main__":
    main()
