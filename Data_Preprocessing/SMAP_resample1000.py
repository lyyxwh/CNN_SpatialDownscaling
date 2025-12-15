import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import zoom
import os
from tqdm import tqdm
import multiprocessing
import logging
from datetime import datetime
import glob

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_grid_spacing(x_coords, y_coords, cell_lat, cell_lon):
    """
    计算原始网格的实际物理间距（米）
    
    Parameters:
    -----------
    x_coords, y_coords : array
        原始x,y坐标
    cell_lat, cell_lon : 2D arrays
        实际的纬度经度坐标
    
    Returns:
    --------
    x_spacing_m, y_spacing_m : float
        x和y方向的平均网格间距（米）
    """
    # 计算中心点的坐标转换因子
    center_lat = np.nanmean(cell_lat)
    
    # 地球半径（米）
    earth_radius = 6371000
    
    # 计算x方向间距（经度方向）
    if len(x_coords) > 1:
        dx = np.abs(x_coords[1] - x_coords[0])  # x坐标差
        # 在中心纬度处，经度的米/度转换
        meters_per_degree_lon = earth_radius * np.cos(np.radians(center_lat)) * np.pi / 180
        
        # 通过实际坐标点计算平均经度间距
        lon_diff = np.abs(np.nanmean(cell_lon[:, 1:] - cell_lon[:, :-1]))
        x_spacing_m = lon_diff * meters_per_degree_lon
    else:
        x_spacing_m = 9000  # 默认9km
    
    # 计算y方向间距（纬度方向）
    if len(y_coords) > 1:
        dy = np.abs(y_coords[1] - y_coords[0])  # y坐标差
        # 纬度的米/度转换（固定值）
        meters_per_degree_lat = earth_radius * np.pi / 180
        
        # 通过实际坐标点计算平均纬度间距
        lat_diff = np.abs(np.nanmean(cell_lat[1:, :] - cell_lat[:-1, :]))
        y_spacing_m = lat_diff * meters_per_degree_lat
    else:
        y_spacing_m = 9000  # 默认9km
    
    logger.info(f"计算得到的网格间距: x方向 {x_spacing_m:.0f}m, y方向 {y_spacing_m:.0f}m")
    
    return x_spacing_m, y_spacing_m

def create_resampled_grid(original_x, original_y, original_spacing_x_m, original_spacing_y_m, target_spacing_m):
    """
    创建重采样后的网格坐标
    
    Parameters:
    -----------
    original_x, original_y : arrays
        原始x,y坐标
    original_spacing_x_m, original_spacing_y_m : float
        原始网格的x,y方向间距（米）
    target_spacing_m : float
        目标网格间距（米）
    
    Returns:
    --------
    new_x, new_y : arrays
        新的x,y坐标
    """
    # 计算重采样比例
    x_ratio = original_spacing_x_m / target_spacing_m
    y_ratio = original_spacing_y_m / target_spacing_m
    
    logger.info(f"重采样比例: x方向 {x_ratio:.2f}, y方向 {y_ratio:.2f}")
    
    # 计算新的网格点数
    new_x_size = int(len(original_x) * x_ratio)
    new_y_size = int(len(original_y) * y_ratio)
    
    logger.info(f"网格大小: 原始 {len(original_y)}x{len(original_x)} -> 目标 {new_y_size}x{new_x_size}")
    
    # 创建新的坐标
    x_min, x_max = original_x.min(), original_x.max()
    y_min, y_max = original_y.min(), original_y.max()
    
    new_x = np.linspace(x_min, x_max, new_x_size)
    new_y = np.linspace(y_min, y_max, new_y_size)
    
    return new_x, new_y

def resample_using_zoom(data_array, x_ratio, y_ratio, order=1):
    """
    使用scipy.ndimage.zoom进行快速重采样
    
    Parameters:
    -----------
    data_array : 3D array (time, y, x)
        输入数据
    x_ratio, y_ratio : float
        重采样比例
    order : int
        插值阶数 (0=最近邻, 1=线性, 3=三次)
    
    Returns:
    --------
    resampled_array : 3D array
        重采样后的数据
    """
    n_time = data_array.shape[0]
    resampled_data = []
    
    for t in range(n_time):
        # 获取当前时间步的数据
        data_t = data_array[t]
        
        # 处理NaN值：zoom不能直接处理NaN
        nan_mask = np.isnan(data_t)
        
        if np.all(nan_mask):
            # 如果全是NaN，创建对应大小的NaN数组
            new_shape = (int(data_t.shape[0] * y_ratio), int(data_t.shape[1] * x_ratio))
            resampled_t = np.full(new_shape, np.nan)
        else:
            # 用最近邻值填充NaN
            data_filled = data_t.copy()
            if np.any(nan_mask):
                # 简单的NaN填充：用有效值的均值
                valid_mean = np.nanmean(data_t)
                data_filled[nan_mask] = valid_mean
            
            # 执行zoom重采样
            resampled_t = zoom(data_filled, (y_ratio, x_ratio), order=order, prefilter=False)
            
            # 如果原始数据有NaN，需要在重采样结果中也标记相应的NaN区域
            if np.any(nan_mask):
                # 重采样NaN掩膜
                nan_mask_resampled = zoom(nan_mask.astype(float), (y_ratio, x_ratio), order=0) > 0.5
                resampled_t[nan_mask_resampled] = np.nan
        
        resampled_data.append(resampled_t)
    
    return np.array(resampled_data)

def resample_coordinates(cell_lat, cell_lon, x_ratio, y_ratio, order=1):
    """
    重采样坐标数组
    
    Parameters:
    -----------
    cell_lat, cell_lon : 2D arrays
        原始坐标
    x_ratio, y_ratio : float
        重采样比例
    order : int
        插值阶数
    
    Returns:
    --------
    new_cell_lat, new_cell_lon : 2D arrays
        重采样后的坐标
    """
    new_cell_lat = zoom(cell_lat, (y_ratio, x_ratio), order=order, prefilter=False)
    new_cell_lon = zoom(cell_lon, (y_ratio, x_ratio), order=order, prefilter=False)
    
    return new_cell_lat, new_cell_lon

def resample_smap_data(input_nc_path, output_nc_path, target_spacing_m=1000, method='linear'):
    """
    重采样单个SMAP NetCDF文件到指定网格间距
    
    Parameters:
    -----------
    input_nc_path : str
        输入NetCDF文件路径
    output_nc_path : str
        输出NetCDF文件路径
    target_spacing_m : int
        目标网格间距，单位为米
    method : str
        插值方法：'linear'(1), 'nearest'(0), 'cubic'(3)
    
    Returns:
    --------
    bool : 处理是否成功
    """
    logger.info(f"开始重采样文件: {os.path.basename(input_nc_path)}")
    
    try:
        # 1. 读取原始数据
        ds = xr.open_dataset(input_nc_path)
        
        # 检查必要的变量是否存在
        if 'sm_surface_wetness' not in ds.data_vars:
            logger.error(f"文件 {os.path.basename(input_nc_path)} 中未找到 sm_surface_wetness 变量")
            return False
            
        if 'cell_lat' not in ds.coords or 'cell_lon' not in ds.coords:
            logger.error(f"文件 {os.path.basename(input_nc_path)} 中未找到坐标变量 cell_lat 和 cell_lon")
            return False
        
        # 2. 获取原始数据
        sm_data = ds['sm_surface_wetness'].values  # shape: (time, y, x)
        cell_lat = ds['cell_lat'].values  # shape: (y, x)
        cell_lon = ds['cell_lon'].values  # shape: (y, x)
        time_coords = ds['time'].values
        x_coords = ds['x'].values
        y_coords = ds['y'].values
        
        # 检查y坐标是否需要翻转（解决图像倒置问题）
        if len(y_coords) > 1 and y_coords[0] > y_coords[-1]:
            logger.info("检测到y坐标递减，翻转数据以确保正确的地理方向")
            sm_data = sm_data[:, ::-1, :]
            cell_lat = cell_lat[::-1, :]
            cell_lon = cell_lon[::-1, :]
            y_coords = y_coords[::-1]
        
        # 3. 计算原始网格的实际物理间距
        x_spacing_m, y_spacing_m = calculate_grid_spacing(x_coords, y_coords, cell_lat, cell_lon)
        
        # 4. 计算重采样比例
        x_ratio = x_spacing_m / target_spacing_m
        y_ratio = y_spacing_m / target_spacing_m
        
        if x_ratio < 1 or y_ratio < 1:
            logger.warning(f"目标分辨率({target_spacing_m}m)高于原始分辨率(x:{x_spacing_m:.0f}m, y:{y_spacing_m:.0f}m)")
            logger.warning("这将进行上采样，可能会产生插值伪影")
        
        # 5. 设置插值阶数
        if method == 'nearest':
            order = 0
        elif method == 'linear':
            order = 1
        elif method == 'cubic':
            order = 3
        else:
            logger.warning(f"未知的插值方法 {method}，使用线性插值")
            order = 1
        
        # 6. 执行重采样
        logger.info("开始重采样数据...")
        resampled_sm = resample_using_zoom(sm_data, x_ratio, y_ratio, order)
        
        logger.info("开始重采样坐标...")
        resampled_cell_lat, resampled_cell_lon = resample_coordinates(
            cell_lat, cell_lon, x_ratio, y_ratio, order
        )
        
        # 7. 根据实际重采样结果创建匹配的x,y坐标
        actual_y_size, actual_x_size = resampled_sm.shape[1], resampled_sm.shape[2]
        logger.info(f"实际重采样后的数据大小: {resampled_sm.shape}")
        
        # 创建与实际数据大小匹配的坐标
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        new_x = np.linspace(x_min, x_max, actual_x_size)
        new_y = np.linspace(y_min, y_max, actual_y_size)
        
        # 验证所有数组的大小匹配
        logger.info(f"数据形状验证:")
        logger.info(f"  sm_surface_wetness: {resampled_sm.shape}")
        logger.info(f"  cell_lat: {resampled_cell_lat.shape}")
        logger.info(f"  cell_lon: {resampled_cell_lon.shape}")
        logger.info(f"  new_x: {len(new_x)}")
        logger.info(f"  new_y: {len(new_y)}")
        
        # 确保所有数组大小匹配
        if resampled_cell_lat.shape != (len(new_y), len(new_x)):
            logger.error(f"坐标数组大小不匹配！重新调整...")
            # 如果大小不匹配，调整坐标数组大小
            if resampled_cell_lat.shape[0] != len(new_y):
                new_y = np.linspace(y_min, y_max, resampled_cell_lat.shape[0])
            if resampled_cell_lat.shape[1] != len(new_x):
                new_x = np.linspace(x_min, x_max, resampled_cell_lat.shape[1])
            logger.info(f"调整后 - new_x: {len(new_x)}, new_y: {len(new_y)}")
        crs_wgs84 = xr.DataArray(
            0,
            attrs={
                'grid_mapping_name': 'latitude_longitude',
                'longitude_of_prime_meridian': 0.0,
                'semi_major_axis': 6378137.0,
                'inverse_flattening': 298.257223563,
                'spatial_ref': 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]',
                'crs_wkt': 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]',
                'epsg_code': 'EPSG:4326'
            }
        )
        
        # 9. 创建输出数据集
        coords = {
            'time': time_coords,
            'y': (['y'], new_y, {
                'long_name': 'y coordinate', 
                'units': 'meters',
                'standard_name': 'projection_y_coordinate'
            }),
            'x': (['x'], new_x, {
                'long_name': 'x coordinate', 
                'units': 'meters',
                'standard_name': 'projection_x_coordinate'
            })
        }
        
        data_vars = {
            'sm_surface_wetness': (['time', 'y', 'x'], resampled_sm, {
                'long_name': 'Surface Soil Moisture (Resampled)',
                'units': 'cm3/cm3',
                'valid_range': [0.0, 1.0],
                'grid_mapping': 'crs',
                'coordinates': 'cell_lon cell_lat',
                'interpolation_method': method,
                'original_spacing_x': f'{x_spacing_m:.0f} m',
                'original_spacing_y': f'{y_spacing_m:.0f} m',
                'target_spacing': f'{target_spacing_m} m'
            }),
            'cell_lat': (['y', 'x'], resampled_cell_lat, {
                'long_name': 'Cell Center Latitude',
                'units': 'degrees_north',
                'standard_name': 'latitude',
                'grid_mapping': 'crs'
            }),
            'cell_lon': (['y', 'x'], resampled_cell_lon, {
                'long_name': 'Cell Center Longitude',
                'units': 'degrees_east',
                'standard_name': 'longitude',
                'grid_mapping': 'crs'
            }),
            'crs': crs_wgs84
        }
        
        # 创建输出数据集
        ds_resampled = xr.Dataset(data_vars, coords=coords)
        
        # 10. 设置全局属性
        original_attrs = dict(ds.attrs) if hasattr(ds, 'attrs') else {}
        ds_resampled.attrs.update({
            'title': 'SMAP Surface Soil Moisture - Resampled',
            'source': 'Resampled from SMAP L4 Global 9 km EASE-Grid Surface and Root-Zone Soil Moisture',
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_file': os.path.basename(input_nc_path),
            'Conventions': 'CF-1.8',
            'coordinate_system': 'WGS84',
            'crs': 'EPSG:4326',
            'geospatial_lat_min': float(np.nanmin(resampled_cell_lat)),
            'geospatial_lat_max': float(np.nanmax(resampled_cell_lat)),
            'geospatial_lon_min': float(np.nanmin(resampled_cell_lon)),
            'geospatial_lon_max': float(np.nanmax(resampled_cell_lon)),
            'spatial_resolution': f'{target_spacing_m} m',
            'grid_mapping_name': 'latitude_longitude',
            'interpolation_method': method,
            'original_spacing_x': f'{x_spacing_m:.0f} m',
            'original_spacing_y': f'{y_spacing_m:.0f} m',
            'resampling_ratio_x': f'{x_ratio:.3f}',
            'resampling_ratio_y': f'{y_ratio:.3f}'
        })
        
        # 保留原始文件的一些关键属性
        for key in ['clip_file', 'original_file']:
            if key in original_attrs:
                ds_resampled.attrs[key] = original_attrs[key]
        
        # 11. 保存文件
        encoding = {
            'sm_surface_wetness': {
                'zlib': True, 
                'complevel': 4, 
                'dtype': 'float32',
                '_FillValue': np.nan
            },
            'cell_lat': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'cell_lon': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'crs': {'dtype': 'int8'},
            'time': {
                'units': 'seconds since 1970-01-01T00:00:00',
                'calendar': 'gregorian',
                'dtype': 'float64'
            },
            'y': {'dtype': 'float64'},
            'x': {'dtype': 'float64'}
        }
        
        os.makedirs(os.path.dirname(output_nc_path), exist_ok=True)
        ds_resampled.to_netcdf(output_nc_path, encoding=encoding)
        
        # 关闭数据集
        ds.close()
        ds_resampled.close()
        
        # 统计信息
        valid_count = np.count_nonzero(~np.isnan(resampled_sm))
        total_count = resampled_sm.size
        logger.info(f"重采样完成: {os.path.basename(input_nc_path)} -> {os.path.basename(output_nc_path)}")
        logger.info(f"数据大小: {sm_data.shape} -> {resampled_sm.shape}")
        logger.info(f"有效数据点: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"重采样文件 {os.path.basename(input_nc_path)} 失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def batch_resample_smap(input_dir, output_dir, target_spacing_m=1000, 
                       method='linear', num_processes=None, file_pattern="*_clipped_*.nc"):
    """
    批量重采样SMAP NetCDF文件
    
    Parameters:
    -----------
    input_dir : str
        输入目录路径
    output_dir : str
        输出目录路径
    target_spacing_m : int
        目标网格间距，单位为米
    method : str
        插值方法：'linear', 'nearest', 'cubic'
    num_processes : int, optional
        并行处理的进程数量。如果为None，则使用CPU核心数-1
    file_pattern : str
        文件匹配模式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有NetCDF文件
    search_pattern = os.path.join(input_dir, file_pattern)
    nc_files = glob.glob(search_pattern)
    
    logger.info(f"找到 {len(nc_files)} 个NetCDF文件")
    
    if len(nc_files) == 0:
        logger.warning(f"在 {input_dir} 中未找到匹配 {file_pattern} 的文件")
        return
    
    # 准备任务列表
    tasks = []
    for nc_file in nc_files:
        base_name = os.path.splitext(os.path.basename(nc_file))[0]
        output_file = f"{base_name}_resampled_{target_spacing_m}m.nc"
        output_path = os.path.join(output_dir, output_file)
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            logger.info(f"输出文件已存在，跳过: {output_file}")
            continue
            
        tasks.append((nc_file, output_path, target_spacing_m, method))
    
    if len(tasks) == 0:
        logger.info("所有文件都已处理完成。")
        return
    
    # 设置进程数量
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"将使用 {num_processes} 个进程进行并行处理")
    
    # 串行处理（推荐，避免内存问题）
    if num_processes == 1:
        logger.info("使用串行处理模式")
        results = []
        for task in tqdm(tasks, desc="重采样文件"):
            result = resample_smap_data(*task)
            results.append(result)
    else:
        # 并行处理
        logger.info(f"使用并行处理模式，进程数: {num_processes}")
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.starmap(resample_smap_data, tasks), 
                              total=len(tasks), desc="重采样文件"))
    
    # 统计结果
    successful_count = sum(1 for r in results if r is True)
    failed_count = len(results) - successful_count
    logger.info(f"批量重采样完成。成功处理 {successful_count} 个文件，失败 {failed_count} 个文件。")

if __name__ == "__main__":
    # 配置路径
    input_folder = 'G:\\CNN_SpatialDownscaling\\SMAP\\SPL4SMGP\\SMAP\\output_clip_nc'  # 原始裁剪后的NC文件目录
    output_folder = 'G:\\CNN_SpatialDownscaling\\SMAP\\SPL4SMGP\\SMAP\\output_resampled_1km'  # 重采样后的输出目录
    
    # 重采样参数
    target_grid_spacing = 1000  # 目标网格间距：1000米
    interpolation_method = 'linear'  # 插值方法：linear, nearest, cubic
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        logger.error(f"错误: 输入文件夹不存在: {input_folder}")
        exit(1)
    
    logger.info("开始批量重采样SMAP数据...")
    logger.info(f"输入目录: {input_folder}")
    logger.info(f"输出目录: {output_folder}")
    logger.info(f"目标网格间距: {target_grid_spacing}m")
    logger.info(f"插值方法: {interpolation_method}")
    
    # 批量处理，建议使用单进程避免内存问题
    batch_resample_smap(
        input_folder, 
        output_folder, 
        target_spacing_m=target_grid_spacing,
        method=interpolation_method,
        num_processes=1  # 建议使用单进程，因为重采样比较占内存
    )

    