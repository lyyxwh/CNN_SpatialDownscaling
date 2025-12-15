import h5py
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm
import multiprocessing
import logging
import re

# 配置日志，避免并行处理时 tqdm 和 print 混乱
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_time_from_smap(h5_file_path):
    """
    从SMAP H5文件中提取时间信息并格式化为文件名用的时间字符串
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'time' in f.keys():
                time_data = f['time'][:]
                time_attrs = dict(f['time'].attrs)               
                units = time_attrs.get('units', 'seconds since 2000-01-01 00:00:00')
                if isinstance(units, bytes):
                    units = units.decode('utf-8')
                elif isinstance(units, np.ndarray):
                    units = str(units.item())                

                if 'since' in units:
                    time_unit, reference_time = units.split(' since ')
                    reference_dt = pd.to_datetime(reference_time)
                    if 'second' in time_unit:
                        time_dt = reference_dt + pd.to_timedelta(time_data[0], unit='s')
                    elif 'day' in time_unit:
                        time_dt = reference_dt + pd.to_timedelta(time_data[0], unit='D')
                    elif 'hour' in time_unit:
                        time_dt = reference_dt + pd.to_timedelta(time_data[0], unit='h')
                    else:
                        time_dt = reference_dt + pd.to_timedelta(time_data[0], unit='s')
                else:
                    time_dt = pd.to_datetime(time_data[0])
            else:
                filename = os.path.basename(h5_file_path)                
                date_match = re.search(r'(\d{8})T(\d{6})', filename)
                if date_match:
                    date_str = date_match.group(1)
                    time_str = date_match.group(2)
                    time_dt = pd.to_datetime(f"{date_str}_{time_str}", format='%Y%m%d_%H%M%S')
                else:
                    time_dt = pd.Timestamp.now()

    except Exception as e:
        logger.warning(f"无法从 {h5_file_path} 中提取时间，尝试从文件名解析。错误: {e}")
        filename = os.path.basename(h5_file_path)

        try:           
            date_match = re.search(r'(\d{8})T(\d{6})', filename)
            if date_match:
                date_str = date_match.group(1)
                time_str = date_match.group(2)
                time_dt = pd.to_datetime(f"{date_str}_{time_str}", format='%Y%m%d_%H%M%S')
            else:
                time_dt = pd.Timestamp.now()

        except Exception as file_e:
            logger.error(f"无法从文件名 {filename} 中解析时间。错误: {file_e}")
            time_dt = pd.Timestamp.now()

    return time_dt.strftime('%Y%m%d%H')


def process_smap_data(h5_file_path, clip_gdf_preloaded, original_shapefile_path, output_nc_path=None):
    """
    处理SMAP H5数据，按预加载的shapefile裁剪并转换为NetCDF格式。
    确保输出文件的坐标系为WGS84。
    此函数将在单独的进程中运行。
    """
    logger.info(f"开始处理文件: {os.path.basename(h5_file_path)}")

    if output_nc_path is None:
        time_str = extract_time_from_smap(h5_file_path)
        base_name = os.path.splitext(os.path.basename(h5_file_path))[0]
        output_nc_path = os.path.join(os.path.dirname(h5_file_path), f"{base_name}_clipped_{time_str}.nc")
    
    # 1. 读取H5数据
    try:
        with h5py.File(h5_file_path, 'r') as f:
            sm_data = f['Geophysical_Data/sm_surface_wetness'][:]
            cell_lat = f['cell_lat'][:]
            cell_lon = f['cell_lon'][:]
            x = f['x'][:]
            y = f['y'][:]
            
            if 'time' in f.keys():
                time_data = f['time'][:]
                units = f['time'].attrs.get('units', b'seconds since 2000-01-01 00:00:00')
                if isinstance(units, bytes):
                    units = units.decode('utf-8')
            else:
                time_data = np.array([0])
                units = 'seconds since 2000-01-01 00:00:00'
                
            # 获取填充值
            fill_value = f['Geophysical_Data/sm_surface_wetness'].attrs.get('_FillValue', -9999)
            if isinstance(fill_value, bytes):
                fill_value = float(fill_value.decode('utf-8'))
            elif isinstance(fill_value, np.ndarray):
                fill_value = float(fill_value.item())

    except Exception as e:
        logger.error(f"读取H5文件 {os.path.basename(h5_file_path)} 失败: {e}")
        return False

    # 处理数据维度
    if len(sm_data.shape) == 3:  # (time, y, x)
        n_time, n_y, n_x = sm_data.shape
    elif len(sm_data.shape) == 2:  # (y, x)
        n_y, n_x = sm_data.shape
        n_time = 1
        sm_data = sm_data[np.newaxis, :, :]
    else:
        logger.error(f"不支持的数据形状: {sm_data.shape} for {os.path.basename(h5_file_path)}")
        return False

    if cell_lat.shape != (n_y, n_x) or cell_lon.shape != (n_y, n_x):
        logger.error(f"坐标数组形状与数据不匹配 for {os.path.basename(h5_file_path)}")
        return False

    # 2. 使用预加载的裁剪区域
    clip_gdf = clip_gdf_preloaded
    total_bounds = clip_gdf.total_bounds

    # 3. 首先处理填充值，将填充值设为NaN
    sm_data_clean = sm_data.copy()
    sm_data_clean[sm_data_clean == fill_value] = np.nan

    # 4. 创建有效数据掩膜
    valid_coord_mask = np.logical_and(
        np.logical_and(np.isfinite(cell_lat), np.isfinite(cell_lon)),
        np.logical_and(cell_lat >= -90, cell_lat <= 90)  # 确保纬度合理
    )
    
    # 边界框预筛选
    bounds_mask = np.logical_and.reduce([
        cell_lon >= total_bounds[0],
        cell_lon <= total_bounds[2],
        cell_lat >= total_bounds[1],
        cell_lat <= total_bounds[3]
    ])
    
    candidate_mask = np.logical_and(valid_coord_mask, bounds_mask)

    if np.sum(candidate_mask) == 0:
        logger.warning(f"文件 {os.path.basename(h5_file_path)} 在边界内没有候选点。跳过。")
        return False

    # 5. 精确的点在多边形内检查
    candidate_rows, candidate_cols = np.where(candidate_mask)    
    points_df = pd.DataFrame({
        'original_row': candidate_rows,
        'original_col': candidate_cols,
        'longitude': cell_lon[candidate_rows, candidate_cols],
        'latitude': cell_lat[candidate_rows, candidate_cols]
    })

    # 确保坐标系为WGS84
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude),
        crs='EPSG:4326'
    )
    
    # 确保裁剪区域也是WGS84
    if clip_gdf.crs != 'EPSG:4326':
        clip_gdf = clip_gdf.to_crs('EPSG:4326')
    
    points_in_clip = gpd.sjoin(points_gdf, clip_gdf, how='inner', predicate='within')
    
    if len(points_in_clip) == 0:
        logger.warning(f"文件 {os.path.basename(h5_file_path)} 没有找到与裁剪区域相交的数据点。跳过。")
        return False
   
    # 6. 创建最终掩膜
    final_mask = np.zeros_like(candidate_mask, dtype=bool)
    final_mask[points_in_clip['original_row'].values, points_in_clip['original_col'].values] = True

    # 7. 提取裁剪后的数据
    valid_rows, valid_cols = np.where(final_mask)
    if valid_rows.size == 0 or valid_cols.size == 0:
        logger.warning(f"文件 {os.path.basename(h5_file_path)} 裁剪后没有有效数据点。跳过。")
        return False
        
    min_row, max_row = valid_rows.min(), valid_rows.max()
    min_col, max_col = valid_cols.min(), valid_cols.max()
    
    # 裁剪数据和坐标
    clipped_mask = final_mask[min_row:max_row+1, min_col:max_col+1]
    clipped_sm = sm_data_clean[:, min_row:max_row+1, min_col:max_col+1].copy()
    clipped_lat = cell_lat[min_row:max_row+1, min_col:max_col+1]
    clipped_lon = cell_lon[min_row:max_row+1, min_col:max_col+1] 
    clipped_x = x[min_col:max_col+1]
    clipped_y = y[min_row:max_row+1]
    
    # 8. 应用掩膜：将不在裁剪区域内的点设为NaN
    for t in range(n_time):
        clipped_sm[t][~clipped_mask] = np.nan
    
    # 9. 创建时间坐标
    try:
        if 'since' in units:
            time_unit, reference_time = units.split(' since ')
            reference_dt = pd.to_datetime(reference_time)
            if 'second' in time_unit:
                time_coords = reference_dt + pd.to_timedelta(time_data, unit='s')
            elif 'day' in time_unit:
                time_coords = reference_dt + pd.to_timedelta(time_data, unit='D')
            elif 'hour' in time_unit:
                time_coords = reference_dt + pd.to_timedelta(time_data, unit='h')
            elif 'minute' in time_unit:
                time_coords = reference_dt + pd.to_timedelta(time_data, unit='m')
            else:
                time_coords = reference_dt + pd.to_timedelta(time_data, unit='s')
        else:
            time_coords = pd.to_datetime(time_data)
    except Exception as e:
        logger.warning(f"文件 {os.path.basename(h5_file_path)} 时间解析失败: {e}，尝试从文件名提取。")
        filename = os.path.basename(h5_file_path)
        date_match = re.search(r'(\d{8})T(\d{6})', filename)
        if date_match:
            date_str = date_match.group(1)
            time_str = date_match.group(2)
            file_time = pd.to_datetime(f"{date_str}_{time_str}", format='%Y%m%d_%H%M%S')
            time_coords = pd.DatetimeIndex([file_time])
        else:
            time_coords = pd.date_range('2000-01-01', periods=n_time, freq='D')
   
    if len(time_coords) != n_time:
        if len(time_coords) == 1:
            time_coords = pd.DatetimeIndex([time_coords[0]] * n_time)
        else:
            time_coords = time_coords[:n_time]   

    # 10. 创建WGS84坐标参考系统
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

    # 11. 创建xarray Dataset
    coords = {
        'time': time_coords[:n_time], 
        'y': (['y'], clipped_y, {'long_name': 'y coordinate', 'units': 'meters'}), 
        'x': (['x'], clipped_x, {'long_name': 'x coordinate', 'units': 'meters'})
    }
    
    data_vars = {
        'sm_surface_wetness': (['time', 'y', 'x'], clipped_sm, {
            'long_name': 'Surface Soil Moisture',
            'units': 'cm3/cm3',
            'valid_range': [0.0, 1.0],
            'grid_mapping': 'crs',
            'coordinates': 'cell_lon cell_lat'
        }),
        'cell_lat': (['y', 'x'], clipped_lat, {
            'long_name': 'Cell Center Latitude',
            'units': 'degrees_north',
            'standard_name': 'latitude',
            'grid_mapping': 'crs'
        }),
        'cell_lon': (['y', 'x'], clipped_lon, {
            'long_name': 'Cell Center Longitude',
            'units': 'degrees_east',
            'standard_name': 'longitude',
            'grid_mapping': 'crs'
        }),
        'crs': crs_wgs84
    }
    
    ds = xr.Dataset(data_vars, coords=coords)   

    # 12. 设置全局属性，强制指定WGS84坐标系
    ds.attrs.update({
        'title': 'SMAP Surface Soil Moisture - Clipped to Study Area',
        'source': 'SMAP L3 Radiometer Global Daily 36 km EASE-Grid Soil Moisture',
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'clip_file': os.path.basename(original_shapefile_path),
        'original_file': os.path.basename(h5_file_path),
        'Conventions': 'CF-1.8',
        'coordinate_system': 'WGS84',
        'crs': 'EPSG:4326',
        'geospatial_lat_min': float(np.nanmin(clipped_lat)),
        'geospatial_lat_max': float(np.nanmax(clipped_lat)),
        'geospatial_lon_min': float(np.nanmin(clipped_lon)),
        'geospatial_lon_max': float(np.nanmax(clipped_lon)),
        'spatial_resolution': '9 km',
        'grid_mapping_name': 'latitude_longitude'
    })

    # 13. 保存为NetCDF
    try:
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
            }
        }

        os.makedirs(os.path.dirname(output_nc_path), exist_ok=True)
        ds.to_netcdf(output_nc_path, encoding=encoding)

        logger.info(f"成功处理并保存: {os.path.basename(h5_file_path)} -> {os.path.basename(output_nc_path)}")
        
        # 记录一些统计信息
        valid_data_count = np.count_nonzero(~np.isnan(clipped_sm))
        total_data_count = clipped_sm.size
        logger.info(f"有效数据点: {valid_data_count}/{total_data_count} ({100*valid_data_count/total_data_count:.1f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"保存NetCDF文件 {os.path.basename(output_nc_path)} 失败: {e}")
        return False


def batch_process_smap(input_dir, shapefile_path, output_dir, num_processes=None):
    """
    批量处理多个SMAP H5文件，支持并行处理。    
    Parameters:
    -----------
    input_dir : str
        包含H5文件的输入目录
    shapefile_path : str
        用于裁剪的shapefile路径
    output_dir : str
        输出目录
    num_processes : int, optional
        并行处理的进程数量。如果为None，则使用CPU核心数-1。
    """    
    os.makedirs(output_dir, exist_ok=True)
    
    # 预加载并处理裁剪Shapefile（只执行一次）
    logger.info("预加载并处理裁剪Shapefile...")
    clip_gdf = gpd.read_file(shapefile_path)
    if clip_gdf.crs is None:
        logger.warning("裁剪Shapefile没有指定CRS，假定为EPSG:4326。")
        clip_gdf = clip_gdf.set_crs('EPSG:4326')
    elif clip_gdf.crs != 'EPSG:4326':
        logger.info(f"将裁剪Shapefile从 {clip_gdf.crs} 转换为 EPSG:4326...")
        clip_gdf = clip_gdf.to_crs('EPSG:4326')
    logger.info(f"裁剪Shapefile的CRS: {clip_gdf.crs}")    

    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    logger.info(f"找到 {len(h5_files)} 个H5文件")

    if len(h5_files) == 0:
        logger.warning("未找到任何H5文件，请检查输入目录。")
        return

    # 准备并行处理的任务列表
    tasks = []
    for h5_file in h5_files:
        input_path = os.path.join(input_dir, h5_file)
        time_str = extract_time_from_smap(input_path)
        base_name = os.path.splitext(h5_file)[0]
        output_file = f"{base_name}_clipped_{time_str}.nc"
        output_path = os.path.join(output_dir, output_file)
        
        # 检查输出文件是否已存在，可选择跳过
        if os.path.exists(output_path):
            logger.info(f"输出文件已存在，跳过: {output_file}")
            continue
            
        tasks.append((input_path, clip_gdf, shapefile_path, output_path))
   
    if len(tasks) == 0:
        logger.info("所有文件都已处理完成。")
        return
   
    # 设置进程数量
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"将使用 {num_processes} 个进程进行并行处理。")
    
    # 使用进程池进行并行处理
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.starmap(process_smap_data, tasks), total=len(tasks), desc="处理文件"))
    
    successful_count = sum(1 for r in results if r is True)
    failed_count = len(results) - successful_count
    logger.info(f"批量处理完成。成功处理 {successful_count} 个文件，失败 {failed_count} 个文件。")


if __name__ == "__main__":
    # 在Windows上，为了使用 multiprocessing，需要将主代码放在 if __name__ == '__main__': 块中
    # 文件路径配置
    input_folder = 'G:\\CNN_SpatialDownscaling\\SMAP\\SPL4SMGP\\SMAP'
    output_folder = 'G:\\CNN_SpatialDownscaling\\SMAP\\SPL4SMGP\\SMAP\\output_clip_nc'
    clip_shapefile = 'G:\\Agr-qu\\clip.shp'
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        logger.error(f"错误: 输入文件夹不存在: {input_folder}")
        exit(1)    
    
    # 检查shapefile是否存在
    if not os.path.exists(clip_shapefile):
        logger.error(f"错误: Shapefile不存在: {clip_shapefile}")
        exit(1)
    
    # 批量处理所有H5文件，可以指定进程数，例如 num_processes=4
    logger.info("开始批量处理SMAP数据...")
    batch_process_smap(input_folder, clip_shapefile, output_folder, num_processes=None)  # 使用默认进程数