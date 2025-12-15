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
import scipy
from scipy.interpolate import griddata
from scipy.ndimage import zoom
import h5py
from scipy.spatial import cKDTree
import logging


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resample_to_target_resolution_wgs84_improved(sm_data, cell_lat, cell_lon, target_resolution_deg=0.00833333333, method='linear'):
    """
    改进的重采样函数：既保持插值效果，又避免在无效区域产生数据
    
    策略：
    1. 正常进行插值以保持数据连续性
    2. 使用原始数据的有效区域掩膜来限制插值结果的范围
    3. 适度扩展有效区域以允许合理的插值
    """
    logger.info(f"开始改进重采样，原始数据形状: {sm_data.shape}")
    
    # 获取有效坐标掩膜
    valid_coord_mask = np.logical_and(
        np.logical_and(np.isfinite(cell_lat), np.isfinite(cell_lon)),
        np.logical_and(cell_lat >= -90, cell_lat <= 90)
    )
    
    if not np.any(valid_coord_mask):
        logger.error("没有有效的坐标点")
        return None, None, None
    
    # 获取有效数据的边界
    valid_lats = cell_lat[valid_coord_mask]
    valid_lons = cell_lon[valid_coord_mask]
    lat_min, lat_max = np.nanmin(valid_lats), np.nanmax(valid_lats)
    lon_min, lon_max = np.nanmin(valid_lons), np.nanmax(valid_lons)
    
    logger.info(f"数据范围: 纬度[{lat_min:.4f}, {lat_max:.4f}], 经度[{lon_min:.4f}, {lon_max:.4f}]")
    
    # 统一分辨率为0.0083333333°，并严格对齐行列数
    target_resolution_deg = 0.00833333333
    # 计算行列数
    n_rows = int(np.round((lat_max - lat_min) / target_resolution_deg))
    n_cols = int(np.round((lon_max - lon_min) / target_resolution_deg))
    lat_max_aligned = lat_min + n_rows * target_resolution_deg
    lon_max_aligned = lon_min + n_cols * target_resolution_deg

    new_lat_1d = np.linspace(lat_min + target_resolution_deg/2, lat_max_aligned - target_resolution_deg/2, n_rows)
    new_lon_1d = np.linspace(lon_min + target_resolution_deg/2, lon_max_aligned - target_resolution_deg/2, n_cols)
    new_lon_grid, new_lat_grid = np.meshgrid(new_lon_1d, new_lat_1d)

    logger.info(f"新网格大小: {new_lat_grid.shape} (分辨率: {target_resolution_deg:.10f}度)")

    # 准备输出数组
    n_time = sm_data.shape[0]
    resampled_sm = np.full((n_time, new_lat_grid.shape[0], new_lat_grid.shape[1]), np.nan, dtype=np.float32)
    
    # 创建总体有效性掩膜（在所有时间步中至少有一次有效数据的区域）
    overall_valid_mask = np.logical_and(
        valid_coord_mask,
        np.any(np.isfinite(sm_data), axis=0)
    )
    
    if not np.any(overall_valid_mask):
        logger.error("没有有效的数据点")
        return None, None, None
    
    # 创建参考有效性掩膜用于后续过滤
    # 方法：将原始有效区域投影到新网格上
    logger.info("创建参考有效性掩膜...")
    
    # 获取原始有效区域的坐标
    valid_original_coords = np.column_stack([
        cell_lon[overall_valid_mask].astype(np.float32),
        cell_lat[overall_valid_mask].astype(np.float32)
    ])
    
    # 目标网格点坐标
    target_coords = np.column_stack([
        new_lon_grid.flatten(),
        new_lat_grid.flatten()
    ])
    
    # 使用KDTree找到每个目标点到最近有效原始数据点的距离
    tree = cKDTree(valid_original_coords)
    distances, _ = tree.query(target_coords)
    
    # 设置合理的最大距离阈值
    # 使用原始分辨率的倍数作为阈值（SMAP原始分辨率约36km = 0.324度）
    original_resolution_deg = 0.324  # SMAP 36km的近似度数
    max_interpolation_distance = original_resolution_deg * 0.8  # 允许一定的插值扩展
    
    # 创建距离掩膜
    distance_mask = distances <= max_interpolation_distance
    validity_mask_1d = distance_mask
    validity_mask_2d = validity_mask_1d.reshape(new_lat_grid.shape)
    
    logger.info(f"有效插值区域比例: {np.sum(validity_mask_2d)}/{validity_mask_2d.size} ({100*np.sum(validity_mask_2d)/validity_mask_2d.size:.1f}%)")
    
    # 对每个时间步进行插值
    for t in range(n_time):
        logger.info(f"处理时间步 {t+1}/{n_time}")
        sm_t = sm_data[t]
        
        # 获取当前时间步的有效数据
        time_valid_mask = np.logical_and(overall_valid_mask, np.isfinite(sm_t))
        
        if np.sum(time_valid_mask) < 3:
            logger.warning(f"时间步 {t} 有效数据点不足({np.sum(time_valid_mask)}个)，跳过插值")
            continue
        
        # 提取有效的原始数据
        original_values = sm_t[time_valid_mask].astype(np.float32)
        original_coords = np.column_stack([
            cell_lon[time_valid_mask].astype(np.float32),
            cell_lat[time_valid_mask].astype(np.float32)
        ])
        
        logger.info(f"时间步 {t}: 使用 {len(original_values)} 个有效数据点进行插值")
        
        try:
            # 进行正常插值（不限制距离，让插值算法发挥作用）
            interpolated_flat = griddata(
                original_coords,
                original_values,
                target_coords,
                method=method,
                fill_value=np.nan
            )
            
            # 将插值结果重新整形
            interpolated_2d = interpolated_flat.reshape(new_lat_grid.shape)
            
            # 应用有效性掩膜：只保留合理距离内的插值结果
            interpolated_2d[~validity_mask_2d] = np.nan
            
            # 存储结果
            resampled_sm[t] = interpolated_2d
            
            # 统计当前时间步的结果
            valid_count = np.count_nonzero(~np.isnan(interpolated_2d))
            total_count = interpolated_2d.size
            logger.info(f"时间步 {t}: 插值后有效数据点 {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
            
        except Exception as e:
            logger.warning(f"时间步 {t} {method}插值失败: {e}")
            # 如果linear插值失败，尝试nearest
            if method != 'nearest':
                try:
                    interpolated_flat = griddata(
                        original_coords,
                        original_values,
                        target_coords,
                        method='nearest',
                        fill_value=np.nan
                    )
                    interpolated_2d = interpolated_flat.reshape(new_lat_grid.shape)
                    interpolated_2d[~validity_mask_2d] = np.nan
                    resampled_sm[t] = interpolated_2d
                    logger.info(f"时间步 {t} 使用nearest插值成功")
                except Exception as e2:
                    logger.error(f"时间步 {t} nearest插值也失败: {e2}")
    
    # 最终统计
    total_points = resampled_sm.size
    valid_points = np.count_nonzero(~np.isnan(resampled_sm))
    logger.info(f"重采样完成，新数据形状: {resampled_sm.shape}")
    logger.info(f"最终有效数据点比例: {valid_points}/{total_points} ({100*valid_points/total_points:.1f}%)")
    
    return resampled_sm, new_lat_grid, new_lon_grid


def resample_with_adaptive_masking(sm_data, cell_lat, cell_lon, target_resolution_deg=0.009, method='linear'):
    """
    自适应掩膜方法：根据数据密度调整有效区域
    """
    logger.info(f"开始自适应掩膜重采样，原始数据形状: {sm_data.shape}")
    
    # 基本有效性检查
    valid_coord_mask = np.logical_and(
        np.logical_and(np.isfinite(cell_lat), np.isfinite(cell_lon)),
        np.logical_and(cell_lat >= -90, cell_lat <= 90)
    )
    
    overall_valid_mask = np.logical_and(
        valid_coord_mask,
        np.any(np.isfinite(sm_data), axis=0)
    )
    
    if not np.any(overall_valid_mask):
        logger.error("没有有效的数据点")
        return None, None, None
    
    # 获取数据范围
    valid_lats = cell_lat[overall_valid_mask]
    valid_lons = cell_lon[overall_valid_mask]
    lat_min, lat_max = np.nanmin(valid_lats), np.nanmax(valid_lats)
    lon_min, lon_max = np.nanmin(valid_lons), np.nanmax(valid_lons)
    
    # 创建目标网格
    new_lat_1d = np.arange(lat_min, lat_max + target_resolution_deg, target_resolution_deg)
    new_lon_1d = np.arange(lon_min, lon_max + target_resolution_deg, target_resolution_deg)
    new_lon_grid, new_lat_grid = np.meshgrid(new_lon_1d, new_lat_1d)
    
    logger.info(f"新网格大小: {new_lat_grid.shape}")
    
    # 计算数据密度和合适的插值距离
    valid_coords = np.column_stack([
        cell_lon[overall_valid_mask],
        cell_lat[overall_valid_mask]
    ])
    
    # 计算最近邻距离的中位数作为数据密度的指标
    tree = cKDTree(valid_coords)
    distances_to_second_nearest, _ = tree.query(valid_coords, k=2)  # k=2因为第一个是自己
    median_spacing = np.median(distances_to_second_nearest[:, 1])  # 使用第二个最近点的距离
    
    # 设置自适应的最大插值距离
    adaptive_max_distance = max(median_spacing * 2.0, target_resolution_deg * 2.0)
    logger.info(f"数据中位间距: {median_spacing:.6f}度, 自适应最大插值距离: {adaptive_max_distance:.6f}度")
    
    # 准备输出
    n_time = sm_data.shape[0]
    resampled_sm = np.full((n_time, new_lat_grid.shape[0], new_lat_grid.shape[1]), np.nan, dtype=np.float32)
    
    # 目标点坐标
    target_coords = np.column_stack([
        new_lon_grid.flatten(),
        new_lat_grid.flatten()
    ])
    
    # 计算到最近有效数据点的距离
    target_distances, _ = tree.query(target_coords)
    target_mask = target_distances <= adaptive_max_distance
    target_mask_2d = target_mask.reshape(new_lat_grid.shape)
    
    logger.info(f"自适应有效区域比例: {np.sum(target_mask_2d)}/{target_mask_2d.size} ({100*np.sum(target_mask_2d)/target_mask_2d.size:.1f}%)")
    
    # 对每个时间步进行插值
    for t in range(n_time):
        logger.info(f"处理时间步 {t+1}/{n_time}")
        sm_t = sm_data[t]
        
        time_valid_mask = np.logical_and(overall_valid_mask, np.isfinite(sm_t))
        
        if np.sum(time_valid_mask) < 3:
            logger.warning(f"时间步 {t} 有效数据点不足，跳过")
            continue
        
        original_coords = np.column_stack([
            cell_lon[time_valid_mask].astype(np.float32),
            cell_lat[time_valid_mask].astype(np.float32)
        ])
        original_values = sm_t[time_valid_mask].astype(np.float32)
        
        try:
            # 正常插值
            interpolated_flat = griddata(
                original_coords,
                original_values,
                target_coords,
                method=method,
                fill_value=np.nan
            )
            
            interpolated_2d = interpolated_flat.reshape(new_lat_grid.shape)
            
            # 应用自适应掩膜
            interpolated_2d[~target_mask_2d] = np.nan
            
            resampled_sm[t] = interpolated_2d
            
        except Exception as e:
            logger.warning(f"时间步 {t} 插值失败: {e}")
            if method != 'nearest':
                try:
                    interpolated_flat = griddata(
                        original_coords,
                        original_values,
                        target_coords,
                        method='nearest',
                        fill_value=np.nan
                    )
                    interpolated_2d = interpolated_flat.reshape(new_lat_grid.shape)
                    interpolated_2d[~target_mask_2d] = np.nan
                    resampled_sm[t] = interpolated_2d
                    logger.info(f"时间步 {t} 使用nearest插值成功")
                except Exception as e2:
                    logger.error(f"时间步 {t} nearest插值也失败: {e2}")
    
    return resampled_sm, new_lat_grid, new_lon_grid

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


def process_smap_data(h5_file_path, clip_gdf_preloaded, original_shapefile_path, output_nc_path=None, target_resolution=1000):
    """
    处理SMAP H5数据：先裁剪，再重采样到指定分辨率，最后转换为NetCDF格式（WGS84坐标系）
    
    Parameters:
    -----------
    target_resolution : int
        目标分辨率（米），默认1000m
    """
    logger.info(f"开始处理文件: {os.path.basename(h5_file_path)}")

    if output_nc_path is None:
        time_str = extract_time_from_smap(h5_file_path)
        base_name = os.path.splitext(os.path.basename(h5_file_path))[0]
        output_nc_path = os.path.join(os.path.dirname(h5_file_path), f"{base_name}_clipped_re_0p0625_{time_str}.nc")
    
    # 1. 读取H5数据
    logger.info(f"正在读取H5文件: {os.path.basename(h5_file_path)}")
    try:
        with h5py.File(h5_file_path, 'r') as f:
            sm_data = f['Geophysical_Data/sm_surface_wetness'][:]
            cell_lat = f['cell_lat'][:]
            cell_lon = f['cell_lon'][:]
            
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

    # 清理数据
    sm_data_clean = sm_data.copy()
    sm_data_clean[sm_data_clean == fill_value] = np.nan

    # 2. 先进行空间裁剪
    logger.info("正在进行空间裁剪...")
    clip_gdf = clip_gdf_preloaded
    total_bounds = clip_gdf.total_bounds

    # 创建有效数据掩膜
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

    # 精确的点在多边形内检查
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
   
    # 创建最终掩膜
    final_mask = np.zeros_like(candidate_mask, dtype=bool)
    final_mask[points_in_clip['original_row'].values, points_in_clip['original_col'].values] = True

    # 提取裁剪后的数据
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
    
    # 应用掩膜：将不在裁剪区域内的点设为NaN
    for t in range(n_time):
        clipped_sm[t][~clipped_mask] = np.nan
    
    logger.info(f"裁剪完成，裁剪后数据形状: {clipped_sm.shape}")

    # 3. 对裁剪后的数据进行重采样
    logger.info("正在进行重采样操作...")
    
    # 计算目标分辨率（度）
    target_resolution_deg = target_resolution / 111000.0  # 1度约111km
    
    try:
        resampled_sm, new_lat_grid, new_lon_grid = resample_to_target_resolution_wgs84_improved(
            clipped_sm, clipped_lat, clipped_lon, 
            target_resolution_deg=target_resolution_deg, 
            method='linear'
        )
        
        if resampled_sm is None:
            logger.error("重采样失败")
            return False
            
        logger.info(f"重采样成功，新数据形状: {resampled_sm.shape}")
    except Exception as e:
        logger.error(f"重采样失败: {e}")
        return False

    # 4. 创建时间坐标
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

    # 5. 创建WGS84坐标参考系统
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

    # 6. 创建xarray Dataset（WGS84坐标系）
    coords = {
        'time': time_coords[:n_time],
        'latitude': (['latitude'], new_lat_grid[:, 0], {
            'long_name': 'latitude', 
            'units': 'degrees_north', 
            'standard_name': 'latitude'
        }),
        'longitude': (['longitude'], new_lon_grid[0, :], {
            'long_name': 'longitude', 
            'units': 'degrees_east', 
            'standard_name': 'longitude'
        })
    }
    
    data_vars = {
        'sm_surface_wetness': (['time', 'latitude', 'longitude'], resampled_sm, {
            'long_name': f'Surface Soil Moisture (Clipped and Resampled to {target_resolution}m)',
            'units': 'cm3/cm3',
            'valid_range': [0.0, 1.0],
            'grid_mapping': 'crs',
            'coordinates': 'longitude latitude',
            'resampling_method': 'linear interpolation',
            'original_resolution': '36 km',
            'target_resolution': f'{target_resolution} m',
            'processing_steps': 'clipped_then_resampled'
        }),
        'crs': crs_wgs84
    }
    
    ds = xr.Dataset(data_vars, coords=coords)   

    # 7. 设置全局属性（WGS84坐标系）
    ds.attrs.update({
        'title': f'SMAP Surface Soil Moisture - Clipped and Resampled to {target_resolution}m',
        'source': 'SMAP L3 Radiometer Global Daily 36 km EASE-Grid Soil Moisture',
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'clip_file': os.path.basename(original_shapefile_path),
        'original_file': os.path.basename(h5_file_path),
        'Conventions': 'CF-1.8',
        'coordinate_system': 'WGS84',
        'crs': 'EPSG:4326',
        'geospatial_lat_min': float(np.nanmin(new_lat_grid)),
        'geospatial_lat_max': float(np.nanmax(new_lat_grid)),
        'geospatial_lon_min': float(np.nanmin(new_lon_grid)),
        'geospatial_lon_max': float(np.nanmax(new_lon_grid)),
        'original_spatial_resolution': '36 km',
        'target_spatial_resolution': f'{target_resolution} m',
        'resampling_method': 'linear interpolation',
        'processing_workflow': 'clip_then_resample',
        'grid_mapping_name': 'latitude_longitude'
    })

    # 8. 保存为NetCDF
    try:
        encoding = {
            'sm_surface_wetness': {
                'zlib': True, 
                'complevel': 4, 
                'dtype': 'float32',
                '_FillValue': np.nan
            },
            'crs': {'dtype': 'int8'},
            'time': {
                'units': 'seconds since 1970-01-01T00:00:00',
                'calendar': 'gregorian',
                'dtype': 'float32'
            },
            'latitude': {'dtype': 'float32'},
            'longitude': {'dtype': 'float32'}
        }

        os.makedirs(os.path.dirname(output_nc_path), exist_ok=True)
        ds.to_netcdf(output_nc_path, encoding=encoding)

        logger.info(f"成功处理并保存: {os.path.basename(h5_file_path)} -> {os.path.basename(output_nc_path)}")
        
        # 记录一些统计信息
        valid_data_count = np.count_nonzero(~np.isnan(resampled_sm))
        total_data_count = resampled_sm.size
        logger.info(f"最终数据形状: {resampled_sm.shape}")
        logger.info(f"有效数据点: {valid_data_count}/{total_data_count} ({100*valid_data_count/total_data_count:.1f}%)")
        logger.info(f"坐标范围: 纬度[{np.nanmin(new_lat_grid):.4f}, {np.nanmax(new_lat_grid):.4f}], 经度[{np.nanmin(new_lon_grid):.4f}, {np.nanmax(new_lon_grid):.4f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"保存NetCDF文件 {os.path.basename(output_nc_path)} 失败: {e}")
        return False


def batch_process_smap(input_dir, shapefile_path, output_dir, target_resolution=1000, num_processes=None):
    """
    批量处理多个SMAP H5文件：先裁剪再重采样，支持并行处理
    
    Parameters:
    -----------
    input_dir : str
        包含H5文件的输入目录
    shapefile_path : str
        用于裁剪的shapefile路径
    output_dir : str
        输出目录
    target_resolution : int
        目标分辨率（米），默认1000m
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
        output_file = f"{base_name}_clipped_resampled_{target_resolution}m_{time_str}.nc"
        output_path = os.path.join(output_dir, output_file)
        
        # 检查输出文件是否已存在，可选择跳过
        if os.path.exists(output_path):
            logger.info(f"输出文件已存在，跳过: {output_file}")
            continue
            
        tasks.append((input_path, clip_gdf, shapefile_path, output_path, target_resolution))
   
    if len(tasks) == 0:
        logger.info("所有文件都已处理完成。")
        return
   
    # 设置进程数量
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() )
    logger.info(f"将使用 {num_processes} 个进程进行并行处理，目标分辨率: {target_resolution}m。")
    logger.info("处理流程: 先裁剪 -> 再重采样 -> 输出WGS84坐标系NC文件")
    
    logger.info(f"开始并行处理 {len(tasks)} 个文件，每批次进程数: {num_processes}")
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = []
            for i, r in enumerate(tqdm(pool.starmap(process_smap_data, tasks), total=len(tasks), desc="处理文件")):
                results.append(r)
                if (i+1) % 10 == 0 or (i+1) == len(tasks):
                    logger.info(f"已完成 {i+1}/{len(tasks)} 文件...")
    except Exception as e:
        logger.error(f"多进程批量处理异常: {e}")
        return

    successful_count = sum(1 for r in results if r is True)
    failed_count = len(results) - successful_count
    logger.info(f"批量处理完成。成功处理 {successful_count} 个文件，失败 {failed_count} 个文件。")


if __name__ == "__main__":
    # 在Windows上，为了使用 multiprocessing，需要将主代码放在 if __name__ == '__main__': 块中
    # 文件路径配置
    input_folder = 'G:\\CNN_SpatialDownscaling\\201712and201901\\SMAP201712'
    output_folder = 'G:\\CNN_SpatialDownscaling\\201712and201901\\SMAP201712\\output_clip_re_1000_nc'
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
    logger.info("开始批量处理SMAP数据，流程: 先裁剪再重采样到1km分辨率，输出WGS84坐标系...")
    batch_process_smap(input_folder, clip_shapefile, output_folder, target_resolution=1000, num_processes=1)  # 使用默认进程数

#重采样有问题，数据没进行重采样插值，虽然空间分辨率正确了，但是原来有数值的区域插值之后却没有数据

