

import os
import glob
import xarray as xr
import geopandas as gpd
import numpy as np
from shapely.prepared import prep
import re

def filter_and_report_gst_files(nc4_files, output_txt_path=None):
    """
    剔除文件名中GST后年月日小时完全重复的文件，只保留每组的第一个。
    同时统计所有可能的时间点，输出缺失时间点到txt。
    返回保留的文件路径列表。
    """
    import collections
    from datetime import datetime, timedelta
    seen = set()
    filtered = []
    time2file = collections.defaultdict(list)
    all_dates = set()
    all_hours = set()
    for fp in nc4_files:
        fname = os.path.basename(fp)
        m = re.search(r'GST-(\d{8})(\d{2})', fname)
        if m:
            date_str = m.group(1)  # YYYYMMDD
            hour_str = m.group(2)  # HH
            key = m.group(0)  # GST-YYYYMMDDHH
            time2file[key].append(fp)
            all_dates.add(date_str)
            all_hours.add(hour_str)
            if key not in seen:
                seen.add(key)
                filtered.append(fp)
            else:
                print(f"剔除重复GST时间文件: {fname}")
        else:
            filtered.append(fp)
    # 推断所有可能的时间点（日期和小时的笛卡尔积）
    if all_dates and all_hours:
        all_times_possible = set()
        for date_str in all_dates:
            for hour_str in all_hours:
                all_times_possible.add(f"GST-{date_str}{hour_str}")
        missing_times = sorted(list(all_times_possible - set(time2file.keys())))
    else:
        missing_times = []
    if output_txt_path is not None:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for t in missing_times:
                f.write(f"{t}\n")
        print(f"缺失时间点已写入: {output_txt_path}")
    return filtered

def filter_gst_duplicate_files(nc4_files):
    """
    剔除文件名中GST后年月日小时完全重复的文件，只保留每组的第一个。
    返回保留的文件路径列表。
    """
    seen = set()
    filtered = []
    for fp in nc4_files:
        fname = os.path.basename(fp)
        # 匹配GST-后8位日期+2位小时
        m = re.search(r'GST-(\d{8})(\d{2})', fname)
        if m:
            key = m.group(0)  # GST-YYYYMMDDHH
            if key not in seen:
                seen.add(key)
                filtered.append(fp)
            else:
                print(f"剔除重复GST时间文件: {fname}")
        else:
            # 没有匹配到的也保留
            filtered.append(fp)
    return filtered

def clean_filename(filename):
    """移除文件名中的(1), (2)等标记"""
    base, ext = os.path.splitext(filename)
    clean_base = re.sub(r'\(\d+\)$', '', base)
    return clean_base + ext

def delete_duplicate_files(file_path, output_dir = None ):
    """
    删除文件名中GST后年月日小时完全重复的文件，只保留每组的第一个。
    返回保留的文件路径列表。
    """
        # 查找所有nc文件
    search_pattern = os.path.join(file_path, '*.nc')
    print(f"搜索模式: {search_pattern}")
    nc4_files = glob.glob(search_pattern)
    print(f"找到 {len(nc4_files)} 个nc文件")
    # 剔除GST后年月日小时重复的文件，并输出缺失时间点
    output_txt_path = os.path.join(file_path, 'missing_gst_times.txt')
    nc4_files = filter_and_report_gst_files(nc4_files, output_txt_path=output_txt_path)
    print(f"去重后剩余 {len(nc4_files)} 个nc文件")

    # 将去重后的文件移动到output_dir
    output_dir = output_dir + '_filtered' if not output_dir.endswith('_filtered') else output_dir
    os.makedirs(output_dir, exist_ok=True)
    for fp in nc4_files:
        fname = os.path.basename(fp)
        # 提取GST-年月日小时
        m = re.search(r'GST-(\d{8})(\d{2})', fname)
        if m:
            date_str = m.group(1)
            hour_str = m.group(2)
            new_fname = f"CLDAS_{date_str}_{hour_str}.nc"
        else:
            new_fname = clean_filename(fname)
        dst = os.path.join(output_dir, new_fname)
        if os.path.abspath(fp) != os.path.abspath(dst):
            try:
                os.rename(fp, dst)
                print(f"已移动: {fp} -> {dst}")
            except Exception as e:
                print(f"移动文件失败: {fp} -> {dst}, 错误: {e}")


def process_nc4_files(input_dir, output_dir, shp_path, lat_range=[20, 50], lon_range=[70, 140]):
    """
    处理nc文件：使用shp文件裁剪数据
    shp_path: shapefile路径
    lat_range: 纬度范围 [最小值, 最大值]
    lon_range: 经度范围 [最小值, 最大值]
    """
    print(f"开始处理文件夹: {input_dir}")
    print(f"输出文件夹: {output_dir}")
    print(f"使用裁剪文件: {shp_path}")
    
   
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取shp文件
    try:
  
        mask_shp = gpd.read_file(shp_path)
        mask_shp = mask_shp.to_crs(epsg=4326)  # 转换为WGS84坐标系
        # 加速空间查询
        prepared_geometry = prep(mask_shp.geometry.unary_union)
        print("成功读取并优化shapefile")
    except Exception as e:
        print(f"读取shapefile出错: {str(e)}")
        return
    
    # 查找所有nc文件
    search_pattern = os.path.join(input_dir, '*.nc')
    print(f"搜索模式: {search_pattern}")
    nc4_files = glob.glob(search_pattern)
    print(f"找到 {len(nc4_files)} 个nc文件")
    '''
    # 剔除GST后年月日小时重复的文件
    nc4_files = filter_gst_duplicate_files(nc4_files)
    print(f"去重后剩余 {len(nc4_files)} 个nc文件")
    '''
    

    if len(nc4_files) == 0:
        print("警告：没有找到nc文件！")
        all_files = os.listdir(input_dir)
        print(f"目录中的所有文件: {all_files}")
        return
    
    # 缓存掩膜
    mask_cache = {}
    
    for file_path in nc4_files:
        original_filename = os.path.basename(file_path)
        if 'DAY' in original_filename.upper():
            print(f"文件名包含'DAY'，跳过: {original_filename}")
            continue
        print(f"\n处理文件: {file_path}")
        clean_name = clean_filename(original_filename)
        output_path = os.path.join(output_dir, clean_name)
        print(f"输出路径: {output_path}")
        if os.path.exists(output_path):
            print(f"文件已存在，跳过: {output_path}")
            continue
        try:
            print("尝试读取nc文件...")
            with xr.open_dataset(file_path) as ds:
                print("文件读取成功，开始裁剪...")
                print(f"数据集信息:\n{ds}")
                # 首先按经纬度范围裁剪
                ds_cropped = ds.sel(
                    LAT=slice(lat_range[0], lat_range[1]),
                    LON=slice(lon_range[0], lon_range[1])
                )
                # 检查缓存中是否已有掩膜
                grid_key = f"{ds_cropped.LON.values.tobytes()}{ds_cropped.LAT.values.tobytes()}"
                if grid_key in mask_cache:
                    mask = mask_cache[grid_key]
                else:
                    # 创建网格点
                    lons, lats = np.meshgrid(ds_cropped.LON, ds_cropped.LAT)
                    points = [gpd.points_from_xy([lon], [lat])[0] 
                            for lon, lat in zip(lons.flatten(), lats.flatten())]
                    # 向量化检查点是否在多边形内
                    mask = np.array([prepared_geometry.contains(point) for point in points])
                    mask = mask.reshape(lons.shape)
                    mask_cache[grid_key] = mask
                ds_masked = ds_cropped.where(mask)
                # 新增：去除全为NaN的行和列，使输出nc文件的行列数为实际有效范围
                ds_masked = ds_masked.dropna(dim='LAT', how='all')
                ds_masked = ds_masked.dropna(dim='LON', how='all')
                print("裁剪完成，准备重采样...")
                '''                
                # 计算1000m分辨率对应的经纬度步长（约0.008333度）
                target_res = 0.008333
                min_lat = float(ds_masked.LAT.min())
                max_lat = float(ds_masked.LAT.max())
                min_lon = float(ds_masked.LON.min())
                max_lon = float(ds_masked.LON.max())
                new_lats = np.arange(min_lat, max_lat + target_res, target_res)
                new_lons = np.arange(min_lon, max_lon + target_res, target_res)
                # 保证降序（如果原始是降序）
                if ds_masked.LAT.values[0] > ds_masked.LAT.values[-1]:
                    new_lats = new_lats[::-1]
                ds_resampled = ds_masked.interp(LAT=new_lats, LON=new_lons, method="linear")
                print("重采样完成，准备保存...")
                '''
                ds_masked.to_netcdf(output_path)
                print(f"成功保存文件: {output_path}")
        except Exception as e:
            print(f"处理文件时出错 {file_path}")
            print(f"错误详情: {str(e)}")
            print(f"错误类型: {type(e)}")

if __name__ == "__main__":


    input_dir = "G:\\cldas\data\\201712and201901"
    output_dir = "G:\\cldas\\data\\201712and201901_clip"
    shp_path = "G:\\Agr-qu\\clip.shp"  

    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在: {input_dir}")
    else:
        print(f"输入目录存在: {input_dir}")
 
        process_nc4_files(input_dir, output_dir, shp_path)
    
    
    input_dir1 = "G:\\cldas\\data\\201712and201901_clip"
    output_dir1 = "G:\\cldas\\data\\201712and201901_clip"

    if not os.path.exists(input_dir1):
        print(f"错误：输入目录不存在: {input_dir1}")
    else:
        print(f"输入目录存在: {input_dir1}")

        delete_duplicate_files(input_dir1, output_dir1)
