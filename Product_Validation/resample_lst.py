import os
import re
import rasterio
import rasterio.merge
from rasterio.enums import Resampling
import numpy as np
from typing import List, Dict

def create_output_filename(original_filenames: List[str]) -> str:
    """
    从原始文件名列表（同一时间点）中解析日期和时间，生成新的目标文件名。
    """
    # 以第一个文件名进行解析
    original_filename = original_filenames[0]
    # 匹配 LST_日期_时间_UTC
    # 注意：我们假设所有同一时间点的文件前缀都是相同的 LST_YYYYMMDD_HHMMSS_UTC
    match = re.match(r'LST_(\d{8})_(\d{6})_UTC', original_filename, re.IGNORECASE)
    
    if match:
        date_str = match.group(1) # 20181003
        time_str = match.group(2) # 024030
        return f'Landsat8_LST_{date_str}_{time_str}_UTC.tif'
    else:
        # 如果格式不匹配，使用一个通用的名称
        return f'Landsat8_LST_Merged_Resampled_{len(original_filenames)}_Files.tif'

def resample_and_rename(input_filepath: str, output_filepath: str, target_res: float):
    """
    使用平均值方法重采样栅格文件到指定的分辨率，将数据类型转为float32，并设置NoData为NaN。
    """
    try:
        with rasterio.open(input_filepath) as src:
            output_dtype = np.float32 
            
            src_res_x = src.res[0]
            src_res_y = src.res[1]
            scale_factor_x = src_res_x / target_res
            scale_factor_y = src_res_y / target_res
            new_height = int(src.height * scale_factor_y)
            new_width = int(src.width * scale_factor_x)

            # 读取和重采样数据 
            data_resampled = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.average
            )
            
            # 转换为 float32 并设置 NoData 为 NaN
            data_resampled = data_resampled.astype(output_dtype)
            
            # 更新 GeoTransform
            new_transform = rasterio.transform.from_bounds(
                src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, 
                new_width, new_height
            )

            # 更新元数据 Profile
            profile = src.profile
            profile.update({
                'height': new_height,
                'width': new_width,
                'transform': new_transform,
                'crs': src.crs,
                'dtype': output_dtype,      # 确保输出是浮点型
                'nodata': np.nan,           # 设置 NoData 标记为 NaN
                'count': src.count
            })

            # 写入新文件
            with rasterio.open(output_filepath, 'w', **profile) as dst:
                dst.write(data_resampled)

            print(f"✅ 重采样成功: {os.path.basename(output_filepath)} (NoData=NaN, Dtype=float32)")

    except Exception as e:
        print(f"❌ 处理文件 {os.path.basename(input_filepath)} 时出错: {e}")
        raise # 重新抛出错误，以便主函数可以删除临时文件

def group_files_by_time(file_list: List[str]) -> Dict[str, List[str]]:
    """
    根据 LST_日期_时间_UTC 字符串将文件分组。
    """
    groups = {}
    pattern = re.compile(r'(LST_\d{8}_\d{6}_UTC)')
    
    for filename in file_list:
        match = pattern.search(filename)
        if match:
            key = match.group(1)
            if key not in groups:
                groups[key] = []
            groups[key].append(filename)
    return groups

def main():
    """主函数，负责文件遍历、合并、重采样流程。"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 获取并分组文件
    all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.tif')]
    if not all_files:
        print(f"⚠️ 在目录 {INPUT_DIR} 中没有找到任何 .tif 文件。")
        return

    grouped_files = group_files_by_time(all_files)
    
    if not grouped_files:
        print(f"⚠️ 找不到匹配 LST_YYYYMMDD_HHMMSS_UTC 模式的文件。请检查文件名格式。")
        return

    print(f"📂 找到 {len(all_files)} 个文件，分为 {len(grouped_files)} 个时间组。")
    
    # 2. 循环处理每个时间组
    for time_key, filenames in grouped_files.items():
        print(f"\n--- ⚙️ 正在处理时间组: {time_key} ({len(filenames)} 个文件) ---")
        
        # 路径列表
        src_files_to_open = [os.path.join(INPUT_DIR, f) for f in filenames]
        
        # 生成输出文件名和临时文件名
        output_filename = create_output_filename(filenames)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        # 临时文件用于存储合并结果
        temp_merged_path = os.path.join(OUTPUT_DIR, f"temp_merged_{time_key}.tif")

        # --- A. 合并 (Mosaic) ---
        print(f"   ① 正在合并文件...")
        try:
            # 打开源文件
            src_files = [rasterio.open(path) for path in src_files_to_open]
            
            # 使用 merge.merge 进行合并。注意：这是在内存中完成的。
            merged_data, merged_transform = rasterio.merge.merge(src_files)
            
            # 获取第一个文件的Profile作为模板
            profile = src_files[0].profile
            
            # 更新 Profile 以适应合并后的数据
            profile.update({
                "driver": "GTiff",
                "height": merged_data.shape[1],
                "width": merged_data.shape[2],
                "transform": merged_transform,
                "count": merged_data.shape[0],
                "nodata": src_files[0].nodata # 保留原始 NoData 进行重采样
            })
            
            # 关闭源文件
            for src in src_files:
                src.close()
                
            # 写入临时合并文件
            with rasterio.open(temp_merged_path, "w", **profile) as dst:
                dst.write(merged_data)
                
            print(f"   👍 合并完成，临时文件保存至: {os.path.basename(temp_merged_path)}")

        except Exception as e:
            print(f"   ❌ 合并 {time_key} 时出错: {e}")
            continue

        # --- B. 重采样 ---
        print(f"   ② 正在对合并文件进行重采样...")
        try:
            resample_and_rename(temp_merged_path, output_path, TARGET_RESOLUTION)
        except Exception as e:
            # 错误信息已在函数内部打印
            print(f"   ❌ 重采样 {time_key} 时失败。")
        
        # --- C. 清理 ---
        if os.path.exists(temp_merged_path):
            os.remove(temp_merged_path)
            print(f"   🧹 已清理临时文件: {os.path.basename(temp_merged_path)}")
            
    print("\n🎉 所有时间组处理完毕!")

if __name__ == "__main__":
    INPUT_DIR = r'G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\output\interpolation_v5\20181006_03_LST.tif'  # 原始LST文件所在的目录
    OUTPUT_DIR = r'G:\CNN_SpatialDownscaling\20181006_03_LST.tif' # 处理后的文件输出目录
    TARGET_RESOLUTION = 0.0625  # 目标分辨率 (度)0.0083333333
    
    resample_and_rename(INPUT_DIR, OUTPUT_DIR, TARGET_RESOLUTION)
    '''# --- 配置参数 ---
    INPUT_DIR = r'D:\lyygi\Downloads\drive-download-20251117T064059Z-1-001'  # 原始LST文件所在的目录
    OUTPUT_DIR = r'D:\lyygi\Downloads\drive-download-20251117T064059Z-1-001\output_lst1' # 处理后的文件输出目录
    TARGET_RESOLUTION = 0.0625  # 目标分辨率 (度)0.0083333333
    # ------------------
    main()'''