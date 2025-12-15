import xarray as xr
import numpy as np
import os
import glob
from datetime import datetime, date, timedelta
import warnings
import multiprocessing
import time
import hashlib

warnings.filterwarnings('ignore')

def process_modis_hourly_data(input_file, output_dir):
    """
    处理MODIS数据，按小时分离、质量筛选并统一变量名。
    此版本在子进程内部直接将数据写入文件。
    返回创建的文件数量，以及一个布尔值指示是否处理成功。
    已增强：将 Day/Night view_time 从本地太阳时/观测时转换为 UTC（世界时），
    并根据 UTC 小时和可能的日期偏移（-1/0/+1 天）输出对应文件。
    """
    
    os.makedirs(output_dir, exist_ok=True) 
    
    print(f"[{os.getpid()}] 正在读取文件: {os.path.basename(input_file)}")
    
    try:
        ds = xr.open_dataset(input_file)
        print(f"[{os.getpid()}] 文件读取成功: {os.path.basename(input_file)}")
    except Exception as e:
        print(f"[{os.getpid()}] 读取文件失败: {os.path.basename(input_file)}, 错误: {e}")
        return 0, False
    
    # --- 日期提取（保持原有多模式解析） ---
    base_date_for_output = None
    filename_base = os.path.basename(input_file)
    try:
        parts = filename_base.split('_')
        date_str_part = None
        for p in parts:
            if len(p) == 10 and p[4] == '-' and p[7] == '-':
                date_str_part = p
                break
        if date_str_part:
            base_date_for_output = datetime.strptime(date_str_part, '%Y-%m-%d').date()
    except Exception:
        base_date_for_output = None

    if base_date_for_output is None and 'A' in filename_base:
        try:
            date_part_modis = filename_base.split('A')[1][:7]
            year = int(date_part_modis[:4])
            day_of_year = int(date_part_modis[4:7])
            base_date_for_output = date.fromordinal(date(year, 1, 1).toordinal() + day_of_year - 1)
        except Exception:
            base_date_for_output = None

    if base_date_for_output is None:
        print(f"[{os.getpid()}] 警告: 无法从文件名 '{filename_base}' 中提取日期。将使用文件名的MD5哈希值作为日期前缀。")
        file_hash = hashlib.md5(filename_base.encode()).hexdigest()[:8]
        date_prefix_for_filename = f"NODATE_{file_hash}"
    else:
        date_prefix_for_filename = f"{base_date_for_output.year:04d}{base_date_for_output.month:02d}{base_date_for_output.day:02d}"
    print(f"[{os.getpid()}] 输出文件日期前缀: {date_prefix_for_filename}")
    # --- 结束日期提取修改 ---
    
    required_vars = ['Day_view_time', 'Night_view_time', 'QC_Day', 'QC_Night']
    missing_vars = [var for var in required_vars if var not in ds.data_vars]
    if missing_vars:
        print(f"[{os.getpid()}] 缺少必要变量: {missing_vars} for {os.path.basename(input_file)}")
        ds.close()
        return 0, False

    # 尝试获取经度数组（与 view_time 维度一致）
    def _find_lon_array(dataset):
        # 常见坐标/变量名候选
        candidates = ['lon', 'longitude', 'LON', 'LONGITUDE', 'x']
        for key in candidates:
            if key in dataset.coords:
                return dataset.coords[key].values
            if key in dataset.data_vars:
                return dataset[key].values
        # 有时经度是二元坐标 (lon, lat) 形式，尝试从索引的属性恢复
        for dim in dataset.dims:
            arr = dataset.coords.get(dim, None)
            if arr is not None and 'lon' in dim.lower():
                return arr.values
        return None

    lon_arr = _find_lon_array(ds)
    if lon_arr is None:
        print(f"[{os.getpid()}] 警告: 未在文件中找到经度信息，无法将 view_time 转换为精确 UTC。将直接使用原始 view_time 作为 UTC（不推荐）。")
    else:
        # 确保 lon_arr 与 view_time 的形状一致（广播）
        try:
            # 如果 lon_arr 是 1D lon 或 2D mesh，广播到 view_time 形状
            ref_shape = ds['Day_view_time'].shape
            if lon_arr.shape != ref_shape:
                # 如果 lon_arr 1D 且与最后一个维度匹配，广播
                if lon_arr.ndim == 1 and lon_arr.size == ref_shape[-1]:
                    lon_arr = np.broadcast_to(lon_arr, ref_shape)
                else:
                    # 尝试转置或广播其他方式失败则置为 None
                    lon_arr = np.broadcast_to(lon_arr, ref_shape)
        except Exception:
            lon_arr = None
            print(f"[{os.getpid()}] 警告: 经度数组无法与 view_time 对齐，跳过经度校正。")

    # 准备输出容器：以 (date_prefix, hour) 为键收集 hour_ds
    outputs_map = {}  # key -> xr.Dataset
    
    shared_vars = ['Emis_31', 'Emis_32']
    day_var_map = {
        'Day_view_time': 'view_time', 'Day_view_angl': 'view_angl', 'Clear_day_cov': 'Clear_cov', 'LST_Day_1km': 'LST_1km', 'QC_Day': 'QC'
    }
    night_var_map = {
        'Night_view_time': 'view_time', 'Night_view_angl': 'view_angl', 'Clear_night_cov': 'Clear_cov', 'LST_Night_1km': 'LST_1km', 'QC_Night': 'QC'
    }

    files_created_count = 0 

    # 遍历 day/night 两个时类，计算 UTC hour 和 day offset，填入对应 outputs_map
    for period, var_map, view_var, qc_var in [
        ('day', day_var_map, 'Day_view_time', 'QC_Day'),
        ('night', night_var_map, 'Night_view_time', 'QC_Night')
    ]:
        if view_var not in ds.data_vars or qc_var not in ds.data_vars:
            continue

        view_vals = ds[view_var].values
        qc_vals = ds[qc_var].values
        # 原始掩码：观测小时匹配 & 质量阈值
        base_hours = np.floor(view_vals).astype(int)
        base_hour_mask_all = np.isfinite(view_vals)  # 有效 view_time 的基础掩码
        quality_mask = (qc_vals <= 2)

        if lon_arr is None:
            # 无经度信息：直接当作 UTC 处理（不变）
            utc_time = view_vals.copy()
            print(f"[{os.getpid()}] 警告: 未找到经度数组，使用原始 view_time 作为 UTC（请确认 view_time 单位）")
        else:
            # 规范化经度到 [-180, 180]（处理 0-360 情况）
            try:
                lon = np.array(lon_arr, dtype=float)
                if lon.max() > 180:
                    lon = ((lon + 180.0) % 360.0) - 180.0

                # 广播到 view_vals 形状（若需要）
                ref_shape = view_vals.shape
                if lon.shape != ref_shape:
                    if lon.ndim == 1 and lon.size == ref_shape[-1]:
                        lon = np.broadcast_to(lon, ref_shape)
                    else:
                        lon = np.broadcast_to(lon, ref_shape)

                lon_hours = lon / 15.0  # 每 15° = 1 小时
                # 若 view_vals 是本地太阳时（local solar time），则 UTC = local - lon/15
                utc_time = view_vals - lon_hours
            except Exception as e:
                print(f"[{os.getpid()}] 经度广播/转换失败，回退为原始 view_time，错误: {e}")
                utc_time = view_vals.copy()

        # raw_hour 可能不在 0-23，需要提取 hour 和 day_offset
        raw_hour = np.floor(utc_time).astype(int)
        utc_hour = (raw_hour % 24).astype(int)
        # day_offset：raw_hour 除以 24 的整数商（可能为 -1,0,1）
        day_offset = np.floor_divide(raw_hour, 24).astype(int)

        # 只保留原始基础掩码且质量通过的位置
        valid_mask = base_hour_mask_all & (quality_mask)

        # 对每个可能的 day_offset (-1,0,1) 和每个小时构建 mask 并填入对应输出 dataset
        unique_offsets = np.unique(day_offset[valid_mask]) if np.any(valid_mask) else np.array([], dtype=int)
        unique_offsets = np.concatenate(([0], unique_offsets)) if unique_offsets.size == 0 else unique_offsets

        for offset in unique_offsets:
            for hour in range(24):
                sel_mask = valid_mask & (utc_hour == hour) & (day_offset == offset)
                if not np.any(sel_mask):
                    continue

                # 目标日期前缀
                if isinstance(base_date_for_output, date):
                    # 使用 datetime.timedelta 处理日期偏移，避免 numpy datetime 与 date 相加类型错误
                    target_date = base_date_for_output + timedelta(days=int(offset))
                    date_prefix = f"{target_date.year:04d}{target_date.month:02d}{target_date.day:02d}"
                else:
                    # 无法解析基准日期，使用原有前缀并附加偏移后缀
                    date_prefix = f"{date_prefix_for_filename}_d{offset:+d}"

                key = (date_prefix, hour)
                if key not in outputs_map:
                    # 新建 hour_ds（保留 coords/attrs）
                    hour_ds = xr.Dataset(coords=ds.coords, attrs=ds.attrs)
                    outputs_map[key] = hour_ds
                else:
                    hour_ds = outputs_map[key]

                # 将对应 period 的变量按 mask 写入 hour_ds（注意合并已有变量）
                for old_name, new_name in var_map.items():
                    if old_name in ds.data_vars:
                        src = ds[old_name].where(xr.DataArray(sel_mask, dims=ds[old_name].dims, coords=ds[old_name].coords))
                        if new_name in hour_ds.data_vars:
                            hour_ds[new_name] = hour_ds[new_name].combine_first(src)
                        else:
                            hour_ds[new_name] = src
                        hour_ds[new_name].attrs = ds[old_name].attrs.copy()
                        if '_FillValue' in hour_ds[new_name].attrs:
                            del hour_ds[new_name].attrs['_FillValue']

                # 共享变量也按同一掩码写入（仅第一次写入或合并）
                for var in shared_vars:
                    if var in ds.data_vars:
                        src = ds[var].where(xr.DataArray(sel_mask, dims=ds[var].dims, coords=ds[var].coords))
                        if var in hour_ds.data_vars:
                            hour_ds[var] = hour_ds[var].combine_first(src)
                        else:
                            hour_ds[var] = src
                        hour_ds[var].attrs = ds[var].attrs.copy()
                        if '_FillValue' in hour_ds[var].attrs:
                            del hour_ds[var].attrs['_FillValue']

    # 遍历 outputs_map 并写出文件
    try:
        for (date_prefix, hour), hour_ds in outputs_map.items():
            if sum([np.sum(hour_ds[var].values) if hour_ds[var].dtype.kind != 'f' else np.count_nonzero(~np.isnan(hour_ds[var].values)) for var in hour_ds.data_vars]) == 0:
                # 跳过空数据集
                continue

            output_file = os.path.join(output_dir, f'MOD11_{date_prefix}_{hour:02d}.nc')
            try:
                encoding = {}
                for var_name in hour_ds.data_vars:
                    if hour_ds[var_name].dtype.kind == 'f':
                        # 设置缺测值为 0（保持历史兼容）
                        encoding[var_name] = {'_FillValue': 0.0, 'zlib': True, 'complevel': 4}
                    else:
                        encoding[var_name] = {'zlib': True, 'complevel': 4}
                
                if 'QC' in hour_ds.data_vars and hour_ds['QC'].dtype.kind == 'f':
                    encoding['QC']['_FillValue'] = np.nan

                hour_ds.to_netcdf(output_file, encoding=encoding)
                files_created_count += 1
                print(f"[{os.getpid()}] ✓ 已创建文件: {os.path.basename(output_file)}")
            except Exception as e:
                print(f"[{os.getpid()}] ✗ 写入文件失败: {output_file}, 错误: {e}")
                return 0, False

    except Exception as e:
        print(f"[{os.getpid()}] 写出 outputs_map 失败: {e}")
        ds.close()
        return 0, False

    ds.close()
    return files_created_count, True

# batch_process_modis_data 函数保持不变
def batch_process_modis_data(input_dir, output_dir):
    """
    批量处理MODIS数据（并行版本，逐文件写入）。
    每个子进程独立处理一个输入文件并写入其所有输出文件。
    """
    
    os.makedirs(output_dir, exist_ok=True) 
    
    nc_files = glob.glob(os.path.join(input_dir, "*.nc"))
    
    if not nc_files:
        print(f"在目录 {input_dir} 中未找到NC文件")
        return
    
    print(f"找到 {len(nc_files)} 个NC文件")
    print("="*60)
    
    total_files_processed = 0
    total_output_files = 0
    failed_files = []
    
    start_time = time.time()
    
    tasks = [(nc_file, output_dir) for nc_file in nc_files]
    
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = pool.starmap(process_modis_hourly_data, tasks)

    for i, (files_created, success_status) in enumerate(results):
        nc_file = nc_files[i]
        total_files_processed += 1
        
        if success_status:
            total_output_files += files_created
            print(f"✓ 文件 {os.path.basename(nc_file)} 处理完成，创建了 {files_created} 个输出文件")
        else:
            print(f"✗ 文件 {os.path.basename(nc_file)} 处理失败")
            failed_files.append(nc_file)
    
    end_time = time.time()
    
    print(f"\n" + "="*60)
    print("批处理完成！")
    print(f"总处理文件数: {total_files_processed}/{len(nc_files)}")
    print(f"总输出文件数: {total_output_files}")
    print(f"失败文件数: {len(failed_files)}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    
    if failed_files:
        print(f"\n失败的文件:")
        for failed_file in failed_files:
            print(f"  - {os.path.basename(failed_file)}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    print("MODIS数据按小时分离与质量筛选工具 (并行版本，逐文件写入)")
    print("="*50)
    
    input_dir = "G:\\CNN_SpatialDownscaling\\201712and201901\\MOD11\\2017" # 注意这里从 MYD11 变成了 MOD11
    output_dir = r"G:\CNN_SpatialDownscaling\modis_lst\2018_mod\output_hourly_data1" # 注意这里从 MYD11 变成了 MOD11

    input_dir1 = "G:\\CNN_SpatialDownscaling\\201712and201901\\MYD11\\2017" # 注意这里从 MYD11 变成了 MOD11
    output_dir1 = r"G:\CNN_SpatialDownscaling\modis_lst\2018_myd\output_hourly_data1" # 注意这里从 MYD11 变成了 MOD11
    '''
    input_dir2 = r"D:\rawdata\MOD11\2018"
    output_dir2 = r"G:\CNN_SpatialDownscaling\modis_lst\2018_mod\output_hourly_data"

    input_dir3 = r"D:\rawdata\MYD11\2018"
    output_dir3 = r"G:\CNN_SpatialDownscaling\modis_lst\2018_myd\output_hourly_data"'''


    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        print("请修改 input_dir 变量为正确的目录路径")
    elif not os.path.isdir(input_dir):
        print(f"错误: {input_dir} 不是一个目录")
    else:
        try:
            batch_process_modis_data(input_dir, output_dir)
            print(f"\n{os.path.basename(input_dir)}所有任务处理完毕。")
            batch_process_modis_data(input_dir1, output_dir1)
            print(f"\n{os.path.basename(input_dir1)}所有任务处理完毕。")
            '''
            batch_process_modis_data(input_dir2, output_dir2)
            print(f"\n{os.path.basename(input_dir2)}所有任务处理完毕。")
            batch_process_modis_data(input_dir3, output_dir3)
            print(f"\n{os.path.basename(input_dir3)}所有任务处理完毕。")'''
        except Exception as e:
            print(f"批处理过程中出现错误: {e}")