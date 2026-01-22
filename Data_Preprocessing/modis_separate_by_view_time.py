import xarray as xr
import numpy as np
import os
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def batch_process_modis_data(input_dir, output_dir):
    """
    批量处理MODIS数据
    
    Parameters:
    input_dir: str, 包含NC文件的输入目录路径
    output_dir: str, 输出目录路径
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
    
    for i, nc_file in enumerate(nc_files, 1):
        print(f"\n[{i}/{len(nc_files)}] 处理文件: {os.path.basename(nc_file)}")
        print("-" * 50)
        
        try:
            files_created, stats = process_modis_hourly_data(nc_file, output_dir)
            total_files_processed += 1
            total_output_files += files_created
            
            print(f"✓ 文件处理完成，创建了 {files_created} 个输出文件")
            
        except Exception as e:
            print(f"✗ 文件处理失败: {e}")
            failed_files.append(nc_file)
    
    print(f"\n" + "="*60)
    print("批处理完成！")
    print(f"总处理文件数: {total_files_processed}/{len(nc_files)}")
    print(f"总输出文件数: {total_output_files}")
    print(f"失败文件数: {len(failed_files)}")
    
    if failed_files:
        print(f"\n失败的文件:")
        for failed_file in failed_files:
            print(f"  - {os.path.basename(failed_file)}")

def process_modis_hourly_data(input_file, output_dir):
    """
    处理MODIS数据，按小时分离、质量筛选并统一变量名。
    
    Parameters:
    input_file: str, 输入的NC文件路径
    output_dir: str, 输出目录路径
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在读取文件: {input_file}")
    
    try:
        ds = xr.open_dataset(input_file)
        print("文件读取成功")
        print(f"数据变量: {list(ds.data_vars.keys())}")
        print(f"数据维度: {ds.dims}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return 0, {}
    
    base_date = None
    try:
        filename = os.path.basename(input_file)
        if 'A' in filename:
            date_part = filename.split('A')[1][:7]
            year = int(date_part[:4])
            day_of_year = int(date_part[4:7])
            base_date = datetime(year, 1, 1) + np.timedelta64(day_of_year - 1, 'D')
        else:
            base_date = datetime.now()
    except:
        base_date = datetime.now()
    
    print(f"基准日期: {base_date.strftime('%Y-%m-%d')}")
    
    required_vars = ['Day_view_time', 'Night_view_time', 'QC_Day', 'QC_Night']
    missing_vars = [var for var in required_vars if var not in ds.data_vars]
    if missing_vars:
        print(f"缺少必要变量: {missing_vars}")
        ds.close() # 确保关闭文件
        return 0, {}
    
    all_vars = list(ds.data_vars.keys())
    
    day_var_map = {
        'Day_view_time': 'view_time',
        'Day_view_angl': 'view_angl',
        'Clear_day_cov': 'Clear_cov',
        'LST_Day_1km': 'LST_1km',
        'QC_Day': 'QC'
    }
    night_var_map = {
        'Night_view_time': 'view_time',
        'Night_view_angl': 'view_angl',
        'Clear_night_cov': 'Clear_cov',
        'LST_Night_1km': 'LST_1km',
        'QC_Night': 'QC'
    }
    shared_vars = ['Emis_31', 'Emis_32']
    
    total_files_created = 0
    hour_stats = {}
    
    for hour in range(24):
        print(f"\n处理第 {hour:02d} 小时的数据...")
        
        # 预先创建一个空数据集来存储该小时的数据
        hour_ds = xr.Dataset(coords=ds.coords, attrs=ds.attrs)
        
        day_mask = None
        night_mask = None

        # --- 白天数据处理与重命名 ---
        if 'Day_view_time' in ds.data_vars and 'QC_Day' in ds.data_vars:
            day_hours = np.floor(ds['Day_view_time'].values).astype(int)
            day_hour_mask = (day_hours == hour)
            day_quality_mask = (ds['QC_Day'].values <= 1)
            day_mask = day_hour_mask & day_quality_mask
            
            if np.sum(day_mask) > 0:
                # 关键修改：将 numpy 数组转换为 xarray.DataArray
                day_mask_da = xr.DataArray(day_mask, dims=ds['QC_Day'].dims, coords=ds['QC_Day'].coords)
                
                day_data = ds.where(day_mask_da, drop=True)
                
                # 将白天变量重命名并添加到hour_ds中
                for old_name, new_name in day_var_map.items():
                    if old_name in day_data.data_vars:
                        # 检查原始变量是否需要重命名
                        if old_name in ds.data_vars:
                            hour_ds[new_name] = ds[old_name].where(day_mask_da)
                            hour_ds[new_name].attrs = ds[old_name].attrs
                            if hour_ds[new_name].dtype.kind == 'i':
                                hour_ds[new_name] = hour_ds[new_name].astype(float)
                                hour_ds[new_name].attrs['_FillValue'] = np.nan

        # --- 夜间数据处理与重命名 ---
        if 'Night_view_time' in ds.data_vars and 'QC_Night' in ds.data_vars:
            night_hours = np.floor(ds['Night_view_time'].values).astype(int)
            night_hour_mask = (night_hours == hour)
            night_quality_mask = (ds['QC_Night'].values <= 1)
            night_mask = night_hour_mask & night_quality_mask

            if np.sum(night_mask) > 0:
                # 关键修改：将 numpy 数组转换为 xarray.DataArray
                night_mask_da = xr.DataArray(night_mask, dims=ds['QC_Night'].dims, coords=ds['QC_Night'].coords)

                # 将夜间变量重命名并添加到hour_ds中
                for old_name, new_name in night_var_map.items():
                    if old_name in ds.data_vars:
                        night_data_da = ds[old_name].where(night_mask_da)
                        # 如果变量已存在（白天数据已写入），则用夜间数据更新
                        if new_name in hour_ds.data_vars:
                            hour_ds[new_name] = hour_ds[new_name].combine_first(night_data_da)
                        else:
                            hour_ds[new_name] = night_data_da
                        
                        hour_ds[new_name].attrs = ds[old_name].attrs
                        if hour_ds[new_name].dtype.kind == 'i':
                            hour_ds[new_name] = hour_ds[new_name].astype(float)
                            hour_ds[new_name].attrs['_FillValue'] = np.nan
        
        # --- 处理共用变量 ---
        has_valid_data = False
        valid_day_count = np.sum(day_mask) if day_mask is not None else 0
        valid_night_count = np.sum(night_mask) if night_mask is not None else 0
        if valid_day_count > 0 or valid_night_count > 0:
            has_valid_data = True
            
            # 关键修改：结合白天和夜间的有效掩码，并转换为 DataArray
            valid_mask = np.zeros_like(ds['LST_Day_1km'].values, dtype=bool) if 'LST_Day_1km' in ds.data_vars else \
                         np.zeros_like(ds['LST_Night_1km'].values, dtype=bool) if 'LST_Night_1km' in ds.data_vars else \
                         None
            if valid_mask is not None:
                if day_mask is not None:
                    valid_mask |= day_mask
                if night_mask is not None:
                    valid_mask |= night_mask
                
                valid_mask_da = xr.DataArray(valid_mask, dims=ds['LST_Day_1km'].dims if 'LST_Day_1km' in ds.data_vars else ds['LST_Night_1km'].dims, 
                                            coords=ds['LST_Day_1km'].coords if 'LST_Day_1km' in ds.data_vars else ds['LST_Night_1km'].coords)

                if np.sum(valid_mask_da) > 0:
                    for var in shared_vars:
                        if var in ds.data_vars:
                            hour_ds[var] = ds[var].where(valid_mask_da)
                            hour_ds[var].attrs = ds[var].attrs
                            if hour_ds[var].dtype.kind == 'i':
                                hour_ds[var] = hour_ds[var].astype(float)
                                hour_ds[var].attrs['_FillValue'] = np.nan
        
        hour_stats[hour] = {
            'valid_day_pixels': valid_day_count,
            'valid_night_pixels': valid_night_count,
            'has_data': has_valid_data
        }
        
        if has_valid_data:
            date_str = f"{base_date.year:04d}{base_date.month:02d}{base_date.day:02d}"
            output_file = os.path.join(output_dir, f'MYD11_{date_str}_{hour:02d}.nc')
            
            try:
                # 确保保存时使用浮点数编码处理NaN
                encoding = {}
                for var_name in hour_ds.data_vars:
                    if hour_ds[var_name].dtype.kind == 'f':
                        encoding[var_name] = {'_FillValue': np.nan, 'zlib': True, 'complevel': 4}
                    else:
                        encoding[var_name] = {'zlib': True, 'complevel': 4}
                        if '_FillValue' in ds[var_name].encoding:
                            encoding[var_name]['_FillValue'] = ds[var_name].encoding['_FillValue']

                hour_ds.to_netcdf(output_file, encoding=encoding)
                total_files_created += 1
                print(f"  ✓ 已创建文件: {os.path.basename(output_file)}")
                print(f"    白天有效像素: {valid_day_count}")
                print(f"    夜间有效像素: {valid_night_count}")
                
            except Exception as e:
                print(f"  ✗ 保存文件失败: {e}")
        else:
            print(f"  - 第{hour:02d}小时无有效数据，跳过")
    
    ds.close()
    
    print(f"\n" + "="*50)
    print("处理完成！")
    print(f"共创建文件数: {total_files_created}")
    print(f"输出目录: {output_dir}")
    
    print(f"\n各小时数据统计:")
    print(f"{'小时':<4} {'白天像素':<8} {'夜间像素':<8} {'是否输出':<8}")
    print("-" * 32)
    for hour in range(24):
        stats = hour_stats[hour]
        status = "是" if stats['has_data'] else "否"
        print(f"{hour:02d}   {stats['valid_day_pixels']:<8} {stats['valid_night_pixels']:<8} {status:<8}")
    
    return total_files_created, hour_stats

if __name__ == "__main__":
    print("MODIS数据按小时分离与质量筛选工具")
    print("="*50)
    
    input_dir = "G:\\CNN_SpatialDownscaling\\MYD11\\2018"
    output_dir = "G:\\CNN_SpatialDownscaling\\MYD11\\2018\\output_hourly_data"
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        print("请修改 input_dir 变量为正确的目录路径")
    elif not os.path.isdir(input_dir):
        print(f"错误: {input_dir} 不是一个目录")
    else:
        try:
            processed, created, failed = batch_process_modis_data(input_dir, output_dir)
            print(f"\n批处理成功完成!")
            print(f"处理文件: {processed} 个")
            print(f"创建文件: {created} 个")
            if failed:
                print(f"失败文件: {len(failed)} 个")
        except Exception as e:
            print(f"批处理过程中出现错误: {e}")