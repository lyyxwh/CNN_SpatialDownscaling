import os
import numpy as np
from netCDF4 import Dataset, num2date

def separate_by_time_var(ds, time_var_name, data_var, output_dir, prefix):
    time_var = ds.variables[time_var_name]
    times = num2date(time_var[:], units=time_var.units)
    data = data_var[:]  # shape: (time, height, width)
    days = np.array([t.replace(hour=0, minute=0, second=0, microsecond=0) for t in times])
    unique_days = np.unique(days)
    for day in unique_days:
        idx = np.where(days == day)[0]
        if len(idx) == 0:
            continue
        day_data = np.nanmean(data[idx], axis=0)
        day_str = day.strftime('%Y%m%d')
        out_path = os.path.join(output_dir, f'{prefix}_day_{day_str}.npy')
        np.save(out_path, day_data)
        print(f"Saved: {out_path}")

def separate_modis_by_time(nc_path, output_dir):
    """
    按24小时分组分离MODIS像元，支持Hight_view_time和Day_view_time
    :param nc_path: MODIS NC 文件路径
    :param output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with Dataset(nc_path, 'r') as ds:
        data_var = ds.variables['data']  # shape: (time, height, width)
        # 分别处理两个时间变量
        for time_var_name in ['Hight_view_time', 'Day_view_time']:
            if time_var_name in ds.variables:
                separate_by_time_var(ds, time_var_name, data_var, output_dir, time_var_name)
            else:
                print(f"Warning: {time_var_name} not found in file.")

if __name__ == '__main__':
    nc_path = r'输入你的MODIS NC文件路径.nc'
    output_dir = r'输出目录'
    separate_modis_by_time(nc_path, output_dir)
