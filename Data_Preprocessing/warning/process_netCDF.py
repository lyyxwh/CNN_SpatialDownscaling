import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# 设置包含 netCDF 文件的目录
data_dir = r'g:\\CNN_SpatialDownscaling\\MYD11\\2018'
# 设置保存统计图的目录
output_dir = os.path.join(data_dir, '..', 'MYD11')
os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建

# 获取目录下所有 .nc 文件，并排序
file_list = sorted(glob.glob(os.path.join(data_dir, '*.nc')))

# 只处理前 17 个文件
file_list = file_list[:17]

day_all = []
night_all = []

for i, f in enumerate(file_list):
    nc = Dataset(f, 'r')
    
    # 打印当前文件的变量信息
    #print(f"File {i + 1}: {f}")
    #print("Variables:", list(nc.variables.keys()))
    
    day_data = np.array(nc.variables['Day_view_time'][:]).flatten()
    night_data = np.array(nc.variables['Night_view_time'][:]).flatten()
    nc.close()

    # 计算当前文件的中间 50% 落点范围
    day_q25, day_q75 = np.percentile(day_data, [25, 75])
    night_q25, night_q75 = np.percentile(night_data, [25, 75])

    # 获取文件名（不包含路径）
    file_name = os.path.basename(f)

    # 绘制当前文件的 Day_view_time 统计图
    fig_day, ax_day = plt.subplots(figsize=(7, 6))
    ax_day.hist(day_data, bins=50, color='skyblue', edgecolor='black')
    ax_day.set_title(f'Day_view_time Frequency ({file_name})')
    ax_day.set_xlabel('Value')
    ax_day.set_ylabel('Frequency')
    ax_day.axvline(day_q25, color='red', linestyle='--', label=f'25%: {day_q25:.2f}')
    ax_day.axvline(day_q75, color='green', linestyle='--', label=f'75%: {day_q75:.2f}')
    ax_day.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Day_view_time_statistics_file_{i + 1}.png'))
    plt.close(fig_day)  # 关闭 Day_view_time 图形对象

    # 绘制当前文件的 Night_view_time 统计图
    fig_night, ax_night = plt.subplots(figsize=(7, 6))
    ax_night.hist(night_data, bins=50, color='lightcoral', edgecolor='black')
    ax_night.set_title(f'Night_view_time Frequency ({file_name})')
    ax_night.set_xlabel('Value')
    ax_night.set_ylabel('Frequency')
    ax_night.axvline(night_q25, color='red', linestyle='--', label=f'25%: {night_q25:.2f}')
    ax_night.axvline(night_q75, color='green', linestyle='--', label=f'75%: {night_q75:.2f}')
    ax_night.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Night_view_time_statistics_file_{i + 1}.png'))
    plt.close(fig_night)  # 关闭 Night_view_time 图形对象

    day_all.extend(day_data)
    night_all.extend(night_data)

day_all = np.array(day_all)
night_all = np.array(night_all)

# 计算中间 50% 落点范围 (25% 到 75%分位)
day_q25, day_q75 = np.percentile(day_all, [25, 75])
night_q25, night_q75 = np.percentile(night_all, [25, 75])

# 分别绘制 Day_view_time 统计图
fig_day, ax_day = plt.subplots(figsize=(7, 6))
ax_day.hist(day_all, bins=50, color='skyblue', edgecolor='black')
ax_day.set_title('Day_view_time Frequency')
ax_day.set_xlabel('Value')
ax_day.set_ylabel('Frequency')
ax_day.axvline(day_q25, color='red', linestyle='--', label=f'25%: {day_q25:.2f}')
ax_day.axvline(day_q75, color='green', linestyle='--', label=f'75%: {day_q75:.2f}')
ax_day.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Day_view_time_statistics.png'))
plt.close(fig_day)  # 关闭整体 Day_view_time 图形对象

# 分别绘制 Night_view_time 统计图
fig_night, ax_night = plt.subplots(figsize=(7, 6))
ax_night.hist(night_all, bins=50, color='lightcoral', edgecolor='black')
ax_night.set_title('Night_view_time Frequency')
ax_night.set_xlabel('Value')
ax_night.set_ylabel('Frequency')
ax_night.axvline(night_q25, color='red', linestyle='--', label=f'25%: {night_q25:.2f}')
ax_night.axvline(night_q75, color='green', linestyle='--', label=f'75%: {night_q75:.2f}')
ax_night.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Night_view_time_statistics.png'))
plt.close(fig_night)  # 关闭整体 Night_view_time 图形对象

plt.show()
