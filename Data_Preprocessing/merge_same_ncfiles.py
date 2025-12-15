import os
import xarray as xr
import numpy as np

def merge_nc_folders(folder1, folder2, output_folder):
    """
    合并两个文件夹中同名的nc文件，若同一网格位置两个文件都有值则取平均值。
    
    参数:
        folder1 (str): 第一个输入文件夹路径
        folder2 (str): 第二个输入文件夹路径
        output_folder (str): 输出文件夹路径
    """

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 获取两个文件夹的文件名集合
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    
    # 找到同名文件
    common_files = files1.intersection(files2)

    for fname in common_files:
        path1 = os.path.join(folder1, fname)
        path2 = os.path.join(folder2, fname)

        # 打开两个文件
        ds1 = xr.open_dataset(path1)
        ds2 = xr.open_dataset(path2)

        merged_vars = {}
        for var in ds1.data_vars:
            if var in ds2.data_vars:
                # 对齐两个数据集
                da1, da2 = xr.align(ds1[var], ds2[var], join="outer")

                # 合并：两个都存在时取平均，一个存在时取该值
                merged = xr.where(np.isnan(da1), da2, 
                                  xr.where(np.isnan(da2), da1, (da1 + da2) / 2))
                merged_vars[var] = merged
            else:
                merged_vars[var] = ds1[var]

        # 补充 ds2 中独有的变量
        for var in ds2.data_vars:
            if var not in merged_vars:
                merged_vars[var] = ds2[var]

        # 重新构建 Dataset
        merged_ds = xr.Dataset(merged_vars, coords=ds1.coords)

        # 输出路径
        output_path = os.path.join(output_folder, fname)
        merged_ds.to_netcdf(output_path)

        print(f"已合并并保存: {output_path}")

        ds1.close()
        ds2.close()
        merged_ds.close()



if __name__ == "__main__":
    input_dir1 = r"G:\CNN_SpatialDownscaling\modis_lst\2018_mod\output_hourly_data"
    input_dir2 = r"G:\CNN_SpatialDownscaling\modis_lst\2018_myd\output_hourly_data"
    output_dir = r"G:\CNN_SpatialDownscaling\modis_lst\2018_mod\merge"

    merge_nc_folders(input_dir1, input_dir2, output_dir)

# ===== 使用示例 =====
# folder1 = "path/to/folder1"
# folder2 = "path/to/folder2"
# output_folder = "path/to/output"
# merge_nc_folders(folder1, folder2, output_folder)
