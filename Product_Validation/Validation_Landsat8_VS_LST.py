import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import glob
import re
import csv

# 忽略 rasterio 内部可能产生的UserWarning，例如 CRS 标识符差异导致的警告
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")

# -----------------------------------------------------
# 核心验证函数 (validate_lst_data) - 已修改支持指定输出路径
# -----------------------------------------------------
def validate_lst_data(landsat_path, validation_path, output_dir="."):
    """
    使用Landsat LST数据对另一个LST文件进行像元级验证和可视化。
    通过地理坐标系统匹配，解决输入文件空间大小不一致的问题。
    新增 output_dir 参数，用于指定图片保存路径。

    Args:
        landsat_path (str): Landsat 8 LST文件的完整路径。
        validation_path (str): 待验证LST文件的完整路径。
        output_dir (str): 结果图片保存的目录。
    """
    
    # --- 1. 数据加载与地理信息读取 ---
    try:
        with rasterio.open(landsat_path) as src_landsat:
            landsat_lst = src_landsat.read(1)
            landsat_nodata = src_landsat.nodata
            landsat_transform = src_landsat.transform
            landsat_crs = src_landsat.crs
            landsat_shape = src_landsat.shape
        
        # 若验证文件为 netCDF，使用 NETCDF:...:TG 打开子数据集
        val_open_path = validation_path
        if validation_path.lower().endswith('.nc'):
            val_open_path = f"NETCDF:{validation_path}:TG"

        with rasterio.open(val_open_path) as src_val:
            val_nodata = src_val.nodata
            val_transform = src_val.transform
            val_crs = src_val.crs

            # 如果验证数据没有 CRS，猜测为 EPSG:4326（纬度/经度）并给出提示
            if val_crs is None:
                print(f"⚠️ 注意：验证数据无 CRS，假定为 EPSG:4326（请确认）。")
                val_crs = rasterio.crs.CRS.from_epsg(4326)

            # --- 2. 坐标系和分辨率检查 (作为警告) ---
            if landsat_crs != val_crs:
                print(f"❌ 警告：坐标系不匹配！Landsat: {landsat_crs}, 验证数据: {val_crs}")

            # 将验证栅格重投影/重采样到 Landsat 的网格 (transform + shape)，以保证像元一一对应
            dst_shape = landsat_shape
            dst_transform = landsat_transform
            dst_crs = landsat_crs

            # 准备目标数组
            dst_dtype = src_val.dtypes[0] if src_val.dtypes else 'float32'
            val_lst_cropped = np.empty(dst_shape, dtype=dst_dtype)

            # 执行重投影
            try:
                reproject(
                    source=rasterio.band(src_val, 1),
                    destination=val_lst_cropped,
                    src_transform=src_val.transform,
                    src_crs=val_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )
            except Exception as e:
                print(f"❌ 错误：在重投影验证数据时失败：{e}")
                return None
            
    except rasterio.RasterioIOError as e:
        print(f"❌ 错误：无法打开或读取文件。错误信息: {e}")
        return None
    except Exception as e:
        print(f"❌ 发生未知错误：{e}")
        return None
    
    # --- 4. 最终形状检查 ---
    if landsat_shape != val_lst_cropped.shape:
        print(f"❌ 错误：地理匹配后数组形状仍不匹配！")
        print(f"Landsat 原始形状: {landsat_shape}")
        print(f"验证数据裁剪形状: {val_lst_cropped.shape}")
        return None

    T_A_raw = landsat_lst
    T_B_raw = val_lst_cropped

    # --- 5. 数据清洗与提取有效像元 ---
    mask_A = np.isfinite(T_A_raw)
    mask_B = np.isfinite(T_B_raw)

    # 排除 Landsat NoData/0
    if landsat_nodata is not None:
        mask_A = mask_A & (T_A_raw != landsat_nodata)
    mask_A = mask_A & (T_A_raw != 0)
    
    # 排除 验证数据 NoData/0
    if val_nodata is not None:
        mask_B = mask_B & (T_B_raw != val_nodata)
    mask_B = mask_B & (T_B_raw != 0)
    
    valid_mask = mask_A & mask_B
    
    T_A = T_A_raw[valid_mask].flatten() 
    T_B = T_B_raw[valid_mask].flatten()
    
    N = len(T_A)
    if N == 0:
        print("⚠️ 警告：没有找到重叠的有效像元值进行比较。")
        return {"N": 0}

    print(f"✅ 成功提取 {N} 个有效像元进行验证（已通过地理坐标匹配）。")

    # --- 6. 统计指标计算 ---
    diff = T_B - T_A
    rmse = np.sqrt(np.mean(diff**2))
    mb = np.mean(diff)
    
    if N > 1:
        R = np.corrcoef(T_A, T_B)[0, 1]
        R2 = R**2
    else:
        R, R2 = np.nan, np.nan
        
    results = {
        "N": N, "RMSE": rmse, "MB": mb, "R": R, "R2": R2,
        "T_A_mean": np.mean(T_A), "T_B_mean": np.mean(T_B)
    }

    # --- 7. 结果可视化与保存 ---
    
    landsat_name = os.path.basename(landsat_path)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    
    # ... (绘图代码保持不变，为节省篇幅此处省略大部分，但功能完整)
    plt.subplot(1, 2, 1) # 散点图
    plt.scatter(T_A, T_B, s=1, alpha=0.5)
    min_val = min(T_A.min(), T_B.min())
    max_val = max(T_A.max(), T_B.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    plt.title('LST Scatter Plot (Geographically Matched)')
    plt.xlabel(f'Landsat LST (Reference)')
    plt.ylabel(f'Validation LST (Cropped)')
    plt.text(0.05, 0.95, 
             f'N: {N}\nRMSE: {rmse:.2f}\nMB: {mb:.2f}\nR²: {R2:.2f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
    plt.legend()
    plt.axis('equal')

    plt.subplot(1, 2, 2) # 差异图
    diff_map = np.full(landsat_shape, np.nan)
    diff_map[valid_mask] = diff
    max_abs_diff = np.nanpercentile(np.abs(diff), 98)
    vmin, vmax = -max_abs_diff, max_abs_diff
    im = plt.imshow(diff_map, cmap='RdBu', vmin=vmin, vmax=vmax)
    plt.title('Spatial Difference Map (Validation LST - Landsat LST)')
    plt.colorbar(im, label='Temperature Difference')
    plt.axis('off')

    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构造完整输出文件路径
    output_filename = f"Validation_Results_Matched_{os.path.splitext(landsat_name)[0]}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"🖼️ 图片已保存至：{output_path}")

    return results

# -----------------------------------------------------
# 主运行逻辑 (main) - 增加图片输出路径变量
# -----------------------------------------------------
def main():
    """
    主函数：配置并运行单个文件的验证。
    """
    print("--- 遥感 LST 验证工具启动：批量模式 ---")

    # =======================================================
    # >>> 批处理设置：修改以下 Landsat 文件夹 与 验证文件夹 <<<
    # =======================================================
    landsat_dir = r'D:\lyygi\Downloads\drive-download-20251117T064059Z-1-001\output_lst'
    validation_dir = r'G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\output\interpolation_v8'#G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\output\interpolation_v8   G:\cldas\data\2018_clip_filtered'
    image_output_directory = r'G:\CNN_SpatialDownscaling\scripts\Product_Validation\output\Validation2'
    summary_csv = os.path.join(image_output_directory, 'validation_summary_0p0083333.csv')
    # =======================================================

    os.makedirs(image_output_directory, exist_ok=True)

    # Helper: 从 Landsat 文件名提取日期和小时（返回 YYYYMMDD, HH）
    def parse_landsat_datehour(fname):
        b = os.path.basename(fname)
        # 常见模式：YYYYMMDD_HHMMSS 或 YYYYMMDDHHMMSS
        m = re.search(r'(\d{8})[_-]?(\d{6})', b)
        if m:
            date = m.group(1)
            hour = m.group(2)[:2]
            return date, hour
        # 备用：只查找 8 位日期和单独小时
        m2 = re.search(r'(\d{8})[_-]?(\d{2})', b)
        if m2:
            return m2.group(1), m2.group(2)
        return None, None

    landsat_files = sorted(glob.glob(os.path.join(landsat_dir, '*.tif')))
    if len(landsat_files) == 0:
        print(f"❌ 错误：未在目录找到任何 .tif 文件: {landsat_dir}")
        return

    summary_rows = []

    for lfp in landsat_files:
        date, hour = parse_landsat_datehour(lfp)
        if date is None:
            print(f"⚠️ 跳过：无法从文件名解析时间 -> {lfp}")
            continue

        # 构造可能的验证文件名（优先精确匹配），例如 20181006_03_LST.tif 或 CLDAS_20181006_03.nc
        candidate_names = [
            f"{date}_{hour}_LST.tif",
            f"{date}_{int(hour)}_LST.tif",
            f"CLDAS_{date}_{hour}.nc",
            f"CLDAS_{date}_{int(hour)}.nc",
            f"CLDAS_{date}_{hour}.tif",
            f"CLDAS{date}_{int(hour)}.tif",
            f"LST_{date}_{hour}_UTC.tif",
            f"LST_{date}_{int(hour)}_UTC.tif"
        ]
        matched_val = None
        for cname in candidate_names:
            vpath = os.path.join(validation_dir, cname)
            if os.path.isfile(vpath):
                matched_val = vpath
                break

        # 退而求其次：模糊匹配包含日期和小时的任何文件（例如不同命名规则或扩展名）
        if matched_val is None:
            pattern1 = os.path.join(validation_dir, f"*{date}*_{hour}*")
            pattern2 = os.path.join(validation_dir, f"*{date}*{hour}*")
            candidates = glob.glob(pattern1) + glob.glob(pattern2)
            if candidates:
                matched_val = candidates[0]
            else:
                print(f"⚠️ 未找到匹配验证文件: {date}_{hour}  对于 {os.path.basename(lfp)}")
                continue
        
        print(f"\n--- 处理: {os.path.basename(lfp)} \n 对应验证文件: {os.path.basename(matched_val)}")

        res = validate_lst_data(os.path.abspath(lfp), os.path.abspath(matched_val), output_dir=image_output_directory)
        row = {
            'landsat_file': os.path.basename(lfp),
            'validation_file': os.path.basename(matched_val),
            'N': None, 'RMSE': None, 'MB': None, 'R': None, 'R2': None, 'T_A_mean': None, 'T_B_mean': None
        }
        if res:
            row.update({k: res.get(k) for k in ['N','RMSE','MB','R','R2','T_A_mean','T_B_mean']})
        summary_rows.append(row)

    # 保存汇总 CSV
    if summary_rows:
        with open(summary_csv, 'w', newline='', encoding='utf-8') as cf:
            fieldnames = ['landsat_file','validation_file','N','RMSE','MB','R','R2','T_A_mean','T_B_mean']
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        print(f"\n🗂️ 汇总结果已保存: {summary_csv}")
    else:
        print("\n⚠️ 未生成任何验证结果，检查匹配规则与输入文件。")


if __name__ == "__main__":
    main()