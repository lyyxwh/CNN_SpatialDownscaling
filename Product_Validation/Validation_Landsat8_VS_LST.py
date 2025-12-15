import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

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
        
        with rasterio.open(validation_path) as src_val:
            val_nodata = src_val.nodata
            val_transform = src_val.transform
            val_crs = src_val.crs
            
            # --- 2. 坐标系和分辨率检查 (作为警告) ---
            if landsat_crs != val_crs:
                print(f"❌ 警告：坐标系不匹配！Landsat: {landsat_crs}, 验证数据: {val_crs}")
            
            res_match = np.isclose(landsat_transform[0], val_transform[0]) and \
                        np.isclose(landsat_transform[4], val_transform[4])
            if not res_match:
                 print("❌ 警告：分辨率似乎不匹配，可能会导致像元对齐错误！")

            # --- 3. 使用 Landsat 的地理边界在验证数据上读取窗口 ---
            
            window = src_val.window(*src_landsat.bounds)
            val_lst_cropped = src_val.read(1, window=window)
            
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
    print(f"--- 遥感 LST 验证工具启动：单文件调试模式 ---")

    # =======================================================
    # >>> 调试设置：请在这里修改您的文件路径 <<<
    # =======================================================
    
    # 示例 Landsat 文件路径
    landsat_file = r'D:\lyygi\Downloads\drive-download-20251117T064059Z-1-001\output_lst1\Landsat8_LST_20181006_031126_UTC.tif'
    # 示例待验证文件路径
    validation_file = r'G:\CNN_SpatialDownscaling\20181006_03_LST.tif' 
    
    # >>> 新增：指定图片输出目录 <<<
    # 例如，保存到 D 盘的 "Validation_Output" 文件夹
    image_output_directory = r'G:\CNN_SpatialDownscaling\scripts\Product_Validation\output\Validation2'
    
    # =======================================================
    # >>> 调试设置结束 <<<
    # =======================================================
    
    l_path = os.path.abspath(landsat_file)
    v_path = os.path.abspath(validation_file)
    
    if not os.path.isfile(l_path):
        print(f"❌ 错误：Landsat 文件不存在: {l_path}")
        return
        
    if not os.path.isfile(v_path):
        print(f"❌ 错误：验证文件不存在: {v_path}")
        return

    print("\n--- 模式：单文件验证 ---")
    print(f"Landsat 文件: {l_path}")
    print(f"验证文件:   {v_path}")
    print(f"图片保存目录: {os.path.abspath(image_output_directory)}")
    
    # 执行单个文件验证，并传递输出目录
    results = validate_lst_data(l_path, v_path, output_dir=image_output_directory)
    
    if results and results.get("N", 0) > 0:
        print("\n--- 验证结果 ---")
        print(f"RMSE (均方根误差): {results['RMSE']:.4f}")
        print(f"MB (平均偏差):      {results['MB']:.4f}")
        print(f"R² (决定系数):      {results['R2']:.4f}")
        print(f"有效像元数 (N):   {results['N']}")
    elif results:
        print("\n❌ 验证失败：无有效像元进行计算。")


if __name__ == "__main__":
    main()