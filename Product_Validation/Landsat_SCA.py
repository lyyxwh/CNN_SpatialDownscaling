import tarfile
import os
import glob
import numpy as np
from pylandtemp import single_channel
from pylandtemp.emissivity import ndvi_emissivity # 用于LSE计算

def lst_inversion_sca_pylandtemp(tgz_filepath, wvc_value):
    """
    使用 pylandtemp 库的单通道算法 (SCA) 反演 Landsat LST。

    参数:
        tgz_filepath (str): Landsat Level 1 产品的 .tgz 文件路径。
        wvc_value (float): 对应图像获取时间的大气水汽含量 (Water Vapor Content, WVC)，
                           单位为 g/cm² 或 kg/m²。

    返回:
        numpy.ndarray: 反演的地表温度 (LST) 栅格数组，单位为开尔文 (K)。
    """
    
    # 1. 解压缩文件
    print(f"1. 正在解压文件: {tgz_filepath}...")
    output_dir = tgz_filepath.replace('.tgz', '_extracted')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        with tarfile.open(tgz_filepath, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print(f"解压完成，文件位于: {output_dir}")
    else:
        print(f"解压目录已存在，跳过解压: {output_dir}")

    # 2. 定位元数据文件 (.MTL.txt)
    # 查找以 _MTL.txt 结尾的文件
    mtl_files = glob.glob(os.path.join(output_dir, '*MTL.txt'))
    
    if not mtl_files:
        raise FileNotFoundError("在解压后的目录中找不到 Landsat MTL 元数据文件。")
    
    mtl_path = mtl_files[0]
    print(f"2. 找到 MTL 文件: {mtl_path}")

    # 3. 使用 pylandtemp 计算 LST (SCA)
    print("3. 正在计算地表比辐射率 (LSE) 和 LST (SCA)...")
    
    # --- LSE 计算 (必需步骤) ---
    # pylandtemp 默认使用基于 NDVI 的方法 (Sobrino et al., 2008) 计算 LSE
    # LSE是SCA算法的输入参数之一。
    lse_result = ndvi_emissivity(
        mtl_path=mtl_path, 
        lse_method='sobrino_08', # 经典方法之一
        # 如果是Landsat 8/9，会自动使用B4(Red)和B5(NIR)
        # 如果是Landsat 5/7，则使用B3(Red)和B4(NIR)
    )

    # --- LST 计算 (SCA) ---
    # Landsat 8/9 默认使用 B10 (热红外)
    # Landsat 5/7 默认使用 B6 (热红外)
    lst_result = single_channel(
        mtl_path=mtl_path,
        wvc=wvc_value, # 关键参数：大气水汽含量
        lse=lse_result.lse, # 使用上一步计算得到的 LSE 数组
        temp_method='jimenez-munoz_09', # 经典的 SCA 算法，适用于 Landsat 8/9
    )

    print("4. LST 反演完成。")
    print(f"LST 结果单位为: {lst_result.units}")

    # LST 结果是一个 LandTempResult 对象，包含 LST 数组和地理信息
    return lst_result.lst

# --- 示例调用 ---

# 请将此路径替换为您的 Landsat .tgz 文件路径
landsat_tgz_file = r"D:\lyygi\Downloads\LC81230312018114LGN00.tgz"
# 替换为实际文件名

# 请将此值替换为您的实际大气水汽含量！
# 示例：0.5 g/cm²（约等于 5.0 kg/m²），这是一个中等水汽条件的值
# 必须根据您的影像采集时间和地理位置查找。
water_vapor_content = 0.5 

try:
    lst_array = lst_inversion_sca_pylandtemp(landsat_tgz_file, water_vapor_content)
    
    # 打印结果的统计信息进行验证
    print("\n--- 结果统计 ---")
    print(f"LST 数组形状: {lst_array.shape}")
    print(f"LST 最小值 (K): {np.nanmin(lst_array):.2f}")
    print(f"LST 最大值 (K): {np.nanmax(lst_array):.2f}")
    print(f"LST 平均值 (K): {np.nanmean(lst_array):.2f}")

    # 进一步可以将 LST 数组写入 GeoTIFF 文件进行可视化
    # (需要使用 rasterio/gdal 来保存 lst_result.lst 数组和其地理参考信息)
    
except FileNotFoundError as e:
    print(f"\n错误：{e}")
    print("请检查您的 .tgz 文件路径是否正确，并确保文件存在。")
except Exception as e:
    print(f"\n处理过程中发生错误: {e}")
    print("请检查 Landsat L1 产品是否完整，或 WVC 参数是否合理。")