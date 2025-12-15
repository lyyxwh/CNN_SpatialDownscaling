import os
import zipfile
import rasterio
from rasterio.merge import merge
import xarray as xr
import numpy as np
import time

def extract_and_merge_img(zip_dir, output_img_path, output_nc_path):
    """
    提取zip文件中的.img文件并合并，输出为一个.img文件和一个.nc文件
    :param zip_dir: 包含zip文件的目录
    :param output_img_path: 合并后的.img文件输出路径
    :param output_nc_path: 合并后的.nc文件输出路径
    """
    img_files = []



    # 遍历目录下所有zip文件
    for fname in os.listdir(zip_dir):
        if fname.lower().endswith('.zip'):
            zip_path = os.path.join(zip_dir, fname)
            print(f"Processing zip file: {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as z:
                # 提取所有.img文件
                img_names = [n for n in z.namelist() if n.lower().endswith('.img')]
                print(f"Found {len(img_names)} .img files in {zip_path}")
                for img_name in img_names:
                    with z.open(img_name) as img_file:
                        # 将.img文件保存到临时目录
                        temp_img_path = os.path.join(zip_dir, os.path.basename(img_name))
                        with open(temp_img_path, 'wb') as f:
                            f.write(img_file.read())
                        img_files.append(temp_img_path)
                    z.close()

    if not img_files:
        print("未找到任何.img文件")
        return

    # 合并所有.img文件
    src_files_to_mosaic = [rasterio.open(fp) for fp in img_files]
    mosaic, out_trans = merge(src_files_to_mosaic)

    # 获取元数据并更新
    out_meta = src_files_to_mosaic[0].meta.copy()
    # 检查 Mosaic 数据
    print("Mosaic shape:", mosaic.shape)
    if mosaic.size == 0:
        print("Mosaic 数据为空，无法写入文件")
        return

    # 更新元数据
    out_meta.update({
        "driver": "HFA",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": mosaic.shape[0],  # 确保波段数正确
        "transform": out_trans
    })
    
    # 保存合并后的.img文件
    print("mosaic.shape:", mosaic.shape)
    # 如果是二维（单波段），扩展为三维
    if len(mosaic.shape) == 2:
        mosaic = mosaic[np.newaxis, ...]
    # 检查波段数
    if mosaic.shape[0] == 0:
        print("没有可用波段，终止写入")
        return
    print("out_meta['count']:", mosaic.shape[0])
    out_meta.update({
        "driver": "HFA",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": mosaic.shape[0],  # 必须等于波段数
        "transform": out_trans
    })
    with rasterio.open(output_img_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # 转换为.nc文件并保存
    data_array = xr.DataArray(
        mosaic,
        dims=["band", "y", "x"],
        coords={
            "band": np.arange(1, mosaic.shape[0] + 1),
            "y": np.linspace(out_trans[5], out_trans[5] + out_trans[4] * mosaic.shape[1], mosaic.shape[1]),
            "x": np.linspace(out_trans[2], out_trans[2] + out_trans[0] * mosaic.shape[2], mosaic.shape[2]),
        },
        attrs={
            "crs": src_files_to_mosaic[0].crs.to_string(),
            "transform": out_trans
        }
    )
    data_array.to_netcdf(output_nc_path)

    # 清理临时文件
    '''   for fp in img_files:
        try:          
            os.remove(fp)
        except PermissionError:
            print(f"PermissionError: Unable to delete {fp}. Retrying after 5 seconds.")
            time.sleep(5)
            os.remove(fp)
    '''
    print(f"合并完成，输出文件：\nIMG: {output_img_path}\nNC: {output_nc_path}")

if __name__ == "__main__":
    zip_dir = r"G:\\CNN_SpatialDownscaling\\SRTM_DEM"  # 包含zip文件的目录
    output_img_path = r"G:\\CNN_SpatialDownscaling\\SRTM_DEM\\SRTM_DEM.img"  # 合并后的.img文件路径
    output_nc_path = r"G:\\CNN_SpatialDownscaling\\SRTM_DEM\\SRTM_DEM.nc"  # 合并后的.nc文件路径

    extract_and_merge_img(zip_dir, output_img_path, output_nc_path)
