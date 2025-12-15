import os
import zipfile
import tempfile
import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

def calculate_es_ea_rh_vpd(t2m_kelvin, d2m_kelvin):
    """
    计算饱和水汽压 (es), 实际水汽压 (ea), 相对湿度 (rh) 和 水汽压差 (vpd)。
    t2m_kelvin: 2米温度 (开尔文)
    d2m_kelvin: 2米露点温度 (开尔文)
    """
    t2m_celsius = t2m_kelvin - 273.15
    d2m_celsius = d2m_kelvin - 273.15

    # 计算饱和水汽压 (es) 和 实际水汽压 (ea) (单位: kPa)
    es = 0.61094 * np.exp((17.625 * t2m_celsius) / (t2m_celsius + 243.04))
    ea = 0.61094 * np.exp((17.625 * d2m_celsius) / (d2m_celsius + 243.04))

    rh = (ea / es) * 100
    rh = rh.clip(min=0, max=100)

    vpd = es - ea
    vpd = vpd.clip(min=0)

    return es, ea, rh, vpd


def process_single_zip_file(zip_path, out_dir):
    """
    处理单个ERA5 zip文件：解压、按小时分割NC文件、计算VPD和相对湿度。
    这是一个独立的函数，适合并行处理。
    """
    logging.info(f"开始处理ZIP文件: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        nc_names = [n for n in z.namelist() if n.lower().endswith('.nc')]
        if not nc_names:
            logging.warning(f"{zip_path} 未找到nc文件，跳过。")
            return f"Skipped: {zip_path} (no .nc file)"

        nc_name = nc_names[0]
        nc_dir_in_zip = os.path.dirname(nc_name) # 获取zip内部的路径结构

        with tempfile.TemporaryDirectory() as tmpdir:
            # 确保解压路径保留内部目录结构
            extract_path = os.path.join(tmpdir, nc_dir_in_zip) if nc_dir_in_zip else tmpdir
            os.makedirs(extract_path, exist_ok=True) # 确保临时目录结构存在

            logging.info(f"正在从 {zip_path} 中解压 {nc_name} 到临时目录: {tmpdir}")
            z.extract(nc_name, tmpdir)
            nc_path = os.path.join(tmpdir, nc_name) # 完整的临时文件路径

            try:
                ds = xr.open_dataset(nc_path, decode_cf=True)
                logging.info(f"已成功打开NetCDF文件: {nc_path}")
                logging.info(f"文件包含变量: {list(ds.data_vars)}")

                required_vars = ['t2m', 'd2m']
                if not all(v in ds.data_vars for v in required_vars):
                    logging.error(f"错误: 文件 {nc_path} 缺少变量 {required_vars} 中的一个或多个，跳过。")
                    ds.close()
                    return f"Failed: {zip_path} (missing variables)"

                time_dim = None
                for dim in ds.dims:
                    if "time" in dim or "valid_time" in dim:
                        time_dim = dim
                        break
                if time_dim is None:
                    for coord in ds.coords:
                        if "time" in coord or "valid_time" in coord:
                            time_dim = coord
                            break
                
                if time_dim is None:
                    logging.error(f"错误: 文件 {nc_path} 未找到时间维度，跳过。")
                    ds.close()
                    return f"Failed: {zip_path} (no time dimension)"
                
                logging.info(f"检测到时间维度: {time_dim}")

                for idx in range(len(ds[time_dim])):
                    ds_slice = ds.isel({time_dim: idx})
                    
                    t2m = ds_slice['t2m']
                    d2m = ds_slice['d2m']
                    

                    es, ea, rh, vpd = calculate_es_ea_rh_vpd(t2m, d2m)

                    ds_slice['rh'] = rh
                    ds_slice['vpd'] = vpd


                    ds_slice['rh'].attrs = {
                        'long_name': 'Relative humidity',
                        'units': '%',
                        'description': 'Calculated from es and ea'
                    }
                    ds_slice['vpd'].attrs = {
                        'long_name': 'Vapor pressure deficit',
                        'units': 'kPa',
                        'description': 'Calculated from es and ea'
                    }

                    ds_slice.attrs = ds.attrs
                    if time_dim in ds_slice.coords:
                        ds_slice[time_dim].attrs = ds[time_dim].attrs

                    t = ds_slice[time_dim].values
                    
                    try:
                        dt_obj = np.datetime_as_string(t, unit='s')
                        year = int(dt_obj[0:4])
                        month = int(dt_obj[5:7])
                        day = int(dt_obj[8:10])
                        hour = int(dt_obj[11:13])
                    except Exception:
                        dt = ds_slice[time_dim].dt
                        year = int(dt.year.values)
                        month = int(dt.month.values)
                        day = int(dt.day.values)
                        hour = int(dt.hour.values)

                    out_name = f"ERA5_{year:04d}_{month:02d}_{day:02d}_{hour:02d}.nc"
                    out_path = os.path.join(out_dir, out_name)
                    
                    ds_out = xr.Dataset({v: ds_slice[v] for v in ds_slice.data_vars}, coords=ds_slice.coords)
                    ds_out.attrs = ds_slice.attrs

                    # 确保保存路径的父目录存在
                    os.makedirs(os.path.dirname(out_path), exist_ok=True) 

                    ds_out.to_netcdf(
                        out_path,
                        format="NETCDF4_CLASSIC",
                        encoding={v: {"zlib": True, "complevel": 5} for v in ds_out.data_vars if ds_out[v].dtype in ['float32', 'float64']},
                        engine="netcdf4"
                    )
                    ds_out.close()
                ds.close()
                logging.info(f"完成处理ZIP文件: {zip_path}")
                return f"Success: {zip_path}"
            except Exception as e:
                logging.error(f"处理文件 {nc_path} 时发生错误: {e}", exc_info=True)
                if 'ds' in locals() and ds is not None:
                    ds.close()
                return f"Failed: {zip_path} ({e})"



def split_era5_zip_parallel(zip_dir, out_dir, max_workers=None):
    """
    遍历zip_dir下所有zip文件，解压后按小时分割nc文件，计算VPD和相对湿度，
    并输出到out_dir，采用并行处理。
    """
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"正在处理目录: {zip_dir}")
    logging.info(f"输出目录: {out_dir}")

    zip_files = [os.path.join(zip_dir, f) for f in os.listdir(zip_dir) if f.lower().endswith('.zip')]
    if not zip_files:
        logging.warning(f"目录 {zip_dir} 中未找到任何ZIP文件。")
        return

    # 如果max_workers未指定，则默认为CPU核心数
    if max_workers is None:
        max_workers = os.cpu_count() if os.cpu_count() else 1
    logging.info(f"将使用 {max_workers} 个进程进行并行处理。")

    # 使用ProcessPoolExecutor进行并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_zip_file, zip_file, out_dir): zip_file for zip_file in zip_files}
        
        # 实时收集结果
        for future in as_completed(futures):
            zip_file = futures[future]
            try:
                result = future.result()
                logging.info(f"处理结果: {result}")
            except Exception as exc:
                logging.error(f"文件 {zip_file} 生成异常: {exc}")
    
    logging.info("\n所有ZIP文件处理完毕！")



if __name__ == "__main__":
    zip_dir = r"G:\\CNN_SpatialDownscaling\\201712and201901\\era5"  # ERA5 zip文件所在目录
    out_dir = r"G:\\CNN_SpatialDownscaling\\201712and201901\\era5_processed_parallel"  # 输出目录

    # 可以根据你的CPU核心数调整max_workers，None表示使用所有核心
    # 注意：过多的进程可能会导致内存不足，尤其是在处理大型文件时
    split_era5_zip_parallel(zip_dir, out_dir, max_workers=None)