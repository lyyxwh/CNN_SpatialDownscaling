import os
import zipfile
import tempfile
import xarray as xr

def split_era5_zip(zip_dir, out_dir):
    """
    遍历zip_dir下所有zip文件，解压后按小时分割nc文件并输出到out_dir
    输出文件名格式：ERA5_yyyy_mm_dd_hh.nc
    """
    os.makedirs(out_dir, exist_ok=True)
    print(os.listdir(zip_dir))  # 打印zip_dir目录下的文件列表
    for fname in os.listdir(zip_dir):
        if fname.lower().endswith('.zip'):
            zip_path = os.path.join(zip_dir, fname)
            with zipfile.ZipFile(zip_path, 'r') as z:
                nc_names = [n for n in z.namelist() if n.lower().endswith('.nc')]
                if not nc_names:
                    print(f"{zip_path} 未找到nc文件")
                    continue
                nc_name = nc_names[0]
                # 修复：确保解压路径目录结构正确
                nc_dir = os.path.dirname(nc_name)
                with tempfile.TemporaryDirectory() as tmpdir:
                    if nc_dir:
                        os.makedirs(os.path.join(tmpdir, nc_dir), exist_ok=True)
                    z.extract(nc_name, tmpdir)
                    nc_path = os.path.join(tmpdir, nc_name)
                    ds = xr.open_dataset(nc_path, decode_cf=True)
                    print(f"{nc_path} 变量信息: {list(ds.data_vars)}")
                    # 自动检测时间维度
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
                        print(f"{nc_path} 未找到时间维度")
                        ds.close()
                        continue
                    print(f"有效时间范围: {ds[time_dim].values[0]} - {ds[time_dim].values[-1]}")
                    for idx in range(len(ds[time_dim])):
                        ds_slice = ds.isel({time_dim: idx})
                        ds_slice.attrs = ds.attrs
                        ds_slice[time_dim].attrs = ds[time_dim].attrs
                        print(f"处理时间索引: {idx}, 有效时间: {ds_slice[time_dim].values}")
                        t = ds_slice[time_dim].values
                        print(f"时间分量: {t}")
                        try:
                            year = int(str(t)[0:4])
                            month = int(str(t)[5:7])
                            day = int(str(t)[8:10])
                            hour = int(str(t)[11:13])
                        except Exception:
                            dt = ds_slice[time_dim].dt
                            year = int(dt.year.values)
                            month = int(dt.month.values)
                            day = int(dt.day.values)
                            hour = int(dt.hour.values)
                        out_name = f"ERA5_{year:04d}_{month:02d}_{day:02d}_{hour:02d}.nc"
                        out_path = os.path.join(out_dir, out_name)
                        ds_out = xr.Dataset({v: ds_slice[v] for v in ds_slice.data_vars}, coords={time_dim: ds_slice[time_dim]})
                        ds_out.to_netcdf(
                            out_path,
                            format="NETCDF4_CLASSIC",
                            encoding={v: {"zlib": True} for v in ds_out.data_vars},
                            engine="netcdf4"
                        )
                        ds_out.close()
                    ds.close()

if __name__ == "__main__":
    zip_dir = r"G:\\CNN_SpatialDownscaling\\ERA5\\2018_rawdata"  # zip文件所在目录
    out_dir = r"G:\\CNN_SpatialDownscaling\\ERA5\\2018_0"  # 输出目录
    split_era5_zip(zip_dir, out_dir)