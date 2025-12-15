import os
import warnings
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial, lru_cache
import gc
from numba import jit, prange
import psutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用警告
warnings.filterwarnings('ignore')

@jit(nopython=True, parallel=True, cache=True)
def fast_bilinear_resample(data, x_indices, y_indices, x_weights, y_weights, output_shape):
    """使用numba优化的双线性插值"""
    output = np.full(output_shape, np.nan, dtype=np.float32)
    
    for i in prange(output_shape[0]):
        for j in prange(output_shape[1]):
            yi = y_indices[i, j]
            xi = x_indices[i, j]
            
            if yi >= 0 and xi >= 0 and yi < data.shape[0]-1 and xi < data.shape[1]-1:
                yw = y_weights[i, j]
                xw = x_weights[i, j]
                
                # 双线性插值
                v00 = data[yi, xi]
                v01 = data[yi, xi+1]
                v10 = data[yi+1, xi]
                v11 = data[yi+1, xi+1]
                
                if not (np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11)):
                    output[i, j] = (v00 * (1-xw) * (1-yw) + 
                                   v01 * xw * (1-yw) + 
                                   v10 * (1-xw) * yw + 
                                   v11 * xw * yw)
    
    return output

@jit(nopython=True, parallel=True, cache=True)
def fast_nearest_resample(data, x_indices, y_indices, output_shape):
    """使用numba优化的最近邻插值"""
    output = np.full(output_shape, np.nan, dtype=np.float32)
    
    for i in prange(output_shape[0]):
        for j in prange(output_shape[1]):
            yi = int(np.round(y_indices[i, j]))
            xi = int(np.round(x_indices[i, j]))
            
            if 0 <= yi < data.shape[0] and 0 <= xi < data.shape[1]:
                output[i, j] = data[yi, xi]
    
    return output

class UltraFastNCResampler:
    def __init__(self, target_resolution, clip_shapefile, output_dir, n_workers=None, memory_limit_gb=32):
        """
        极速NC文件重采样和裁剪处理器
        
        Parameters:
        -----------
        target_resolution : str
            目标分辨率 (如 "1000m")
        clip_shapefile : str
            裁剪用的shapefile路径
        output_dir : str
            输出目录路径
        n_workers : int
            并行进程数
        memory_limit_gb : float
            内存限制(GB)
        """
        self.target_resolution = target_resolution
        self.clip_shapefile = clip_shapefile
        self.output_dir = output_dir
        self.target_crs = "EPSG:4326"
        self.memory_limit = memory_limit_gb * 1024**3  # 转换为字节
        
        # 智能设置并行进程数
        available_memory = psutil.virtual_memory().available
        cpu_count = mp.cpu_count()
        
        if n_workers is None:
            # 根据内存和CPU数量智能设置
            memory_per_worker = self.memory_limit
            max_workers_by_memory = max(1, int(available_memory * 0.8 / memory_per_worker))
            self.n_workers = min(cpu_count - 1, max_workers_by_memory, 8)  # 最多8个进程
        else:
            self.n_workers = n_workers
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 解析分辨率
        self.resolution_meters = self._parse_resolution(target_resolution)
        
        # 预加载和缓存关键数据
        self._setup_grid_cache()
        
        logger.info(f"初始化完成 - 进程数: {self.n_workers}, 内存限制: {memory_limit_gb}GB")
        logger.info(f"目标分辨率: {self.resolution_meters}m")
    
    @lru_cache(maxsize=1)
    def _parse_resolution(self, resolution_str):
        """解析分辨率字符串"""
        import re
        numbers = re.findall(r'\d+', resolution_str)
        if numbers:
            return int(numbers[0])
        return 1000
    
    @lru_cache(maxsize=1)
    def _load_clip_boundary(self):
        """加载裁剪边界（缓存）"""
        gdf = gpd.read_file(self.clip_shapefile)
        if gdf.crs != self.target_crs:
            gdf = gdf.to_crs(self.target_crs)
        return gdf
    
    def _setup_grid_cache(self):
        """预设置网格缓存"""
        self.clip_gdf = self._load_clip_boundary()
        self.clip_bounds = self.clip_gdf.total_bounds
        
        # 计算目标分辨率（度）
        center_lat = (self.clip_bounds[1] + self.clip_bounds[3]) / 2
        self.x_res_deg = self.resolution_meters / (111320.0 * np.cos(np.radians(center_lat)))
        self.y_res_deg = self.resolution_meters / 111320.0
        
        # 计算目标网格
        bounds = self.clip_bounds
        buffer = max(self.x_res_deg, self.y_res_deg) * 2  # 添加缓冲区
        
        min_x = np.floor((bounds[0] - buffer) / self.x_res_deg) * self.x_res_deg
        max_x = np.ceil((bounds[2] + buffer) / self.x_res_deg) * self.x_res_deg
        min_y = np.floor((bounds[1] - buffer) / self.y_res_deg) * self.y_res_deg
        max_y = np.ceil((bounds[3] + buffer) / self.y_res_deg) * self.y_res_deg
        
        self.target_x_coords = np.arange(min_x, max_x + self.x_res_deg/2, self.x_res_deg, dtype=np.float64)
        self.target_y_coords = np.arange(max_y, min_y - self.y_res_deg/2, -self.y_res_deg, dtype=np.float64)
        
        self.target_shape = (len(self.target_y_coords), len(self.target_x_coords))
        self.target_transform = from_bounds(min_x, min_y, max_x, max_y, 
                                          len(self.target_x_coords), len(self.target_y_coords))
        
        # 预计算裁剪掩膜
        self.clip_mask = rasterize(
            [(geom, 1) for geom in self.clip_gdf.geometry],
            out_shape=self.target_shape,
            transform=self.target_transform,
            fill=0,
            dtype=np.uint8
        ).astype(bool)
        
        logger.info(f"目标网格: {self.target_shape}, 分辨率(度): {self.x_res_deg:.6f}x{self.y_res_deg:.6f}")
    
    def _ultra_fast_resample(self, da, method='bilinear'):
        """极速重采样函数"""
        try:
            # 确保数据在正确的坐标系
            if da.rio.crs != self.target_crs:
                da = da.rio.reproject(self.target_crs, resampling=Resampling.bilinear)
            
            # 获取原始坐标
            src_x = da.coords[da.rio.x_dim].values.astype(np.float64)
            src_y = da.coords[da.rio.y_dim].values.astype(np.float64)
            
            # 确保坐标单调
            if src_x[0] > src_x[-1]:
                src_x = src_x[::-1]
                flip_x = True
            else:
                flip_x = False
                
            if src_y[0] < src_y[-1]:
                src_y = src_y[::-1]
                flip_y = True
            else:
                flip_y = False
            
            # 预计算索引和权重
            target_xx, target_yy = np.meshgrid(self.target_x_coords, self.target_y_coords)
            
            # 计算索引
            x_indices = np.interp(target_xx, src_x, np.arange(len(src_x))).astype(np.float32)
            y_indices = np.interp(target_yy, src_y, np.arange(len(src_y))).astype(np.float32)
            
            # 获取数据
            data = da.values.astype(np.float32)
            
            # 处理坐标翻转
            if flip_x or flip_y:
                if flip_x and flip_y:
                    data = np.flip(np.flip(data, axis=-1), axis=-2)
                elif flip_x:
                    data = np.flip(data, axis=-1)
                elif flip_y:
                    data = np.flip(data, axis=-2)
            
            # 执行重采样
            if data.ndim == 2:
                if method == 'bilinear':
                    x_weights = x_indices - np.floor(x_indices)
                    y_weights = y_indices - np.floor(y_indices)
                    x_indices = np.floor(x_indices).astype(np.int32)
                    y_indices = np.floor(y_indices).astype(np.int32)
                    
                    resampled_data = fast_bilinear_resample(
                        data, x_indices, y_indices, x_weights, y_weights, self.target_shape
                    )
                else:  # nearest
                    x_indices = x_indices.astype(np.float32)
                    y_indices = y_indices.astype(np.float32)
                    resampled_data = fast_nearest_resample(data, x_indices, y_indices, self.target_shape)
            else:
                # 多维数据
                resampled_data = np.full((data.shape[0],) + self.target_shape, np.nan, dtype=np.float32)
                
                if method == 'bilinear':
                    x_weights = x_indices - np.floor(x_indices)
                    y_weights = y_indices - np.floor(y_indices)
                    x_indices = np.floor(x_indices).astype(np.int32)
                    y_indices = np.floor(y_indices).astype(np.int32)
                    
                    for i in range(data.shape[0]):
                        resampled_data[i] = fast_bilinear_resample(
                            data[i], x_indices, y_indices, x_weights, y_weights, self.target_shape
                        )
                else:  # nearest
                    x_indices = x_indices.astype(np.float32)
                    y_indices = y_indices.astype(np.float32)
                    for i in range(data.shape[0]):
                        resampled_data[i] = fast_nearest_resample(data[i], x_indices, y_indices, self.target_shape)
            
            # 创建新的DataArray
            new_coords = {}
            for dim, coord in da.coords.items():
                if dim == da.rio.x_dim:
                    new_coords[dim] = (dim, self.target_x_coords)
                elif dim == da.rio.y_dim:
                    new_coords[dim] = (dim, self.target_y_coords)
                else:
                    new_coords[dim] = coord
            
            da_resampled = xr.DataArray(
                resampled_data,
                coords=new_coords,
                dims=da.dims,
                attrs=da.attrs
            )
            
            da_resampled = da_resampled.rio.write_crs(self.target_crs)
            da_resampled = da_resampled.rio.write_transform(self.target_transform)
            
            return da_resampled
            
        except Exception as e:
            logger.error(f"极速重采样失败: {e}")
            raise
    
    def _ultra_fast_clip(self, da):
        """极速裁剪函数"""
        try:
            # 直接应用预计算的掩膜
            if da.ndim == 2:
                da.values[~self.clip_mask] = np.nan
            else:
                for i in range(da.shape[0]):
                    da.values[i][~self.clip_mask] = np.nan
            
            return da
            
        except Exception as e:
            logger.error(f"极速裁剪失败: {e}")
            raise
    
    def process_single_file_ultra_fast(self, nc_file, output_filename=None, resample_method='bilinear', skip_existing=True):
        """极速单文件处理"""
        start_time = time.time()
        
        try:
            # 确定输出路径
            if output_filename is None:
                base_name = os.path.splitext(os.path.basename(nc_file))[0]
                output_filename = f"{base_name}_1000m.nc"
            
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 跳过已存在文件
            if skip_existing and os.path.exists(output_path):
                logger.info(f"跳过: {os.path.basename(output_path)}")
                return output_path
            
            logger.info(f"处理: {os.path.basename(nc_file)}")
            
            # 打开数据集，使用内存映射
            with xr.open_dataset(nc_file, chunks=None, cache=False) as ds:
                processed_vars = {}
                
                for var_name, da in ds.data_vars.items():
                    var_start = time.time()
                    
                    # 跳过非地理变量
                    if not hasattr(da, 'rio') or da.rio.x_dim is None or da.rio.y_dim is None:
                        processed_vars[var_name] = da
                        continue
                    
                    # 设置CRS
                    if da.rio.crs is None:
                        da = da.rio.write_crs(self.target_crs)
                    
                    # 极速处理流水线
                    da_processed = self._ultra_fast_resample(da, method=resample_method)
                    da_processed = self._ultra_fast_clip(da_processed)
                    
                    processed_vars[var_name] = da_processed
                    
                    var_time = time.time() - var_start
                    logger.info(f"  变量 {var_name}: {var_time:.2f}s")
                    
                    # 强制垃圾回收
                    del da_processed
                    gc.collect()
                
                # 创建输出数据集
                processed_ds = xr.Dataset(processed_vars, attrs=ds.attrs)
                
                # 高效编码设置
                encoding = {
                    var_name: {
                        'zlib': True,
                        'complevel': 1,  # 降低压缩级别提高速度
                        'shuffle': False,  # 禁用shuffle提高速度
                        '_FillValue': -9999,
                        'dtype': 'float32'  # 使用float32减少文件大小
                    }
                    for var_name in processed_ds.data_vars
                }
                
                # 保存文件
                save_start = time.time()
                processed_ds.to_netcdf(
                    output_path, 
                    encoding=encoding,
                    engine='netcdf4',
                    format='NETCDF4'
                )
                save_time = time.time() - save_start
                
                processed_ds.close()
                del processed_ds, processed_vars
                gc.collect()
            
            total_time = time.time() - start_time
            logger.info(f"完成 {os.path.basename(output_path)} ({total_time:.2f}s, 保存:{save_time:.2f}s)")
            return output_path
            
        except Exception as e:
            logger.error(f"处理失败 {nc_file}: {e}")
            raise
    
    def process_batch_ultra_fast(self, input_folder, resample_method='bilinear', 
                                recursive=False, skip_existing=True, use_parallel=True):
        """极速批量处理"""
        start_time = time.time()
        
        # 搜索文件
        nc_files = []
        if recursive:
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(('.nc', '.netcdf')):
                        nc_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(input_folder):
                if file.lower().endswith(('.nc', '.netcdf')):
                    nc_files.append(os.path.join(input_folder, file))
        
        if not nc_files:
            logger.warning(f"未找到NC文件: {input_folder}")
            return [], []
        
        logger.info(f"找到 {len(nc_files)} 个文件")
        
        processed_files = []
        failed_files = []
        
        if use_parallel and len(nc_files) > 1:
            # 并行处理
            logger.info(f"并行处理 ({self.n_workers} 进程)")
            
            process_func = partial(
                ultra_fast_worker,
                target_resolution=self.target_resolution,
                clip_shapefile=self.clip_shapefile,
                output_dir=self.output_dir,
                resample_method=resample_method,
                skip_existing=skip_existing
            )
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_file = {
                    executor.submit(process_func, nc_file): nc_file 
                    for nc_file in nc_files
                }
                
                for i, future in enumerate(as_completed(future_to_file), 1):
                    nc_file = future_to_file[future]
                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        if result:
                            processed_files.append(result)
                        else:
                            failed_files.append(nc_file)
                        
                        if i % 10 == 0 or i == len(nc_files):
                            logger.info(f"进度: {i}/{len(nc_files)}")
                            
                    except Exception as e:
                        failed_files.append(nc_file)
                        logger.error(f"失败: {os.path.basename(nc_file)} - {e}")
        else:
            # 串行处理
            logger.info("串行处理")
            for i, nc_file in enumerate(nc_files, 1):
                try:
                    result = self.process_single_file_ultra_fast(
                        nc_file, resample_method=resample_method, skip_existing=skip_existing
                    )
                    processed_files.append(result)
                    
                    if i % 10 == 0 or i == len(nc_files):
                        logger.info(f"进度: {i}/{len(nc_files)}")
                        
                except Exception as e:
                    failed_files.append(nc_file)
                    logger.error(f"失败: {os.path.basename(nc_file)} - {e}")
        
        total_time = time.time() - start_time
        logger.info(f"批量处理完成 ({total_time:.2f}s). 成功: {len(processed_files)}, 失败: {len(failed_files)}")
        
        return processed_files, failed_files


def ultra_fast_worker(nc_file, target_resolution, clip_shapefile, output_dir, resample_method, skip_existing):
    """极速工作进程"""
    try:
        # 创建临时处理器
        processor = UltraFastNCResampler(
            target_resolution=target_resolution,
            clip_shapefile=clip_shapefile,
            output_dir=output_dir,
            n_workers=1  # 工作进程内部不再并行
        )
        
        return processor.process_single_file_ultra_fast(
            nc_file, resample_method=resample_method, skip_existing=skip_existing
        )
        
    except Exception as e:
        logger.error(f"工作进程失败 {nc_file}: {e}")
        return None


if __name__ == "__main__":
    # 配置参数
    target_resolution = "1000m"
    clip_shapefile = "G:\\Agr-qu\\clip.shp"
    output_dir = "G:\\CNN_SpatialDownscaling\\ERA5\\ERA5_1000m_1"
    input_folder = "G:\\CNN_SpatialDownscaling\\ERA5\\2018_processed_parallel"
    
    # 性能配置
    use_parallel = True
    skip_existing = True
    memory_limit_gb = 32  # 根据你的内存情况调整
    n_workers = None  # 自动检测
    
    # 创建极速处理器
    processor = UltraFastNCResampler(
        target_resolution=target_resolution,
        clip_shapefile=clip_shapefile,
        output_dir=output_dir,
        n_workers=n_workers,
        memory_limit_gb=memory_limit_gb
    )
    
    # 执行处理
    try:
        processed_files, failed_files = processor.process_batch_ultra_fast(
            input_folder=input_folder,
            resample_method='bilinear',
            recursive=False,
            skip_existing=skip_existing,
            use_parallel=use_parallel
        )
        
        print(f"\n=== 极速处理完成 ===")
        print(f"成功: {len(processed_files)} 个文件")
        print(f"失败: {len(failed_files)} 个文件")
        
        if failed_files:
            print("\n失败文件:")
            for ff in failed_files[:10]:
                print(f"  ✗ {os.path.basename(ff)}")
                
    except Exception as e:
        print(f"处理出错: {e}")