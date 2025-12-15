import os
import glob
import warnings
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.warp import transform_geom
from rasterio.transform import from_bounds
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from functools import partial
import gc
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator
#from scripts.Data_Preprocessing.process_netCDF import output_dir

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用dask相关警告
warnings.filterwarnings('ignore', category=UserWarning, module='xarray')

class NCResampler:
    def __init__(self, target_resolution, clip_shapefile, output_dir, n_workers=None, chunk_size=None, use_fast_resample=True):
        """
        NC文件重采样和裁剪处理器（性能优化版）
        
        Parameters:
        -----------
        target_resolution : float, tuple, or str
            目标分辨率
        clip_shapefile : str
            裁剪用的shapefile路径
        output_dir : str
            输出目录路径
        n_workers : int
            并行处理的进程数，默认为CPU核心数-1
        chunk_size : str or dict
            xarray chunk大小，None表示不使用chunking
        use_fast_resample : bool
            是否使用快速重采样（scipy），默认True
        """
        # 解析目标分辨率
        if isinstance(target_resolution, (tuple, list)):
            self.x_resolution, self.x_unit = self._detect_resolution_unit(target_resolution[0])
            self.y_resolution, self.y_unit = self._detect_resolution_unit(target_resolution[1])
        else:
            self.x_resolution, self.x_unit = self._detect_resolution_unit(target_resolution)
            self.y_resolution, self.y_unit = self.x_resolution, self.x_unit
        
        self.clip_shapefile = clip_shapefile
        self.output_dir = output_dir
        self.target_crs = "EPSG:4326"
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.chunk_size = chunk_size
        self.use_fast_resample = use_fast_resample
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取裁剪边界（缓存）
        self.clip_gdf = self._load_clip_boundary()
        self.clip_bounds = self.clip_gdf.total_bounds
        
        # 预计算目标网格信息
        self._precalculate_grid_info()
        
        # 预计算裁剪掩膜
        self._precalculate_clip_mask()
        
        logger.info(f"初始化完成 - 并行进程数: {self.n_workers}")
        logger.info(f"目标分辨率: X={self.x_resolution}({self.x_unit}), Y={self.y_resolution}({self.y_unit})")
        logger.info(f"快速重采样: {'启用' if use_fast_resample else '禁用'}")
        
    def _load_clip_boundary(self):
        """加载并预处理裁剪边界"""
        try:
            gdf = gpd.read_file(self.clip_shapefile)
            
            if gdf.crs != self.target_crs:
                logger.info(f"转换shapefile坐标系从 {gdf.crs} 到 {self.target_crs}")
                gdf = gdf.to_crs(self.target_crs)
            
            return gdf
            
        except Exception as e:
            logger.error(f"读取shapefile失败: {e}")
            raise
    
    def _detect_resolution_unit(self, resolution_input):
        """检测分辨率单位"""
        if isinstance(resolution_input, (int, float)):
            return resolution_input, "degrees"
        
        resolution_str = str(resolution_input).lower()
        degree_markers = ['°', '′', '″', 'degree', 'deg', 'arc']
        meter_markers = ['km', 'm', 'meter', 'metre', 'kilometer', 'kilometre']
        
        has_degree_marker = any(marker in resolution_str for marker in degree_markers)
        has_meter_marker = any(marker in resolution_str for marker in meter_markers)
        
        import re
        numbers = re.findall(r'\d+\.?\d*', resolution_str)
        if not numbers:
            raise ValueError(f"无法从分辨率输入中提取数值: {resolution_input}")
        
        numeric_value = float(numbers[0])
        
        if has_degree_marker:
            if '°' in resolution_str:
                unit_type = "degrees"
                if '′' in resolution_str or '″' in resolution_str:
                    degree_match = re.search(r'(\d+\.?\d*)°', resolution_str)
                    minute_match = re.search(r'(\d+\.?\d*)′', resolution_str)
                    second_match = re.search(r'(\d+\.?\d*)″', resolution_str)
                    
                    degrees = float(degree_match.group(1)) if degree_match else 0
                    minutes = float(minute_match.group(1)) if minute_match else 0
                    seconds = float(second_match.group(1)) if second_match else 0
                    
                    numeric_value = degrees + minutes/60.0 + seconds/3600.0
            elif '′' in resolution_str:
                unit_type = "degrees"
                numeric_value = numeric_value / 60.0
            elif '″' in resolution_str:
                unit_type = "degrees"
                numeric_value = numeric_value / 3600.0
            else:
                unit_type = "degrees"
                
        elif has_meter_marker:
            unit_type = "meters"
            if 'km' in resolution_str:
                numeric_value = numeric_value * 1000
        else:
            if numeric_value > 180:
                unit_type = "meters"
                logger.warning(f"未检测到单位标记，根据数值大小({numeric_value})推断为米")
            else:
                unit_type = "degrees"
                logger.warning(f"未检测到单位标记，根据数值大小({numeric_value})推断为度")
        
        logger.info(f"分辨率解析: {resolution_input} -> {numeric_value} ({unit_type})")
        return numeric_value, unit_type
    
    def _meters_to_degrees(self, meters, latitude=None):
        """
        将米转换为度，考虑纬度变化
        """
        # 纬度方向：1度 ≈ 111320米（固定）
        lat_degrees = meters / 111320.0
        
        # 经度方向：随纬度变化
        if latitude is None:
            # 使用研究区域中心纬度
            latitude = (self.clip_bounds[1] + self.clip_bounds[3]) / 2
        
        # 1度经度的米数 = 111320 * cos(纬度)
        meters_per_lon_degree = 111320.0 * np.cos(np.radians(latitude))
        lon_degrees = meters / meters_per_lon_degree
        
        return lon_degrees, lat_degrees
    
    def _precalculate_grid_info(self):
        """预计算目标网格信息"""
        # 如果分辨率单位是米，转换为度
        if self.x_unit == "meters" or self.y_unit == "meters":
            # 计算研究区域中心纬度
            center_lat = (self.clip_bounds[1] + self.clip_bounds[3]) / 2
            
            if self.x_unit == "meters":
                self.x_res_deg, _ = self._meters_to_degrees(self.x_resolution, center_lat)
            else:
                self.x_res_deg = self.x_resolution
                
            if self.y_unit == "meters":
                _, self.y_res_deg = self._meters_to_degrees(self.y_resolution, center_lat)
            else:
                self.y_res_deg = self.y_resolution
        else:
            self.x_res_deg = self.x_resolution
            self.y_res_deg = self.y_resolution
        
        # 预计算目标坐标网格，确保严格按照 target_resolution 生成行列数
        bounds = self.clip_bounds

        # 计算行列数，确保每个像元严格为 target_resolution
        min_x = bounds[0]
        max_x = bounds[2]
        min_y = bounds[1]
        max_y = bounds[3]

        n_cols = int(np.round((max_x - min_x) / self.x_res_deg))
        n_rows = int(np.round((max_y - min_y) / self.y_res_deg))

        # 重新计算max_x, max_y，确保边界正好覆盖且像元数与分辨率严格一致
        max_x_aligned = min_x + n_cols * self.x_res_deg
        max_y_aligned = min_y + n_rows * self.y_res_deg

        self.target_x_coords = np.linspace(min_x + self.x_res_deg/2, max_x_aligned - self.x_res_deg/2, n_cols)
        self.target_y_coords = np.linspace(min_y + self.y_res_deg/2, max_y_aligned - self.y_res_deg/2, n_rows)

        # y坐标递增（从南到北），如需从北到南可反转
        if self.target_y_coords[0] < self.target_y_coords[-1]:
            self.target_y_coords = self.target_y_coords[::-1]

        self.target_shape = (len(self.target_y_coords), len(self.target_x_coords))

        # 创建目标transform，严格与行列数和分辨率一致
        self.target_transform = from_bounds(
            min_x, min_y, max_x_aligned, max_y_aligned,
            n_cols, n_rows
        )

        logger.info(f"目标网格尺寸: {self.target_shape}")
        logger.info(f"目标分辨率(度): X={self.x_res_deg:.6f}, Y={self.y_res_deg:.6f}")

    def _precalculate_clip_mask(self):
        """预计算裁剪掩膜以加速矢量裁剪"""
        try:
            from rasterio.features import rasterize
            
            # 创建目标网格的坐标
            xx, yy = np.meshgrid(self.target_x_coords, self.target_y_coords[::-1])  # 翻转y坐标
            
            # 栅格化shapefile几何体
            self.clip_mask = rasterize(
                [(geom, 1) for geom in self.clip_gdf.geometry],
                out_shape=self.target_shape,
                transform=self.target_transform,
                fill=0,
                dtype=np.uint8
            ).astype(bool)
            
            logger.info("预计算裁剪掩膜完成")
            
        except Exception as e:
            logger.warning(f"预计算裁剪掩膜失败，将使用传统方法: {e}")
            self.clip_mask = None

    def _convert_to_wgs84(self, da):
        """转换数据到WGS84坐标系"""
        try:
            # 如果已经是WGS84，直接返回
            if da.rio.crs == self.target_crs:
                return da
                
            # 1. 处理缺失CRS的情况
            if da.rio.crs is None:
                try:
                    da = da.rio.write_crs("EPSG:6933")  # SMAP常用投影
                    logger.warning(f"DataArray CRS缺失，假设为 EPSG:6933")
                except Exception:
                    da = da.rio.write_crs(self.target_crs)
                    logger.warning(f"DataArray CRS缺失，回退到默认 {self.target_crs}")
            
            # 2. 重投影到WGS84
            if da.rio.crs != self.target_crs:
                logger.info(f"重投影 DataArray 从 {da.rio.crs} 到 {self.target_crs}")
                da = da.rio.reproject(self.target_crs, resampling=Resampling.bilinear)
            
            return da
            
        except Exception as e:
            logger.error(f"坐标系转换失败: {e}")
            raise
    
    def _fast_resample_scipy(self, da, method='linear'):
        """使用scipy进行快速重采样"""
        try:
            # 获取原始坐标
            x_coords = da.coords[da.rio.x_dim].values
            y_coords = da.coords[da.rio.y_dim].values
            
            # 确保坐标是单调的
            if x_coords[0] > x_coords[-1]:
                x_coords = x_coords[::-1]
                flip_x = True
            else:
                flip_x = False
                
            if y_coords[0] > y_coords[-1]:
                y_coords = y_coords[::-1]
                flip_y = True
            else:
                flip_y = False
            
            data = da.values
            if flip_x or flip_y:
                if flip_x and flip_y:
                    data = np.flip(np.flip(data, axis=-1), axis=-2)
                elif flip_x:
                    data = np.flip(data, axis=-1)
                elif flip_y:
                    data = np.flip(data, axis=-2)
            
            # 处理多维数据
            if data.ndim == 2:
                # 2D数据直接处理
                interpolator = RegularGridInterpolator(
                    (y_coords, x_coords), 
                    data, 
                    method=method, 
                    bounds_error=False, 
                    fill_value=np.nan
                )
                
                # 创建目标网格
                target_xx, target_yy = np.meshgrid(self.target_x_coords, self.target_y_coords)
                points = np.column_stack([target_yy.ravel(), target_xx.ravel()])
                
                # 插值
                resampled_data = interpolator(points).reshape(self.target_shape)
                
            else:
                # 多维数据，逐层处理
                resampled_data = np.full((data.shape[0],) + self.target_shape, np.nan)
                
                target_xx, target_yy = np.meshgrid(self.target_x_coords, self.target_y_coords)
                points = np.column_stack([target_yy.ravel(), target_xx.ravel()])
                
                for i in range(data.shape[0]):
                    interpolator = RegularGridInterpolator(
                        (y_coords, x_coords), 
                        data[i], 
                        method=method, 
                        bounds_error=False, 
                        fill_value=np.nan
                    )
                    resampled_data[i] = interpolator(points).reshape(self.target_shape)
            
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
            
            # 设置CRS和transform
            da_resampled = da_resampled.rio.write_crs(self.target_crs)
            da_resampled = da_resampled.rio.write_transform(self.target_transform)
            
            return da_resampled
            
        except Exception as e:
            logger.warning(f"快速重采样失败，回退到rioxarray方法: {e}")
            return None
    
    def _resample_data_optimized(self, da, method='bilinear'):
        """优化的重采样函数"""
        try:
            # 确保数据在WGS84坐标系中
            if da.rio.crs != self.target_crs:
                da = da.rio.reproject(self.target_crs, resampling=Resampling.bilinear)
            
            # 尝试快速重采样
            if self.use_fast_resample:
                scipy_method = 'linear' if method == 'bilinear' else 'nearest'
                da_resampled = self._fast_resample_scipy(da, method=scipy_method)
                if da_resampled is not None:
                    logger.info(f"使用scipy快速重采样到: {da_resampled.shape}")
                    return da_resampled
            
            # 回退到rioxarray方法
            resampling_methods = {
                'bilinear': Resampling.bilinear,
                'nearest': Resampling.nearest,
                'cubic': Resampling.cubic,
                'average': Resampling.average
            }
            
            resampling = resampling_methods.get(method, Resampling.bilinear)
            
            logger.info(f"使用rioxarray重采样到: {self.target_shape}")
            da_resampled = da.rio.reproject(
                self.target_crs,
                shape=self.target_shape,
                transform=self.target_transform,
                resampling=resampling
            )
            
            return da_resampled
            
        except Exception as e:
            logger.error(f"重采样失败: {e}")
            raise
    
    def _clip_with_mask_optimized(self, da):
        """使用预计算掩膜进行快速裁剪"""
        try:
            if self.clip_mask is not None:
                # 使用预计算的掩膜
                if da.ndim == 2:
                    da.values[~self.clip_mask] = np.nan
                else:
                    # 多维数据
                    for i in range(da.shape[0]):
                        da.values[i][~self.clip_mask] = np.nan
                
                return da
            else:
                # 回退到传统方法
                return self._clip_with_shapefile_traditional(da)
                
        except Exception as e:
            logger.warning(f"快速裁剪失败，使用传统方法: {e}")
            return self._clip_with_shapefile_traditional(da)
    
    def _clip_with_shapefile_traditional(self, da):
        """传统的矢量裁剪函数"""
        try:
            da_clipped = da.rio.clip(
                self.clip_gdf.geometry.values,
                self.clip_gdf.crs,
                drop=False,  # 不删除像素，只设置为NaN
                invert=False
            )
            return da_clipped
            
        except Exception as e:
            logger.error(f"矢量裁剪失败: {e}")
            raise
    
    def _fill_edge_simple(self, da):
        """简化的边缘填充函数"""
        try:
            # 跳过边缘填充以提高速度
            # 如果确实需要，可以启用此功能
            return da
            
        except Exception as e:
            logger.warning(f"边缘处理失败: {e}")
            return da
    
    def process_single_file_optimized(self, nc_file, output_filename=None, resample_method='bilinear', skip_existing=True):
        """优化的单文件处理函数"""
        start_time = time.time()
        
        try:
            # 确定输出文件路径
            if output_filename is None:
                base_name = os.path.splitext(os.path.basename(nc_file))[0]
                output_filename = f"{base_name}_1000m.nc"
            
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 检查输出文件是否已存在
            if skip_existing and os.path.exists(output_path):
                logger.info(f"跳过已存在文件: {os.path.basename(output_path)}")
                return output_path
            
            logger.info(f"开始处理: {os.path.basename(nc_file)}")
            
            # 读取数据（避免使用chunks）
            ds = xr.open_dataset(nc_file, chunks=None)
            
            processed_vars = {}
            
            for var_name, da in ds.data_vars.items():
                var_start = time.time()
                logger.info(f"处理变量: {var_name} - 形状: {da.shape}")

                # 跳过无空间维度的变量（如crs、time_bnds等）
                spatial_dims = set(da.dims)
                # 常见空间维度名
                possible_x = ["x", "lon", "longitude"]
                possible_y = ["y", "lat", "latitude"]
                x_dim = next((d for d in possible_x if d in spatial_dims), None)
                y_dim = next((d for d in possible_y if d in spatial_dims), None)
                if x_dim is None or y_dim is None:
                    logger.info(f"跳过无空间维度变量: {var_name}")
                    processed_vars[var_name] = da
                    continue

                # 如果空间维度不是x/y，自动rename并set_spatial_dims
                rename_dict = {}
                if x_dim != "x":
                    rename_dict[x_dim] = "x"
                if y_dim != "y":
                    rename_dict[y_dim] = "y"
                if rename_dict:
                    da = da.rename(rename_dict)
                    try:
                        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
                        logger.info(f"变量 {var_name} 空间维度已重命名为x/y")
                    except Exception as e:
                        logger.warning(f"变量 {var_name} set_spatial_dims失败: {e}")
                        processed_vars[var_name] = da
                        continue

                # 确保有地理信息
                if da.rio.crs is None:
                    if hasattr(da, 'spatial_ref'):
                        da = da.rio.write_crs(da.spatial_ref)
                    else:
                        da = da.rio.write_crs(self.target_crs)

                # 检查是否需要处理（跳过非地理变量）
                if not hasattr(da, 'rio') or da.rio.x_dim is None or da.rio.y_dim is None:
                    logger.info(f"跳过非地理变量: {var_name}")
                    processed_vars[var_name] = da
                    continue

                # 流水线处理
                da_processed = self._convert_to_wgs84(da)
                da_processed = self._resample_data_optimized(da_processed, method=resample_method)
                da_processed = self._clip_with_mask_optimized(da_processed)
                da_processed = self._fill_edge_simple(da_processed)

                # 转为float32类型
                da_processed = da_processed.astype(np.float32)

                processed_vars[var_name] = da_processed

                var_time = time.time() - var_start
                logger.info(f"变量 {var_name} 处理完成 ({var_time:.2f}s)")

                # 释放内存
                del da_processed
                gc.collect()
            
            # 重建数据集
            processed_ds = xr.Dataset(processed_vars, attrs=ds.attrs)
            
            # 保存结果
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 优化的编码设置，强制float32输出
            encoding = {}
            for var_name in processed_ds.data_vars:
                encoding[var_name] = {
                    'zlib': True,
                    'complevel': 4,
                    'shuffle': True,
                    '_FillValue': np.float32(-9999),
                    'dtype': 'float32',
                    'chunksizes': None  # 禁用chunking
                }
            
            # 保存文件
            save_start = time.time()
            processed_ds.to_netcdf(output_path, encoding=encoding)
            save_time = time.time() - save_start
            
            # 关闭数据集，释放内存
            ds.close()
            processed_ds.close()
            del ds, processed_ds, processed_vars
            gc.collect()
            
            elapsed_time = time.time() - start_time
            logger.info(f"文件处理完成 ({elapsed_time:.2f}s, 保存:{save_time:.2f}s): {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"处理文件 {nc_file} 时出错: {e}")
            raise
    
    def process_multiple_files_parallel(self, input_folder, resample_method='bilinear', recursive=False, skip_existing=True):
        """并行批量处理"""
        start_time = time.time()
        
        if not os.path.exists(input_folder):
            logger.error(f"输入文件夹不存在: {input_folder}")
            raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
        
        # 搜索NC文件
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
            logger.warning(f"在文件夹 {input_folder} 中没有找到NC文件")
            return [], []
        
        logger.info(f"找到 {len(nc_files)} 个NC文件，开始并行处理...")
        logger.info(f"使用 {self.n_workers} 个并行进程")
        
        processed_files = []
        failed_files = []
        
        # 创建处理函数的部分应用
        process_func = partial(
            process_single_file_worker,
            target_resolution=(self.x_resolution, self.x_unit, self.y_resolution, self.y_unit),
            clip_shapefile=self.clip_shapefile,
            output_dir=self.output_dir,
            target_crs=self.target_crs,
            resample_method=resample_method,
            use_fast_resample=self.use_fast_resample,
            skip_existing=skip_existing
        )
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_file = {
                executor.submit(process_func, nc_file): nc_file 
                for nc_file in nc_files
            }
            
            for i, future in enumerate(as_completed(future_to_file), 1):
                nc_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result)
                        logger.info(f"进度: {i}/{len(nc_files)} - 完成: {os.path.basename(nc_file)}")
                    else:
                        failed_files.append(nc_file)
                        logger.error(f"进度: {i}/{len(nc_files)} - 失败: {os.path.basename(nc_file)}")
                except Exception as e:
                    failed_files.append(nc_file)
                    logger.error(f"进度: {i}/{len(nc_files)} - 失败: {os.path.basename(nc_file)} - 错误: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"批量处理完成 ({elapsed_time:.2f}s)。成功: {len(processed_files)}, 失败: {len(failed_files)}")
        
        return processed_files, failed_files

    def process_multiple_files_serial(self, input_folder, resample_method='bilinear', recursive=False, skip_existing=True):
        """串行批量处理"""
        start_time = time.time()
        
        if not os.path.exists(input_folder):
            logger.error(f"输入文件夹不存在: {input_folder}")
            raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
        
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
            logger.warning(f"在文件夹 {input_folder} 中没有找到NC文件")
            return [], []
        
        logger.info(f"找到 {len(nc_files)} 个NC文件，开始串行处理...")
        
        processed_files = []
        failed_files = []
        
        for i, nc_file in enumerate(nc_files, 1):
            try:
                logger.info(f"处理进度: {i}/{len(nc_files)} - {os.path.basename(nc_file)}")
                output_path = self.process_single_file_optimized(nc_file, resample_method=resample_method, skip_existing=skip_existing)
                processed_files.append(output_path)
            except Exception as e:
                logger.error(f"文件 {nc_file} 处理失败: {e}")
                failed_files.append(nc_file)
        
        elapsed_time = time.time() - start_time
        logger.info(f"串行处理完成 ({elapsed_time:.2f}s)。成功: {len(processed_files)}, 失败: {len(failed_files)}")
        
        return processed_files, failed_files


def process_single_file_worker(nc_file, target_resolution, clip_shapefile, output_dir, target_crs, resample_method, use_fast_resample, skip_existing):
    """工作进程函数（用于并行处理）"""
    try:
        # 在工作进程中重新创建处理器实例
        x_resolution, x_unit, y_resolution, y_unit = target_resolution
        
        # 创建临时处理器
        temp_resampler = NCResampler.__new__(NCResampler)
        temp_resampler.x_resolution = x_resolution
        temp_resampler.x_unit = x_unit
        temp_resampler.y_resolution = y_resolution
        temp_resampler.y_unit = y_unit
        temp_resampler.clip_shapefile = clip_shapefile
        temp_resampler.output_dir = output_dir
        temp_resampler.target_crs = target_crs
        temp_resampler.chunk_size = None
        temp_resampler.use_fast_resample = use_fast_resample
        
        # 加载必要的数据
        temp_resampler.clip_gdf = temp_resampler._load_clip_boundary()
        temp_resampler.clip_bounds = temp_resampler.clip_gdf.total_bounds
        temp_resampler._precalculate_grid_info()
        temp_resampler._precalculate_clip_mask()
        
        # 处理文件
        return temp_resampler.process_single_file_optimized(nc_file, resample_method=resample_method, skip_existing=skip_existing)
        
    except Exception as e:
        logger.error(f"工作进程处理文件 {nc_file} 失败: {e}")
        return None


if __name__ == "__main__":
    # 配置参数
    target_resolution = "0.0083333333°" #0.0083333333
    clip_shapefile = "G:\\Agr-qu\\clip.shp"

 
    # 处理ERA5数据
    output_dir = "I:\\CNN_SpatialDownscaling\\ERA5\\ERA5_1000"
    input_folder = "G:\\CNN_SpatialDownscaling\\ERA5\\era5_processed_parallel"
    '''
    # 处理NDVI数据
    output_dir = "G:\\CNN_SpatialDownscaling\\201712and201901\\ndvi\\2019\\NDVI_0p0625"
    input_folder = "G:\\CNN_SpatialDownscaling\\201712and201901\\ndvi\\2019"
    '''
    

    recursive_search = False
    
    # 性能配置
    n_workers = 8  # 建议设置为较小值，避免内存不足
    use_parallel = True  # 是否使用并行处理
    use_fast_resample = True  # 是否使用快速重采样
    skip_existing = True  # 是否跳过已存在的输出文件
    
    # 创建处理器
    resampler = NCResampler(
        target_resolution=target_resolution,
        clip_shapefile=clip_shapefile,
        output_dir=output_dir,
        n_workers=n_workers,
        chunk_size=None,  # 不使用chunks
        use_fast_resample=use_fast_resample
    )
    
    # 选择处理模式
    try:
        if use_parallel:
            logger.info("使用并行处理模式")
            processed_files, failed_files = resampler.process_multiple_files_parallel(
                input_folder=input_folder,
                resample_method='bilinear',
                recursive=recursive_search,
                skip_existing=skip_existing
            )
        else:
            logger.info("使用串行处理模式")
            processed_files, failed_files = resampler.process_multiple_files_serial(
                input_folder=input_folder,
                resample_method='bilinear',
                recursive=recursive_search,
                skip_existing=skip_existing
            )
        
        print(f"\n=== 处理完成 ===")
        print(f"输入文件夹: {input_folder}")
        print(f"输出文件夹: {output_dir}")
        print(f"成功处理: {len(processed_files)} 个文件")
        
        if processed_files:
            print(f"\n成功处理的文件 (前10个):")
            for pf in processed_files[:10]:
                print(f"  ✓ {os.path.basename(pf)}")
            if len(processed_files) > 10:
                print(f"  ... 还有 {len(processed_files) - 10} 个文件")
        
        if failed_files:
            print(f"\n失败: {len(failed_files)} 个文件")
            for ff in failed_files[:10]:
                print(f"  ✗ {os.path.basename(ff)}")
            if len(failed_files) > 10:
                print(f"  ... 还有 {len(failed_files) - 10} 个失败文件")
        else:
            print("所有文件处理成功！")
            
    except Exception as e:
        print(f"批量处理出错: {e}")