import os
import json
import numpy as np
import xarray as xr
import rasterio
import logging
import pandas as pd
from datetime import datetime, timedelta
import re
import netCDF4
from osgeo import gdal

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
gdal.UseExceptions()

# 禁用 HDF5 文件锁定
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class DataLoader:
    def __init__(self, config_path):
        """初始化数据加载器，加载配置文件
        
        Args:
            config_path (str): 配置文件路径
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在，请检查路径或创建文件")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.global_paths = self.config['global_data_paths']
        self.local_paths = self.config['local_data_paths']
        self.var_names = self.config['var_names']
        self.nodata_values = self.config['nodata_values']
        self.time_window = self.config['data_params']['time_window']
        self.global_shape = tuple(self.config['data_params']['global_shape'])
        self.local_shape = tuple(self.config['data_params']['local_shape'])
        
        # 预加载静态数据和地理信息
        self.static_data_global, self.lon_lat_global = self._load_static_data_and_coords(self.global_paths, self.global_shape, is_global=True)
        self.static_data_local, self.lon_lat_local = self._load_static_data_and_coords(self.local_paths, self.local_shape, is_global=False)
        
    def _load_static_data_and_coords(self, paths, shape, is_global):
        """加载所有静态数据并生成经纬度坐标"""
        data = {}
        logger.info("开始加载静态数据...")
        
        # 确定静态数据的路径前缀
        prefix = 'dem'
        if not is_global: prefix = 'dem_1km'
        
        dem_path = paths.get(prefix)
        if dem_path is None or not os.path.exists(dem_path):
            logger.error(f"静态文件 {dem_path} 不存在")
            return {}, None

        # 使用gdal读取以兼容.img和.tif
        try:
            ds = gdal.Open(dem_path)
            if ds is None:
                raise FileNotFoundError(f"无法打开文件: {dem_path}")
            height, width = ds.RasterYSize, ds.RasterXSize
            gt = ds.GetGeoTransform()
            ds = None  # 关闭文件
        except Exception as e:
            logger.error(f"无法从{dem_path}获取地理信息: {e}")
            return {}, None

        lon_grid = np.array([gt[0] + gt[1] * j for j in range(width)])
        lat_grid = np.array([gt[3] + gt[5] * i for i in range(height)])
        lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
        lon_lat = np.stack([lon_grid.flatten(), lat_grid.flatten()], axis=-1)
            
        # 修正: 从静态数据列表中移除 ndvi 和 ndvi_1km
        static_vars = ['dem', 'slope', 'aspect', 'clcd']
        for var_type in static_vars:
            path_key = var_type
            if not is_global: path_key = var_type + '_1km'
            path = paths.get(path_key, None)
            
            if not path or not os.path.exists(path):
                logger.warning(f"静态文件 {path} 不存在，跳过。")
                continue
            try:
                # 使用rasterio兼容img和tif
                with rasterio.open(path) as src:
                    d = src.read(1)
                    nodata = src.nodata
                    if nodata is not None:
                        d = np.where(d == nodata, np.nan, d)
                    data[var_type] = d.flatten().astype(np.float32)
                logger.info(f"静态数据 {var_type} 加载成功，形状: {data[var_type].shape}")
            except Exception as e:
                logger.error(f"加载静态数据 {var_type} 失败: {e}")
                data[var_type] = np.full(height * width, np.nan)
        return data, lon_lat

    def parse_time_from_filename(self, filename, data_type):
        """从文件名解析时间"""
        filename = os.path.basename(filename)
        try:
            if data_type == 'cldas':
                m = re.match(r'CLDAS_(\d{8})_(\d{2})\.nc', filename)
                if m:
                    return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H')
            elif data_type in ['smap', 'smap_1km']:
                parts = filename.split('_')
                for part in parts:
                    if 'T' in part and len(part) == 15:
                        return datetime.strptime(part, '%Y%m%dT%H%M%S')
            elif data_type in ['era5', 'era5_1km']:
                parts = filename.split('_')
                if len(parts) >= 5:
                    time_parts = parts[1:5]
                    time_str = '_'.join(time_parts)
                    return datetime.strptime(time_str, '%Y_%m_%d_%H')
            elif data_type == 'modis':
                m = re.match(r'MOD11_(\d{8})_(\d{2})\.nc', filename)
                if m:
                    return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H')
            elif data_type in ['ndvi', 'ndvi_1km']:
                m = filename.split('_')[1] + '_' + filename.split('_')[2]
                if m:
                    return datetime.strptime(m, '%Y-%m-%d_%H%M%S')
        except Exception as e:
            logger.warning(f"解析文件名 {filename} 失败，类型: {data_type}, 错误: {str(e)}")
        return None

    def get_file_list(self, path, start_date, end_date, data_type):
        """获取指定时间范围内的文件列表"""
        if not os.path.exists(path):
            logger.warning(f"数据目录 {path} 不存在，返回空文件列表")
            return []
        
        # 根据数据类型调整时间窗口
        adj_start_date = start_date
        adj_end_date = end_date
        if data_type in ['smap', 'smap_1km']:
            adj_start_date = start_date - timedelta(hours=3)
            adj_end_date = end_date + timedelta(hours=3)
        elif data_type in ['ndvi', 'ndvi_1km']:
            adj_start_date = start_date - timedelta(days=16)
            adj_end_date = end_date + timedelta(days=16)

        files = []
        for filename in os.listdir(path):
            file_time = self.parse_time_from_filename(filename, data_type)
            if file_time and adj_start_date <= file_time <= adj_end_date:
                files.append((os.path.join(path, filename), file_time))
        
        files = sorted(files, key=lambda x: x[1])
        logger.info(f"获取 {len(files)} 个 {data_type} 文件，路径: {path}, 时间范围: {adj_start_date} 至 {adj_end_date}")
        return files

    def find_closest_file(self, files, target_time, max_delta):
        """
        查找时间最接近且不晚于目标时间的文件。
        这实现了单向搜索，只使用历史数据。
        
        Args:
            files (list): 包含(文件路径, 文件时间)元组的列表。
            target_time (datetime): 目标时间。
            max_delta (timedelta): 初始最大时间差，用于设定搜索窗口。
        """
        if not files: return None
        
        closest_file = None
        min_delta = max_delta
        
        for file_path, file_time in files:
            # 只考虑时间在目标时间点或之前的文件
            if file_time <= target_time:
                delta = target_time - file_time
                if delta < min_delta:
                    min_delta = delta
                    closest_file = file_path
        
        return closest_file

    def _classify_soil_moisture(self, data, nodata_value):
        """
        土壤湿度分类函数
        分类：缺失值(0), 0-20%(1), 20-40%(2), 40-60%(3), 60-80%(4), 80-100%(5)
        
        Args:
            data: 土壤湿度数据数组
            nodata_value: 缺失值标记
        
        Returns:
            分类后的数组 (整型)
        """
        classified = np.zeros_like(data, dtype=np.int32)
        
        # 处理缺失值
        if nodata_value is not None:
            missing_mask = (data == nodata_value) | np.isnan(data)
        else:
            missing_mask = np.isnan(data)
        
        classified[missing_mask] = 0  # 缺失值类别
        
        # 对有效数据进行分类
        valid_mask = ~missing_mask
        valid_data = data[valid_mask]
        
        # 创建分类条件
        conditions = [
            (valid_data >= 0) & (valid_data < 0.2),    # 0-20%
            (valid_data >= 0.2) & (valid_data < 0.4),  # 20-40%
            (valid_data >= 0.4) & (valid_data < 0.6),  # 40-60%
            (valid_data >= 0.6) & (valid_data < 0.8),  # 60-80%
            (valid_data >= 0.8) & (valid_data <= 1.0)  # 80-100%
        ]
        
        choices = [1, 2, 3, 4, 5]
        
        classified[valid_mask] = np.select(conditions, choices, default=0)
        
        logger.info(f"土壤湿度分类完成: 缺失值={np.sum(classified == 0)}, "
                   f"0-20%={np.sum(classified == 1)}, "
                   f"20-40%={np.sum(classified == 2)}, "
                   f"40-60%={np.sum(classified == 3)}, "
                   f"60-80%={np.sum(classified == 4)}, "
                   f"80-100%={np.sum(classified == 5)}")
        
        return classified

    def _load_dynamic_data(self, file_path, var_name, shape, nodata_value, data_type=None):
        """
        稳妥地加载动态数据并展平：
        - 优先使用 xarray 打开文件（关闭自动 mask/scale）获取"原始"数组，避免自动将 _FillValue 误当 NaN。
        - 安全检测 encoding/_FillValue/attrs/missing_value，并在确认后将该值替换为 np.nan（否则保留原值）。
        - 特别在 MODIS LST (LST_1km) 上输出额外调试信息。
        - **新增**: 对SMAP土壤湿度数据进行分类处理
        返回：flatten 后的 1D float32 数组（长度 = np.prod(shape)），若失败返回全 NaN 数组。
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                return np.full(np.prod(shape), np.nan, dtype=np.float32)

            # 先用 xarray 打开，关闭自动 mask/scale 与 decode（保证拿到原始数据）
            try:
                ds = xr.open_dataset(file_path, decode_cf=False, mask_and_scale=False, engine="netcdf4")
            except Exception:
                # 若 xarray 打不开再回退到 netCDF4 读取（兼容性保障）
                ds = None

            data = None
            encoding_fill = None
            attrs_fill = None

            if ds is not None and var_name in ds.variables:
                var = ds[var_name]
                # 原始数组（尽量避免 xarray 自动处理）
                data = var.values

                # 从 encoding / attrs 中读取可能的 _FillValue / missing_value
                encoding_fill = var.encoding.get('_FillValue') if isinstance(var.encoding, dict) else None
                attrs_fill = var.attrs.get('_FillValue', None) or var.attrs.get('missing_value', None)

                # 若 encoding_fill 为 numpy scalar object，转换为 python number
                try:
                    if encoding_fill is not None:
                        encoding_fill = float(encoding_fill)
                except Exception:
                    pass

                try:
                    if attrs_fill is not None:
                        attrs_fill = float(attrs_fill)
                except Exception:
                    pass

                # 关闭 dataset（释放文件句柄）
                ds.close()

            else:
                # 回退到 netCDF4 逐变量读取（保留原来的兼容逻辑）
                with netCDF4.Dataset(file_path, 'r') as nc:
                    if var_name not in nc.variables:
                        logger.error(f"{file_path} 中不存在变量 {var_name}")
                        return np.full(np.prod(shape), np.nan, dtype=np.float32)
                    var = nc.variables[var_name]
                    data = var[:]
                    # netCDF4 返回 maskedarray 时，暂不直接用 filled(np.nan)
                    # 我们稍后基于 encoding/_FillValue 做替换
                    # 识别 var 属性中的缺测标记
                    if hasattr(var, "_FillValue"):
                        attrs_fill = float(var._FillValue)
                    elif hasattr(var, "missing_value"):
                        attrs_fill = float(var.missing_value)
                    # scale/offset 若存在，在 decode_cf=False 的前提下通常未应用；若需要可添加
                    # 但你已说明 MODIS 无缩放因子，所以通常不需要处理 scale/add_offset

            # 若没成功读出数据，返回全 NaN
            if data is None:
                logger.error(f"未能从 {file_path} 读取到变量 {var_name} 的数据。")
                return np.full(np.prod(shape), np.nan, dtype=np.float32)

            # 如果是 MaskedArray，则取其 data 和 mask（但不要直接 filled(np.nan)）
            # 如果是 MaskedArray
            if isinstance(data, np.ma.MaskedArray):
                masked = data.mask
                data_values = np.asarray(data.data, dtype=np.float32)  # ✅ 修复：使用 np.asarray 替代 np.array(..., copy=False)
                if hasattr(masked, "any") and masked.any():
                    data_values[masked] = np.nan
                data = data_values
            else:
                data = np.asarray(data, dtype=np.float32)  # ✅ 修复：使用 np.asarray 替代 np.array(..., copy=False)

            # 如果是三维 (time, lat, lon) 并且 time 轴为 1，则取第0层
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]

            # --- 决定哪个值应当被视作缺测（优先级：encoding_fill -> attrs_fill -> nodata_value） ---
            fill_candidates = []
            if encoding_fill is not None:
                fill_candidates.append(encoding_fill)
            if attrs_fill is not None:
                fill_candidates.append(attrs_fill)
            if nodata_value is not None and not (isinstance(nodata_value, float) and np.isnan(nodata_value)):
                fill_candidates.append(float(nodata_value))

            # 过滤掉 nan、None，并去重
            fill_candidates = [fc for fc in set(fill_candidates) if fc is not None and not np.isnan(fc)]

            # 针对 MODIS LST 做特别提示（但不要盲目替换）
            if data_type == "modis" and var_name == "LST_1km":
                # 输出更多调试信息，帮助定位问题
                sample_non_nan = None
                try:
                    non_nan_idx = np.where(~np.isnan(data.flatten()))[0]
                    if non_nan_idx.size > 0:
                        sample_non_nan = data.flatten()[non_nan_idx[:10]]
                except Exception:
                    sample_non_nan = None

                logger.info(f"MODIS {var_name} 文件 {os.path.basename(file_path)} 读取：dtype={data.dtype}, shape={data.shape}, "
                            f"encoding_fill={encoding_fill}, attrs_fill={attrs_fill}, config_nodata={nodata_value}, "
                            f"sample_non_nan_values={sample_non_nan}")

            # 只有当我们确实获取到了候选 fill 值时，才进行替换（并确保替换前检查不会把有效数据置空）
            if fill_candidates:
                replaced_any = False
                for fc in fill_candidates:
                    # 较为稳妥的替换：仅替换与 candidate 精确相等的元素
                    # 注意 float 精度问题——但通常 fill 值是整型（0 或 特定 int），所以精确比较安全
                    try:
                        mask_fc = (data == fc)
                        if np.any(mask_fc):
                            data[mask_fc] = np.nan
                            replaced_any = True
                    except Exception:
                        # 若比较失败（比如 data 含 NaN），用 np.isclose 做后备
                        try:
                            mask_fc = np.isclose(data, fc, atol=1e-6, equal_nan=False)
                            if np.any(mask_fc):
                                data[mask_fc] = np.nan
                                replaced_any = True
                        except Exception:
                            pass

                # 调试信息
                if replaced_any:
                    logger.debug(f"已将候选填充值 {fill_candidates} 替换为 np.nan（文件: {os.path.basename(file_path)}）")
            else:
                # 无候选 fill 值：不进行替换
                logger.debug(f"{os.path.basename(file_path)} 未检测到可替换的 fill 值（encoding/attrs/config 均无），保留原始数值。")

            # **新增：SMAP土壤湿度数据分类处理**
            is_smap = (data_type in ['smap', 'smap_1km']) and (var_name == 'sm_surface_wetness' or 'sm' in var_name.lower())
            if is_smap:
                logger.info(f"对 SMAP 土壤湿度数据进行分类处理: {os.path.basename(file_path)}")
                # 备份原始数据用于分类
                original_data = data.copy()
                # 进行土壤湿度分类
                classified_data = self._classify_soil_moisture(original_data, nodata_value)
                # 返回分类结果（整型）
                return classified_data.flatten().astype(np.float32)

            # 最后检查：若全为 NaN 则单独记录（避免 nanmin/nanmax 抛异常）
            flat = data.flatten()
            if np.all(np.isnan(flat)):
                logger.warning(f"{var_name} 文件 {os.path.basename(file_path)} 全部为 NaN（替换后）。")
                # 返回长度匹配 shape 的全 NaN
                return np.full(np.prod(shape), np.nan, dtype=np.float32)

            # 统计并记录（谨慎使用 nanmin/nanmax）
            try:
                nan_ratio = np.isnan(flat).mean()
                cur_min = float(np.nanmin(flat))
                cur_max = float(np.nanmax(flat))
                logger.info(f"{var_name} 文件 {os.path.basename(file_path)} 加载成功: min={cur_min:.2f}, max={cur_max:.2f}, NaN比例={nan_ratio*100:.2f}%")
            except Exception:
                # fallback（尽量避免抛错）
                logger.info(f"{var_name} 文件 {os.path.basename(file_path)} 加载成功（无法计算 min/max，可能包含 NaN）")

            return flat.astype(np.float32)

        except Exception as e:
            logger.error(f"加载动态数据失败: {file_path}, 变量: {var_name}, 错误: {str(e)}")
            return np.full(np.prod(shape), np.nan, dtype=np.float32)

    def load_data(self, target_date, is_global, is_pretrain):
        """
        加载指定日期的数据，并根据is_global和is_pretrain参数进行区分。
        
        Args:
            target_date (datetime): 目标日期
            is_global (bool): 是否为全局数据
            is_pretrain (bool): 是否为预训练阶段
            
        Returns:
            dict: 包含特征和目标变量的字典，键为'features'和'target'
        """
        paths = self.global_paths if is_global else self.local_paths
        shape = self.global_shape if is_global else self.local_shape
        lon_lat = self.lon_lat_global if is_global else self.lon_lat_local
        static_data = self.static_data_global if is_global else self.static_data_local
        
        if is_pretrain:
            start_date = datetime(target_date.year, target_date.month, 1)
            end_date = start_date.replace(month=start_date.month % 12 + 1, day=1) - timedelta(hours=1)
        else:
            # 微调时间窗口为目标日期前后共30天，并包含最后一天的所有小时
            start_date = target_date - timedelta(days=15)
            end_date = target_date + timedelta(days=14, hours=23)
            
        logger.info(f"加载{'全局' if is_global else '局部'}数据，阶段：{'预训练' if is_pretrain else '微调'}")
        logger.info(f"时间范围：{start_date.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

        all_hourly_data = []
        
        dynamic_file_lists = {}
        dynamic_vars = []
        if is_global:
            dynamic_vars = ['cldas', 'era5', 'smap', 'ndvi']
        else:
            dynamic_vars = ['modis', 'era5_1km', 'smap_1km', 'ndvi_1km']

        for var_type in dynamic_vars:
            path_key = var_type
            if path_key in paths:
                dynamic_file_lists[var_type] = self.get_file_list(paths[path_key], start_date, end_date, var_type)
        
        # 修正: 只处理有目标数据的时间步
        target_var_type = 'cldas' if is_global else 'modis'
        if target_var_type not in dynamic_file_lists:
             logger.error(f"目标数据类型 {target_var_type} 的文件列表为空，无法继续。")
             return None
        
        # 提取所有目标数据的时间点进行循环
        target_file_list = dynamic_file_lists[target_var_type]
        
        # 定义动态数据的初始时间窗口
        max_deltas = {
            'cldas': timedelta(hours=1),
            'era5': timedelta(hours=1),
            'smap': timedelta(hours=3),
            'ndvi': timedelta(days=16),
            'modis': timedelta(hours=1),
            'era5_1km': timedelta(hours=1),
            'smap_1km': timedelta(hours=3),
            'ndvi_1km': timedelta(days=16)
        }

        for target_file_path, current_time in target_file_list:
            hourly_features = {}
            
            hourly_target_data = self._load_dynamic_data(
                target_file_path,
                self.var_names[target_var_type],
                shape,
                self.nodata_values.get(target_var_type),
                data_type=target_var_type
            )

            # 检查目标数据是否有效
            if hourly_target_data is None:
                continue

            valid_indices = ~np.isnan(hourly_target_data)
            
            # 针对全局和局部数据，如果nodata_value存在，也将其视为无效
            if is_global and self.nodata_values.get('cldas') is not None:
                valid_indices = valid_indices & (hourly_target_data != self.nodata_values['cldas'])
            if not is_global and self.nodata_values.get('modis') is not None:
                valid_indices = valid_indices & (hourly_target_data != self.nodata_values['modis'])
            
            num_valid_pixels = np.sum(valid_indices)

            # 修正：如果有效像元数为0，不跳过，直接创建空 DataFrame
            if num_valid_pixels == 0:
                logger.warning(f"时间步 {current_time.strftime('%Y-%m-%d %H:%M:%S')} 无有效像元数据。")
                all_hourly_data.append(pd.DataFrame())
                continue
            
            logger.info(f"时间步 {current_time.strftime('%Y-%m-%d %H:%M:%S')} 找到 {num_valid_pixels} 个有效像元。")

            hourly_features[self.var_names[target_var_type]] = hourly_target_data[valid_indices]
            
            # 修正: 统一动态变量的加载和键名
            dynamic_vars_to_load = ['era5', 'smap', 'ndvi']
            for var_type in dynamic_vars_to_load:
                path_key = var_type
                if not is_global: path_key = var_type + '_1km'
                
                if path_key in dynamic_file_lists:
                    # 使用动态时间窗口
                    file_path = self.find_closest_file(dynamic_file_lists[path_key], current_time, max_deltas[path_key])
                    if file_path:
                        var_names_list = self.var_names.get(var_type, [])
                        if isinstance(var_names_list, str): var_names_list = [var_names_list]
                        for var_name in var_names_list:
                            data = self._load_dynamic_data(file_path, var_name, shape, self.nodata_values.get(var_type), data_type=path_key)
                            if data is not None:
                                hourly_features[var_name] = data[valid_indices]

            # 静态数据现在不包含 ndvi
            static_vars = static_data.keys()
            for var_name in static_vars:
                if var_name in static_data:
                    hourly_features[var_name] = static_data[var_name][valid_indices]
            

            # 添加空间和时间特征
            hourly_features['lon'] = lon_lat[valid_indices, 0]
            hourly_features['lat'] = lon_lat[valid_indices, 1]

            # --- 地理编码 ---
            lon_rad = np.radians(hourly_features['lon'])
            lat_rad = np.radians(hourly_features['lat'])
            hourly_features['lon_sin'] = np.sin(lon_rad)
            hourly_features['lon_cos'] = np.cos(lon_rad)
            hourly_features['lat_sin'] = np.sin(lat_rad)
            hourly_features['lat_cos'] = np.cos(lat_rad)

            hourly_features['month'] = np.full(num_valid_pixels, current_time.month)
            hourly_features['day'] = np.full(num_valid_pixels, current_time.day)
            hourly_features['hour'] = np.full(num_valid_pixels, current_time.hour)
            hourly_features['doy'] = np.full(num_valid_pixels, current_time.timetuple().tm_yday)
            
            all_hourly_data.append(pd.DataFrame(hourly_features))
            
        if not all_hourly_data:
            logger.error("在指定的时间范围内没有找到任何有效数据。")
            return None
            
        final_df = pd.concat(all_hourly_data, ignore_index=True)
        
        # 将最终的DataFrame转换为字典格式
        target_var_name = self.var_names['cldas'] if is_global else self.var_names['modis']
        
        if target_var_name not in final_df.columns:
            logger.error(f"最终数据中不包含目标变量 {target_var_name}。")
            return None
            
        target_dict = {target_var_name: final_df[target_var_name].values}
        
        feature_cols = [col for col in final_df.columns if col != target_var_name]
        features_dict = {col: final_df[col].values for col in feature_cols}
        
        return {'features': features_dict, 'target': target_dict}
    def load_data_for_interpolation(self, target_date):
        """
        专门用于插补的数据加载方法，保留完整空间网格结构，避免内存问题
        """
        paths = self.local_paths
        shape = self.local_shape
        lon_lat = self.lon_lat_local
        static_data = self.static_data_local
        
        # 微调时间窗口：目标日期 ±15天
        start_date = target_date - timedelta(days=15)
        end_date = target_date + timedelta(days=14, hours=23)
            
        logger.info(f"加载局部插补数据")
        logger.info(f"时间范围：{start_date.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

        # 获取动态变量的文件列表
        dynamic_file_lists = {}
        dynamic_vars = ['modis', 'era5_1km', 'smap_1km', 'ndvi_1km']

        for var_type in dynamic_vars:
            if var_type in paths:
                dynamic_file_lists[var_type] = self.get_file_list(paths[var_type], start_date, end_date, var_type)
        
        target_var_type = 'modis'
        if target_var_type not in dynamic_file_lists:
            logger.error(f"目标数据类型 {target_var_type} 文件列表为空")
            return None
        
        target_file_list = dynamic_file_lists[target_var_type]
        
        max_deltas = {
            'era5_1km': timedelta(hours=1),
            'smap_1km': timedelta(hours=3),
            'ndvi_1km': timedelta(days=16),
            'modis': timedelta(hours=1)
        }

        # 预分配数组而不是使用DataFrame - 避免内存爆炸
        total_pixels = np.prod(shape)
        num_timesteps = len(target_file_list)
        
        logger.info(f"预期数据大小: {num_timesteps} 时间步 × {total_pixels} 像元 = {num_timesteps * total_pixels:,} 总数据点")
        
        # 检查内存需求（粗略估算）
        estimated_memory_gb = (num_timesteps * total_pixels * 24 * 23) / (1024**3)  # 假设20个特征，每个4字节
        if estimated_memory_gb > 16:  # 如果超过16GB
            logger.warning(f"估算内存需求: {estimated_memory_gb:.1f} GB，可能导致内存不足")
            # 可以选择缩小时间窗口或其他策略
            return None

        # 初始化特征字典
        all_features = {}
        all_targets = []
        
        # 预加载静态数据和坐标（这些对所有时间步都相同）
        static_features = {}
        total_pixels = np.prod(shape)
        
        # 静态数据
        for var_name in static_data.keys():
            if var_name in static_data:
                static_features[var_name] = static_data[var_name]
        
        # 空间特征
        static_features['lon'] = lon_lat[:, 0]
        static_features['lat'] = lon_lat[:, 1]
        lon_rad = np.radians(static_features['lon'])
        lat_rad = np.radians(static_features['lat'])
        static_features['lon_sin'] = np.sin(lon_rad)
        static_features['lon_cos'] = np.cos(lon_rad)
        static_features['lat_sin'] = np.sin(lat_rad)
        static_features['lat_cos'] = np.cos(lat_rad)

        # 逐时间步加载数据
        for timestep, (target_file_path, current_time) in enumerate(target_file_list):
            logger.info(f"处理时间步 {timestep+1}/{num_timesteps}: {current_time}")
            
            # 加载目标数据
            hourly_target_data = self._load_dynamic_data(
                target_file_path,
                self.var_names[target_var_type],
                shape,
                self.nodata_values.get(target_var_type),
                data_type=target_var_type
            )

            if hourly_target_data is None:
                continue

            # 目标数据
            all_targets.append(hourly_target_data)
            
            # 动态特征
            for var_type in ['era5_1km', 'smap_1km', 'ndvi_1km']:
                if var_type in dynamic_file_lists:
                    file_path = self.find_closest_file(dynamic_file_lists[var_type], current_time, max_deltas[var_type])
                    if file_path:
                        var_names_list = self.var_names.get(var_type.replace('_1km', ''), [])
                        if isinstance(var_names_list, str): 
                            var_names_list = [var_names_list]
                        for var_name in var_names_list:
                            data = self._load_dynamic_data(file_path, var_name, shape, self.nodata_values.get(var_type.replace('_1km', '')), data_type=var_type)
                            if data is not None:
                                if var_name not in all_features:
                                    all_features[var_name] = []
                                all_features[var_name].append(data)
            
            # 时间特征
            for time_var, time_val in [('month', current_time.month), ('day', current_time.day), 
                                    ('hour', current_time.hour), ('doy', current_time.timetuple().tm_yday)]:
                if time_var not in all_features:
                    all_features[time_var] = []
                all_features[time_var].append(np.full(total_pixels, time_val))
        
        if not all_targets:
            logger.error("没有找到有效的目标数据")
            return None
        
        # 拼接所有数据
        logger.info("拼接所有时间步数据...")
        final_features = {}
        
        # 静态特征：重复到所有时间步
        for var_name, data in static_features.items():
            final_features[var_name] = np.tile(data, len(all_targets))
        
        # 动态特征：直接拼接
        for var_name, data_list in all_features.items():
            if data_list:
                final_features[var_name] = np.concatenate(data_list)
        
        # 目标数据
        target_var_name = self.var_names['modis']
        final_targets = np.concatenate(all_targets)
        
        return {
            'features': final_features,
            'target': {target_var_name: final_targets},
            'spatial_shape': shape,
            'time_steps': len(all_targets)
        }

if __name__ == "__main__":
    logger.info("开始调试 DataLoader 模块，使用实际数据")
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    
    try:
        loader = DataLoader(config_path)
        target_date = datetime(2018, 1, 15)
        '''
        
        logger.info("\n--- 调试全局预训练数据加载 ---")
        pretrain_data = loader.load_data(target_date, is_global=True, is_pretrain=True)
        if pretrain_data is not None:
            logger.info(f"全局预训练数据加载成功。特征数量: {len(pretrain_data['features'])}，样本总数: {len(pretrain_data['features']['lat'])}")
            
        logger.info("\n--- 调试全局微调数据加载 ---")
        global_finetune_data = loader.load_data(target_date, is_global=True, is_pretrain=False)
        if global_finetune_data is not None:
            logger.info(f"全局微调数据加载成功。特征数量: {len(global_finetune_data['features'])}，样本总数: {len(global_finetune_data['features']['lat'])}")
            '''

        logger.info("\n--- 调试局部微调数据加载 ---")
        local_finetune_data = loader.load_data(target_date, is_global=False, is_pretrain=False)
        if local_finetune_data is not None:
            logger.info(f"局部微调数据加载成功。特征数量: {len(local_finetune_data['features'])}，样本总数: {len(local_finetune_data['features']['lat'])}")
            
    except FileNotFoundError as e:
        logger.error(f"调试失败: {str(e)}")
    except Exception as e:
        logger.error(f"调试失败，未知错误: {str(e)}")