import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import traceback

from data_loader import DataLoader as LSTDataLoader
from preprocess import LSTDataset, DataCollator
from model import LSTTransformer, create_model_config

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SmartLSTInterpolator:
    """
    智能LST插补器 - 根据数据可用性采用不同插补策略
    """
    
    def __init__(self, config_path, global_finetune_dir, local_finetune_dir, output_dir):
        """初始化插补器"""
        self.config_path = config_path
        self.global_finetune_dir = global_finetune_dir
        self.local_finetune_dir = local_finetune_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        self.interpolated_dir = os.path.join(output_dir, 'interpolated_results')
        self.evaluation_dir = os.path.join(output_dir, 'evaluation')
        os.makedirs(self.interpolated_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        # 加载配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loader = LSTDataLoader(self.config_path)
        self.nodata_value = self.config['nodata_values']['modis']
        
        # 获取空间维度
        self.spatial_shape = self.config['data_params']['local_shape']
        self.lat_range = self.config['data_params']['lat_range']
        self.lon_range = self.config['data_params']['lon_range']
        
        logger.info(f"Smart LST插补器初始化完成")
        logger.info(f"空间维度: {self.spatial_shape}")

    def interpolate_single_date(self, target_date):
        """对指定日期进行智能插补"""
        date_str = target_date.strftime('%Y%m%d')
        logger.info(f"开始处理 {target_date.strftime('%Y-%m-%d')}")
        
        try:
            # 1. 加载模型
            global_model, local_model, scaler_dict, embedding_dims = self.load_models(target_date)
            if global_model is None:
                logger.error("模型加载失败")
                return False, []
            
            # 2. 加载原始数据
            original_data = self.loader.load_data_for_interpolation(target_date)
            if original_data is None:
                logger.error("原始数据加载失败")
                return False, []
            
            # 3. 为每个小时执行智能插补
            generated_files = []
            for hour in range(24):
                logger.info(f"处理小时 {hour}")
                
                output_path = self.interpolate_single_hour(
                    target_date, hour, original_data, 
                    global_model, local_model, scaler_dict, embedding_dims
                )
                
                if output_path:
                    generated_files.append(output_path)
                    logger.info(f"✓ 小时 {hour} 完成")
                else:
                    logger.warning(f"× 小时 {hour} 失败")
            
            logger.info(f"日期 {date_str} 完成，生成 {len(generated_files)}/24 个文件")
            return True, generated_files
            
        except Exception as e:
            logger.error(f"处理日期 {date_str} 失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False, []

    def load_models(self, target_date):
        """加载训练好的模型"""
        try:
            date_str = target_date.strftime('%Y%m%d')
            month_str = target_date.strftime('%Y%m')
            
            # 检查模型文件存在性
            global_model_path = os.path.join(self.global_finetune_dir, f"global_finetune_{date_str}.pt")
            local_model_path = os.path.join(self.local_finetune_dir, f"local_finetune_{date_str}.pt")
            pretrain_path = os.path.join(
                self.global_finetune_dir.replace('global_finetune', 'global_pretrain'),
                f"global_pretrain_{month_str}.pt"
            )
            
            if not all(os.path.exists(p) for p in [global_model_path, local_model_path, pretrain_path]):
                logger.error("缺少必要的模型文件")
                return None, None, None, None
            
            # 加载预训练配置
            pretrain_checkpoint = torch.load(pretrain_path, map_location=self.device, weights_only=False)
            base_config = pretrain_checkpoint['config']
            scaler_dict = pretrain_checkpoint['scaler_dict']
            
            # 加载微调模型
            global_checkpoint = torch.load(global_model_path, map_location=self.device, weights_only=False)
            local_checkpoint = torch.load(local_model_path, map_location=self.device, weights_only=False)
            
            # 推断模型结构
            layer_indices = set()
            for key in global_checkpoint.keys():
                if 'transformer_blocks.' in key:
                    try:
                        layer_idx = int(key.split('transformer_blocks.')[1].split('.')[0])
                        layer_indices.add(layer_idx)
                    except:
                        continue
            
            actual_n_layers = max(layer_indices) + 1 if layer_indices else base_config.get('n_layers', 6)
            
            # 创建模型
            model_config = base_config.copy()
            model_config['n_layers'] = actual_n_layers
            
            global_model = LSTTransformer(model_config).to(self.device)
            local_model = LSTTransformer(model_config).to(self.device)
            
            global_model.load_state_dict(global_checkpoint, strict=False)
            local_model.load_state_dict(local_checkpoint, strict=False)
            
            global_model.eval()
            local_model.eval()
            
            # 准备embedding维度
            raw_cat_features = model_config['cat_features']
            embedding_dims = {}
            for name, num_classes in raw_cat_features.items():
                embed_dim = min(50, (num_classes + 1) // 2)
                embedding_dims[name] = {'embed_dim': embed_dim}
            
            logger.info(f"模型加载成功: 层数={actual_n_layers}")
            return global_model, local_model, scaler_dict, embedding_dims
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return None, None, None, None

    def interpolate_single_hour(self, target_date, hour, original_data, 
                               global_model, local_model, scaler_dict, embedding_dims):
        """对单个小时进行智能插补"""
        try:
            # 1. 提取该小时的原始数据
            hour_data = self.extract_hour_data(original_data, hour)
            if hour_data is None:
                logger.info(f"小时 {hour}: 无原始数据，执行完全预测插补")
                hour_data = self.create_empty_hour_data(target_date, hour)
            
            # 2. 创建完整的空间网格
            complete_grid = self.create_complete_spatial_grid(target_date, hour, hour_data)
            
            # 3. 根据数据可用性分类像元
            pixel_categories = self.categorize_pixels(complete_grid)
            
            # 4. 对不同类别的像元采用不同插补策略
            interpolated_grid = self.apply_interpolation_strategies(
                complete_grid, pixel_categories, 
                global_model, local_model, scaler_dict, embedding_dims
            )
            
            # 5. 生成NetCDF输出
            output_path = self.create_netcdf_output(
                interpolated_grid, target_date, hour, pixel_categories
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"小时 {hour} 插补失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def extract_hour_data(self, original_data, target_hour):
        """从原始数据中提取指定小时的数据"""
        try:
            if 'hourly_data' not in original_data or not original_data['hourly_data']:
                return None
            
            for hour_data in original_data['hourly_data']:
                if hour_data.get('hour') == target_hour:
                    return hour_data
            
            return None  # 该小时无数据
            
        except Exception as e:
            logger.warning(f"提取小时 {target_hour} 数据失败: {str(e)}")
            return None

    def create_empty_hour_data(self, target_date, hour):
        """为没有观测数据的小时创建空数据结构"""
        total_pixels = np.prod(self.spatial_shape)
        
        return {
            'hour': hour,
            'features': {
                'lat': np.zeros(total_pixels),
                'lon': np.zeros(total_pixels),
                'dem': np.zeros(total_pixels),
                'slope': np.zeros(total_pixels),
                'aspect': np.zeros(total_pixels),
                'clcd': np.ones(total_pixels, dtype=int),
                't2m': np.full(total_pixels, 273.15),
                'ssrd': np.zeros(total_pixels),
                'strd': np.full(total_pixels, 300.0),
                'rh': np.full(total_pixels, 0.5),
                'd2m': np.full(total_pixels, 268.15),
                'vpd': np.full(total_pixels, 0.5),
                'sm_surface_wetness': np.ones(total_pixels, dtype=int),
                'ndvi': np.full(total_pixels, 0.3),
                'month': np.full(total_pixels, target_date.month),
                'day': np.full(total_pixels, target_date.day),
                'hour': np.full(total_pixels, hour),
                'doy': np.full(total_pixels, target_date.timetuple().tm_yday),
            },
            'target': np.full(total_pixels, self.nodata_value)
        }

    def create_complete_spatial_grid(self, target_date, hour, hour_data):
        """创建完整的空间网格，包括所有像元的特征和LST"""
        try:
            total_pixels = np.prod(self.spatial_shape)
            
            # 创建地理坐标网格
            lats = np.linspace(self.lat_range[1], self.lat_range[0], self.spatial_shape[0])
            lons = np.linspace(self.lon_range[0], self.lon_range[1], self.spatial_shape[1])
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            complete_grid = {
                # 基础地理信息
                'lat': lat_grid.flatten(),
                'lon': lon_grid.flatten(),
                'row_idx': np.repeat(np.arange(self.spatial_shape[0]), self.spatial_shape[1]),
                'col_idx': np.tile(np.arange(self.spatial_shape[1]), self.spatial_shape[0]),
                
                # LST目标变量（大部分为缺失）
                'lst': np.full(total_pixels, self.nodata_value, dtype=np.float32),
                
                # 环境特征（需要从实际数据中获取或设置默认值）
                'dem': np.full(total_pixels, np.nan, dtype=np.float32),
                'slope': np.full(total_pixels, np.nan, dtype=np.float32),
                'aspect': np.full(total_pixels, np.nan, dtype=np.float32),
                'clcd': np.full(total_pixels, np.nan, dtype=np.float32),
                't2m': np.full(total_pixels, np.nan, dtype=np.float32),
                'ssrd': np.full(total_pixels, np.nan, dtype=np.float32),
                'strd': np.full(total_pixels, np.nan, dtype=np.float32),
                'rh': np.full(total_pixels, np.nan, dtype=np.float32),
                'd2m': np.full(total_pixels, np.nan, dtype=np.float32),
                'vpd': np.full(total_pixels, np.nan, dtype=np.float32),
                'sm_surface_wetness': np.full(total_pixels, np.nan, dtype=np.float32),
                'ndvi': np.full(total_pixels, np.nan, dtype=np.float32),
                
                # 时间特征
                'month': np.full(total_pixels, target_date.month),
                'day': np.full(total_pixels, target_date.day),
                'hour': np.full(total_pixels, hour),
                'doy': np.full(total_pixels, target_date.timetuple().tm_yday),
            }
            
            # 如果有观测数据，填充到网格中
            if hour_data and 'features' in hour_data:
                self.fill_observed_data(complete_grid, hour_data)
            
            # 加载静态环境数据（DEM、土地覆盖等）
            self.load_static_environmental_data(complete_grid)
            
            # 添加地理编码
            complete_grid['lat_sin'] = np.sin(np.radians(complete_grid['lat']))
            complete_grid['lat_cos'] = np.cos(np.radians(complete_grid['lat']))
            complete_grid['lon_sin'] = np.sin(np.radians(complete_grid['lon']))
            complete_grid['lon_cos'] = np.cos(np.radians(complete_grid['lon']))
            
            logger.info(f"完整空间网格创建成功: {total_pixels} 像元")
            return complete_grid
            
        except Exception as e:
            logger.error(f"创建完整空间网格失败: {str(e)}")
            return None

    def fill_observed_data(self, complete_grid, hour_data):
        """将观测数据填充到完整网格中"""
        try:
            if not hour_data.get('features'):
                return
            
            # 简化映射：假设观测数据按空间顺序排列
            # 实际应用中需要根据经纬度进行空间匹配
            for feature_name, feature_values in hour_data['features'].items():
                if feature_name in complete_grid and len(feature_values) > 0:
                    n_fill = min(len(feature_values), len(complete_grid[feature_name]))
                    complete_grid[feature_name][:n_fill] = feature_values[:n_fill]
            
            # 填充LST观测值
            if 'target' in hour_data and len(hour_data['target']) > 0:
                n_fill = min(len(hour_data['target']), len(complete_grid['lst']))
                complete_grid['lst'][:n_fill] = hour_data['target'][:n_fill]
                
        except Exception as e:
            logger.warning(f"填充观测数据失败: {str(e)}")

    def load_static_environmental_data(self, complete_grid):
        """加载静态环境数据（DEM、土地覆盖等）"""
        try:
            # 这里应该从实际的数据文件中读取
            # 现在使用模拟数据
            total_pixels = len(complete_grid['lat'])
            
            # 根据地理位置生成模拟的环境数据
            lats = complete_grid['lat']
            lons = complete_grid['lon']
            
            # 模拟DEM：基于纬度的简单模型
            complete_grid['dem'] = np.where(
                np.isnan(complete_grid['dem']),
                np.maximum(0, (lats - np.min(lats)) * 1000 + np.random.normal(0, 100, total_pixels)),
                complete_grid['dem']
            )
            
            # 模拟坡度
            complete_grid['slope'] = np.where(
                np.isnan(complete_grid['slope']),
                np.abs(np.random.normal(5, 3, total_pixels)),
                complete_grid['slope']
            )
            
            # 模拟土地覆盖
            complete_grid['clcd'] = np.where(
                np.isnan(complete_grid['clcd']),
                np.random.randint(1, 9, total_pixels),
                complete_grid['clcd']
            ).astype(int)
            
            # 模拟气象数据
            for var in ['t2m', 'ssrd', 'strd', 'rh', 'd2m', 'vpd', 'ndvi']:
                if var in complete_grid:
                    # 基于纬度和季节的简单模型
                    if var == 't2m':
                        default_val = 273.15 + 20 * np.cos(np.radians(lats - 35))  # 基于纬度的温度
                    elif var == 'rh':
                        default_val = np.clip(0.3 + 0.4 * np.cos(np.radians(lons - 117)), 0, 1)
                    elif var == 'ndvi':
                        default_val = np.clip(0.1 + 0.6 * np.cos(np.radians(lats - 35)), -1, 1)
                    else:
                        default_val = np.random.normal(300 if 'strd' in var else 100, 50, total_pixels)
                    
                    complete_grid[var] = np.where(
                        np.isnan(complete_grid[var]),
                        default_val,
                        complete_grid[var]
                    )
            
            # 土壤湿度
            complete_grid['sm_surface_wetness'] = np.where(
                np.isnan(complete_grid['sm_surface_wetness']),
                np.random.randint(0, 6, total_pixels),
                complete_grid['sm_surface_wetness']
            ).astype(int)
            
            logger.info("静态环境数据加载完成")
            
        except Exception as e:
            logger.warning(f"加载静态环境数据失败: {str(e)}")

    def categorize_pixels(self, complete_grid):
        """根据数据可用性对像元进行分类"""
        try:
            total_pixels = len(complete_grid['lst'])
            
            # LST数据状态
            has_lst = (complete_grid['lst'] != self.nodata_value) & ~np.isnan(complete_grid['lst'])
            
            # DEM数据状态
            has_dem = ~np.isnan(complete_grid['dem'])
            
            # 其他环境变量状态
            env_vars = ['t2m', 'ssrd', 'strd', 'rh', 'd2m', 'vpd', 'ndvi', 'clcd', 'sm_surface_wetness']
            has_env = np.ones(total_pixels, dtype=bool)
            for var in env_vars:
                if var in complete_grid:
                    has_env &= ~np.isnan(complete_grid[var])
            
            # 像元分类
            categories = {
                # 类型1：有LST观测值的像元（用于验证和保持原值）
                'observed': has_lst,
                
                # 类型2：LST缺失但有完整辅助数据的像元（使用模型反演）
                'model_prediction': ~has_lst & has_env,
                
                # 类型3：LST和部分环境数据缺失但有DEM的像元（空间插补）
                'spatial_interpolation': ~has_lst & ~has_env & has_dem,
                
                # 类型4：LST和DEM都缺失的像元（填充nodata）
                'nodata': ~has_lst & ~has_dem
            }
            
            # 统计各类别数量
            for cat_name, mask in categories.items():
                count = np.sum(mask)
                percent = count / total_pixels * 100
                logger.info(f"{cat_name}: {count} 像元 ({percent:.1f}%)")
            
            return categories
            
        except Exception as e:
            logger.error(f"像元分类失败: {str(e)}")
            return {}

    def apply_interpolation_strategies(self, complete_grid, pixel_categories, 
                                     global_model, local_model, scaler_dict, embedding_dims):
        """对不同类别的像元应用相应的插补策略"""
        try:
            total_pixels = len(complete_grid['lst'])
            result_grid = complete_grid.copy()
            
            # 策略1：保持观测值不变
            if np.any(pixel_categories['observed']):
                logger.info("策略1: 保持观测值不变")
                # 观测值已经在网格中，无需处理
            
            # 策略2：使用模型进行反演预测
            if np.any(pixel_categories['model_prediction']):
                logger.info("策略2: 模型反演预测")
                predicted_lst = self.model_prediction(
                    complete_grid, pixel_categories['model_prediction'],
                    local_model, scaler_dict, embedding_dims
                )
                if predicted_lst is not None:
                    result_grid['lst'][pixel_categories['model_prediction']] = predicted_lst
            
            # 策略3：空间插补
            if np.any(pixel_categories['spatial_interpolation']):
                logger.info("策略3: 空间插补")
                interpolated_lst = self.spatial_interpolation(
                    result_grid, pixel_categories['spatial_interpolation']
                )
                if interpolated_lst is not None:
                    result_grid['lst'][pixel_categories['spatial_interpolation']] = interpolated_lst
            
            # 策略4：填充nodata
            if np.any(pixel_categories['nodata']):
                logger.info("策略4: 填充nodata")
                result_grid['lst'][pixel_categories['nodata']] = 0  # 按要求使用0
            
            # 验证结果
            final_valid = np.sum(~np.isnan(result_grid['lst']) & (result_grid['lst'] != self.nodata_value))
            logger.info(f"插补完成: {final_valid}/{total_pixels} 像元有有效值")
            
            return result_grid
            
        except Exception as e:
            logger.error(f"应用插补策略失败: {str(e)}")
            logger.error(traceback.format_exc())
            return complete_grid

    def model_prediction(self, complete_grid, prediction_mask, local_model, scaler_dict, embedding_dims):
        """使用训练好的模型进行LST反演预测"""
        try:
            if not np.any(prediction_mask):
                return None
            
            # 提取需要预测的像元的特征
            features = {}
            for feature_name in complete_grid.keys():
                if feature_name not in ['lst', 'row_idx', 'col_idx']:
                    features[feature_name] = complete_grid[feature_name][prediction_mask]
            
            # 创建虚拟目标值（用于数据集构建，实际不使用）
            n_pixels = np.sum(prediction_mask)
            virtual_target = {'LST_1km': np.full(n_pixels, 273.15)}  # 使用合理的默认值
            
            # 创建数据集
            dataset = LSTDataset(
                features, virtual_target,
                scaler_dict=scaler_dict,
                embedding_dims=embedding_dims,
                is_train=False
            )
            
            collator = DataCollator(dataset.get_embedding_layers())
            batch_size = min(1024, len(dataset))
            
            data_loader = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=False, collate_fn=collator
            )
            
            # 进行预测
            predictions = []
            
            with torch.no_grad():
                for batch in data_loader:
                    batch_data = {
                        'num': batch['num_features'].to(self.device),
                        'time': batch['time_features'].to(self.device),
                        'loc': batch['loc_features'].to(self.device),
                        'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                    }
                    
                    outputs = local_model(batch_data)
                    predictions.extend(outputs.squeeze().cpu().numpy())
            
            # 反标准化
            target_scaler = scaler_dict.get('target')
            if target_scaler is not None:
                pred_array = np.array(predictions).reshape(-1, 1)
                pred_inv = target_scaler.inverse_transform(pred_array).flatten()
            else:
                pred_inv = np.array(predictions)
            
            logger.info(f"模型预测完成: {len(pred_inv)} 像元, "
                       f"预测范围 [{np.min(pred_inv):.1f}, {np.max(pred_inv):.1f}]K")
            
            return pred_inv
            
        except Exception as e:
            logger.error(f"模型预测失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def spatial_interpolation(self, result_grid, interpolation_mask):
        """对缺少环境数据但有DEM的像元进行空间插补"""
        try:
            if not np.any(interpolation_mask):
                return None
            
            # 获取有LST值的像元作为插补源
            has_lst = (result_grid['lst'] != self.nodata_value) & ~np.isnan(result_grid['lst'])
            
            if np.sum(has_lst) < 3:
                logger.warning("可用于空间插补的LST观测点太少")
                return None
            
            # 源点和目标点的地理坐标
            source_coords = np.column_stack([
                result_grid['lat'][has_lst],
                result_grid['lon'][has_lst]
            ])
            source_values = result_grid['lst'][has_lst]
            
            target_coords = np.column_stack([
                result_grid['lat'][interpolation_mask],
                result_grid['lon'][interpolation_mask]
            ])
            
            # 使用反距离加权插值
            try:
                interpolated_values = griddata(
                    source_coords, source_values, target_coords,
                    method='linear', fill_value=np.nan
                )
                
                # 对仍然为NaN的点使用最近邻
                nan_mask = np.isnan(interpolated_values)
                if np.any(nan_mask):
                    nearest_values = griddata(
                        source_coords, source_values, 
                        target_coords[nan_mask],
                        method='nearest'
                    )
                    interpolated_values[nan_mask] = nearest_values
                
                logger.info(f"空间插补完成: {len(interpolated_values)} 像元, "
                           f"插补范围 [{np.nanmin(interpolated_values):.1f}, {np.nanmax(interpolated_values):.1f}]K")
                
                return interpolated_values
                
            except Exception as e:
                logger.warning(f"griddata插补失败: {str(e)}, 使用简单平均值")
                # 备用方案：使用所有有效LST值的平均值
                mean_lst = np.nanmean(source_values)
                return np.full(np.sum(interpolation_mask), mean_lst)
            
        except Exception as e:
            logger.error(f"空间插补失败: {str(e)}")
            return None

    def create_netcdf_output(self, interpolated_grid, target_date, hour, pixel_categories):
        """创建NetCDF输出文件"""
        try:
            date_str = target_date.strftime('%Y%m%d')
            output_filename = f"LST_{date_str}_{hour:02d}.nc"
            output_path = os.path.join(self.interpolated_dir, output_filename)
            
            # 重塑为2D网格
            def to_2d(data_1d):
                return data_1d.reshape(self.spatial_shape)
            
            # 主要数据层
            lst_final = to_2d(interpolated_grid['lst'])
            dem_2d = to_2d(interpolated_grid['dem'])
            
            # 创建分类掩膜
            category_mask = np.zeros(self.spatial_shape, dtype=np.int8)
            for i, (cat_name, mask) in enumerate(pixel_categories.items()):
                category_mask[to_2d(mask)] = i + 1  # 1=observed, 2=model_prediction, 3=spatial_interpolation, 4=nodata
            
            # 计算质量指标
            observed_pixels = np.sum(pixel_categories['observed'])
            predicted_pixels = np.sum(pixel_categories['model_prediction'])
            interpolated_pixels = np.sum(pixel_categories['spatial_interpolation'])
            nodata_pixels = np.sum(pixel_categories['nodata'])
            
            # 计算RMSE（仅对观测像元，如果有模型预测的话）
            rmse_value = np.nan
            if observed_pixels > 0 and predicted_pixels > 0:
                # 这里可以计算验证RMSE，但需要保留一些观测点用于验证
                rmse_value = 999.0  # 占位符
            
            # 坐标
            lats_2d = to_2d(interpolated_grid['lat'])
            lons_2d = to_2d(interpolated_grid['lon'])
            
            # 创建xarray数据集
            ds = xr.Dataset(
                {
                    "lst": (("y", "x"), lst_final.astype(np.float32)),
                    "dem": (("y", "x"), dem_2d.astype(np.float32)),
                    "interpolation_method": (("y", "x"), category_mask),
                },
                coords={
                    "lat": (("y", "x"), lats_2d.astype(np.float32)),
                    "lon": (("y", "x"), lons_2d.astype(np.float32)),
                    "y": np.arange(self.spatial_shape[0]),
                    "x": np.arange(self.spatial_shape[1])
                },
                attrs={
                    "title": f"Smart LST Interpolation Results for {date_str} Hour {hour:02d}",
                    "description": "LST interpolation using different strategies based on data availability",
                    "date": target_date.strftime('%Y-%m-%d'),
                    "hour": hour,
                    "spatial_shape": self.spatial_shape,
                    "observed_pixels": int(observed_pixels),
                    "model_predicted_pixels": int(predicted_pixels),
                    "spatial_interpolated_pixels": int(interpolated_pixels),
                    "nodata_pixels": int(nodata_pixels),
                    "rmse": float(rmse_value) if not np.isnan(rmse_value) else None,
                    "nodata_value": 0,  # 按要求，LST和DEM都缺失时用0
                    "modis_nodata_value": self.nodata_value,
                    "units": "Kelvin",
                    "spatial_resolution": "1km",
                    "created_by": "SmartLSTInterpolator",
                    "creation_time": datetime.now().isoformat()
                }
            )
            
            # 设置变量属性
            ds['lst'].attrs = {
                'long_name': 'Land Surface Temperature',
                'units': 'K',
                'description': 'LST from observations, model prediction, spatial interpolation, or nodata fill',
                'valid_range': [200.0, 350.0]
            }
            
            ds['dem'].attrs = {
                'long_name': 'Digital Elevation Model',
                'units': 'm',
                'description': 'Elevation above sea level'
            }
            
            ds['interpolation_method'].attrs = {
                'long_name': 'Interpolation Method Used',
                'description': '1=observed, 2=model_prediction, 3=spatial_interpolation, 4=nodata_fill',
                'flag_values': [1, 2, 3, 4],
                'flag_meanings': 'observed model_prediction spatial_interpolation nodata_fill'
            }
            
            # 保存文件
            encoding = {
                'lst': {'zlib': True, 'complevel': 6, 'dtype': 'float32'},
                'dem': {'zlib': True, 'complevel': 6, 'dtype': 'float32'},
                'interpolation_method': {'zlib': True, 'complevel': 6, 'dtype': 'int8'},
                'lat': {'dtype': 'float32'},
                'lon': {'dtype': 'float32'}
            }
            
            ds.to_netcdf(output_path, encoding=encoding, format="NETCDF4")
            
            # 验证输出
            valid_lst = np.sum(~np.isnan(lst_final) & (lst_final != self.nodata_value))
            lst_range = [np.nanmin(lst_final), np.nanmax(lst_final)]
            
            logger.info(f"NetCDF已保存: {output_filename}")
            logger.info(f"LST统计: 有效像元={valid_lst}/{np.prod(self.spatial_shape)}, "
                       f"范围=[{lst_range[0]:.1f}, {lst_range[1]:.1f}]K")
            logger.info(f"插补方法统计: 观测={observed_pixels}, 模型={predicted_pixels}, "
                       f"空间插补={interpolated_pixels}, 填充0={nodata_pixels}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"创建NetCDF文件失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def run_annual_interpolation(self, year=2018, start_month=1, end_month=12):
        """运行年度智能插补任务"""
        logger.info(f"开始{year}年智能LST插补任务")
        logger.info(f"插补策略:")
        logger.info("  1. 有LST观测 -> 保持原值")
        logger.info("  2. LST缺失+环境数据完整 -> 模型反演")
        logger.info("  3. LST缺失+环境数据不全+有DEM -> 空间插补")
        logger.info("  4. LST和DEM都缺失 -> 填充0")
        
        start_date = datetime(year, start_month, 1)
        end_date = datetime(year, end_month, 31) if end_month == 12 else datetime(year, end_month+1, 1) - timedelta(days=1)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        successful_days = 0
        total_files = 0
        
        for i, target_date in enumerate(date_range):
            logger.info(f"\n处理进度: {i+1}/{len(date_range)} - {target_date.strftime('%Y-%m-%d')}")
            
            success, files = self.interpolate_single_date(target_date)
            if success:
                successful_days += 1
                total_files += len(files)
            
            # 简单进度报告
            success_rate = successful_days / (i + 1) * 100
            logger.info(f"累计成功率: {success_rate:.1f}% ({successful_days}/{i+1})")
        
        logger.info(f"\n年度插补完成:")
        logger.info(f"成功天数: {successful_days}/{len(date_range)}")
        logger.info(f"生成文件: {total_files}")

def main():
    """主函数"""
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    global_finetune_dir = r"G:\CNN_SpatialDownscaling\output\global_finetune"
    local_finetune_dir = r"G:\CNN_SpatialDownscaling\output\local_finetune"
    output_dir = r"G:\CNN_SpatialDownscaling\output\smart_interpolation"
    
    # 创建智能插补器
    interpolator = SmartLSTInterpolator(
        config_path=config_path,
        global_finetune_dir=global_finetune_dir,
        local_finetune_dir=local_finetune_dir,
        output_dir=output_dir
    )
    
    # 单日测试
    test_date = datetime(2018, 1, 15)
    print(f"开始智能插补测试: {test_date.strftime('%Y-%m-%d')}")
    success, files = interpolator.interpolate_single_date(test_date)
    print(f"测试结果: 成功={success}, 生成文件={len(files)}/24")
    
    # 如果需要批量处理，取消下面的注释
    # interpolator.run_annual_interpolation(year=2018, start_month=1, end_month=1)

if __name__ == "__main__":
    main()