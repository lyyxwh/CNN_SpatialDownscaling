import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import netCDF4 as nc
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
from torch.utils.data import DataLoader

from data_loader import DataLoader as LSTDataLoader
from preprocess import LSTDataset, DataCollator
from model import LSTTransformer, create_model_config

# 配置日志
log_dir = os.path.join(os.path.dirname(__file__), '../output/inference')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'inference.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class LSTInference:
    def __init__(self, config_path, global_finetune_dir, local_finetune_dir, output_dir):
        self.config_path = config_path
        self.global_finetune_dir = global_finetune_dir
        self.local_finetune_dir = local_finetune_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"无法加载配置文件 {config_path}: {e}")
            raise
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        self.data_loader = LSTDataLoader(config_path)
        self.local_shape = tuple(self.config['data_params']['local_shape'])  # (1921, 1801)
        self.min_coverage = self.config['data_params'].get('min_coverage', 0.5)
    
    def spatial_interpolate(self, lst, mask, ndvi):
        """使用周围像元进行空间插补（当NDVI可用时）"""
        def mean_filter(values):
            valid = values[~np.isnan(values)]
            return np.mean(valid) if len(valid) > 0 else np.nan
        
        lst_2d = lst.reshape(self.local_shape)
        ndvi_2d = ndvi.reshape(self.local_shape)
        mask_2d = mask.reshape(self.local_shape)
        
        # 仅对LST缺失且NDVI有效的像元进行插补
        interp_mask = (mask_2d == 1) & (~np.isnan(ndvi_2d))
        if np.any(interp_mask):
            # 使用3x3窗口的均值滤波
            lst_interp = generic_filter(lst_2d, mean_filter, size=3, mode='constant', cval=np.nan)
            lst_2d[interp_mask] = lst_interp[interp_mask]
        
        return lst_2d.flatten()
    
    def infer_daily(self, target_date, is_debug=False):
        date_str = target_date.strftime('%Y%m%d')
        month_str = target_date.strftime('%Y%m')
        logger.info(f"开始 {date_str} 的LST时空插补")
        
        # 加载模型
        global_model_path = os.path.join(self.global_finetune_dir, f"global_finetune_{date_str}.pt")
        local_model_path = os.path.join(self.local_finetune_dir, f"local_finetune_{date_str}.pt")
        pretrain_config_path = os.path.join(self.global_finetune_dir.replace('global_finetune', 'global_pretrain'), f"global_pretrain_{month_str}.pt")
        
        if not os.path.exists(pretrain_config_path):
            logger.error(f"预训练配置文件 {pretrain_config_path} 不存在，跳过 {date_str}")
            return False
        
        try:
            checkpoint = torch.load(pretrain_config_path, map_location=self.device, weights_only=False)
            model_config = checkpoint['config']
            scaler_dict = checkpoint['scaler_dict']
        except Exception as e:
            logger.error(f"加载预训练模型失败 {pretrain_config_path}: {e}")
            return False
        
        model = LSTTransformer(model_config).to(self.device)
        if os.path.exists(local_model_path):
            model.load_state_dict(torch.load(local_model_path, map_location=self.device))
            logger.info(f"加载局部fine-tune模型: {local_model_path}")
        elif os.path.exists(global_model_path):
            model.load_state_dict(torch.load(global_model_path, map_location=self.device))
            logger.info(f"加载全局fine-tune模型: {global_model_path}")
        else:
            logger.warning(f"无fine-tune模型，使用预训练模型")
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        
        # 加载数据
        try:
            data = self.data_loader.load_data(target_date, is_global=False, is_pretrain=False)
        except Exception as e:
            logger.error(f"数据加载失败 for {date_str}: {e}")
            data = None
        
        if data is None or not data.get('hourly_data'):
            logger.error(f"数据加载失败或hourly_data为空，跳过 {date_str}")
            for var_type in ['modis', 'era5_1km', 'smap_1km', 'ndvi_1km', 'cldas']:
                path = self.config['local_data_paths'].get(var_type, '')
                files = self.data_loader.get_file_list(path, target_date, target_date, var_type)
                logger.info(f"{var_type} 文件列表: {len(files)} 个文件")
            return False
        
        embedding_dims = {name: {'embed_dim': dim} for name, dim in model_config['cat_features'].items()}
        
        hours = range(24) if not is_debug else range(3)
        success_count = 0
        for hour in hours:
            start_time = time.time()
            hourly_data = next((h for h in data['hourly_data'] if h['hour'] == hour), None)
            if hourly_data is None:
                logger.warning(f"小时 {hour} 无数据，跳过")
                continue
            
            target = hourly_data['target']
            nan_ratio = np.isnan(target).mean()
            logger.info(f"小时 {hour} MODIS LST NaN比例: {nan_ratio:.2%}")
            if nan_ratio > (1 - self.min_coverage):
                logger.warning(f"小时 {hour} 数据覆盖率过低 ({1-nan_ratio:.2%} < {self.min_coverage})，跳过")
                continue
            
            try:
                # 检查辅助特征缺失情况
                features = hourly_data['features']
                ndvi = features.get('_1_km_16_days_NDVI')
                feature_mask = np.array([np.isnan(f).any() if f is not None else True for f in features.values()]).any(axis=0)
                
                dataset = LSTDataset(
                    features, 
                    {'LST_1km': target}, 
                    scaler_dict=scaler_dict,
                    embedding_dims=embedding_dims,
                    is_train=False
                )
                
                collator = DataCollator(dataset.get_embedding_layers())
                dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, collate_fn=collator)
                
                # 推理
                all_outputs = []
                with torch.no_grad():
                    for batch in tqdm(dataloader, desc=f"小时 {hour} 推理"):
                        batch_data = {
                            'num': batch['num_features'].to(self.device),
                            'time': batch['time_features'].to(self.device),
                            'loc': batch['loc_features'].to(self.device),
                            'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                        }
                        outputs = model(batch_data).cpu().numpy()
                        all_outputs.extend(outputs)
                
                predicted_lst = scaler_dict['target'].inverse_transform(np.array(all_outputs).reshape(-1, 1)).flatten()
                original_lst = target
                mask = np.isnan(original_lst).astype(np.int8)
                
                # 插补策略
                lst_filled = np.where(mask, predicted_lst, original_lst)
                
                # NDVI插补：LST缺失且其他特征缺失但NDVI有效
                if ndvi is not None:
                    ndvi_valid = ~np.isnan(ndvi)
                    interp_mask = (mask == 1) & feature_mask & ndvi_valid
                    if np.any(interp_mask):
                        logger.info(f"小时 {hour} 使用NDVI辅助空间插补，像元数: {np.sum(interp_mask)}")
                        lst_filled = self.spatial_interpolate(lst_filled, interp_mask, ndvi)
                
                # nodata=0：LST和NDVI均缺失
                no_data_mask = (mask == 1) & (np.isnan(ndvi) if ndvi is not None else True)
                if np.any(no_data_mask):
                    logger.info(f"小时 {hour} 设置nodata=0，像元数: {np.sum(no_data_mask)}")
                    lst_filled[no_data_mask] = 0
                
                # 上采样CLDAS
                cldas_file = os.path.join(self.config['local_data_paths'].get('cldas', ''), f"CLDAS_{date_str}_{hour:02d}.nc")
                if os.path.exists(cldas_file):
                    with nc.Dataset(cldas_file) as ds:
                        cldas_lst = ds['TG'][:].filled(np.nan)
                        cldas_lon = ds['lon'][:]
                        cldas_lat = ds['lat'][:]
                    
                    lon_grid, lat_grid = np.meshgrid(
                        np.linspace(self.config['data_params']['lon_range'][0], self.config['data_params']['lon_range'][1], self.local_shape[1]),
                        np.linspace(self.config['data_params']['lat_range'][0], self.config['data_params']['lat_range'][1], self.local_shape[0])
                    )
                    lst_global_upsampled = griddata(
                        (cldas_lon.flatten(), cldas_lat.flatten()), 
                        cldas_lst.flatten(), 
                        (lon_grid, lat_grid), 
                        method='linear'
                    ).flatten()
                else:
                    logger.warning(f"CLDAS文件 {cldas_file} 不存在，使用NaN填充")
                    lst_global_upsampled = np.full_like(lst_filled, np.nan)
                
                # 评估
                known_mask = ~np.isnan(original_lst)
                coverage = np.sum(known_mask) / len(original_lst)
                metrics = {}
                if np.sum(known_mask) > 0:
                    true_lst = original_lst[known_mask]
                    pred_lst = lst_filled[known_mask]
                    metrics['rmse'] = np.sqrt(mean_squared_error(true_lst, pred_lst))
                    metrics['r2'] = r2_score(true_lst, pred_lst)
                    metrics['mae'] = mean_absolute_error(true_lst, pred_lst)
                    metrics['spatial_corr'], _ = pearsonr(true_lst, pred_lst)
                    
                    logger.info(f"小时 {hour} MODIS评估: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}, "
                               f"MAE={metrics['mae']:.2f}, SpatialCorr={metrics['spatial_corr']:.4f}, 覆盖率={coverage:.2%}")
                    
                    if coverage < self.min_coverage:
                        logger.info(f"MODIS覆盖率低 ({coverage:.2%})，尝试CLDAS/NDVI验证")
                        valid_cldas = ~np.isnan(lst_global_upsampled)
                        if np.sum(valid_cldas) > 0:
                            cldas_metrics = {
                                'rmse': np.sqrt(mean_squared_error(lst_global_upsampled[valid_cldas], lst_filled[valid_cldas])),
                                'r2': r2_score(lst_global_upsampled[valid_cldas], lst_filled[valid_cldas]),
                                'mae': mean_absolute_error(lst_global_upsampled[valid_cldas], lst_filled[valid_cldas]),
                                'spatial_corr': pearsonr(lst_global_upsampled[valid_cldas], lst_filled[valid_cldas])[0]
                            }
                            logger.info(f"小时 {hour} CLDAS验证: RMSE={cldas_metrics['rmse']:.2f}, R2={cldas_metrics['r2']:.4f}, "
                                       f"MAE={cldas_metrics['mae']:.2f}, SpatialCorr={cldas_metrics['spatial_corr']:.4f}")
                        
                        if ndvi is not None and not np.all(np.isnan(ndvi)):
                            valid_ndvi = ~np.isnan(ndvi)
                            ndvi_metrics = {
                                'rmse': np.sqrt(mean_squared_error(ndvi[valid_ndvi], lst_filled[valid_ndvi])),
                                'r2': r2_score(ndvi[valid_ndvi], lst_filled[valid_ndvi]),
                                'mae': mean_absolute_error(ndvi[valid_ndvi], lst_filled[valid_ndvi]),
                                'spatial_corr': pearsonr(ndvi[valid_ndvi], lst_filled[valid_ndvi])[0]
                            }
                            logger.info(f"小时 {hour} NDVI验证: RMSE={ndvi_metrics['rmse']:.2f}, R2={ndvi_metrics['r2']:.4f}, "
                                       f"MAE={ndvi_metrics['mae']:.2f}, SpatialCorr={ndvi_metrics['spatial_corr']:.4f}")
                    
                    self.plot_scatter(true_lst, pred_lst, self.output_dir, f"{date_str}_{hour:02d}")
                
                # 保存NetCDF
                self.save_netcdf(lst_filled, lst_global_upsampled, mask, date_str, hour, metrics)
                success_count += 1
                
                elapsed = time.time() - start_time
                logger.info(f"小时 {hour} 处理完成，耗时: {elapsed:.1f}s")
            except Exception as e:
                logger.error(f"小时 {hour} 处理失败: {e}")
                continue
        
        logger.info(f"{date_str} 处理完成，成功处理 {success_count}/{len(hours)} 小时")
        return success_count > 0
    
    def save_netcdf(self, lst, lst_upsampled, mask, date_str, hour, metrics):
        file_name = os.path.join(self.output_dir, f"lst_{date_str}_{hour:02d}.nc")
        height, width = self.local_shape
        lst_2d = lst.reshape(height, width)
        upsampled_2d = lst_upsampled.reshape(height, width)
        mask_2d = mask.reshape(height, width)
        
        try:
            with nc.Dataset(file_name, 'w', format='NETCDF4') as ds:
                lat_dim = ds.createDimension('lat', height)
                lon_dim = ds.createDimension('lon', width)
                lats = ds.createVariable('lat', np.float32, ('lat',))
                lons = ds.createVariable('lon', np.float32, ('lon',))
                lst_var = ds.createVariable('lst', np.float32, ('lat', 'lon'), zlib=True)
                upsampled_var = ds.createVariable('lst_global_upsampled', np.float32, ('lat', 'lon'), zlib=True)
                mask_var = ds.createVariable('mask', np.int8, ('lat', 'lon'), zlib=True)
                
                lon_range = self.config['data_params']['lon_range']
                lat_range = self.config['data_params']['lat_range']
                lons[:] = np.linspace(lon_range[0], lon_range[1], width)
                lats[:] = np.linspace(lat_range[0], lat_range[1], height)
                
                lst_var[:] = lst_2d
                upsampled_var[:] = upsampled_2d
                mask_var[:] = mask_2d
                
                ds.title = f"LST Interpolation for {date_str} {hour:02d}:00"
                lst_var.units = 'Kelvin'
                mask_var.description = '1: missing, 0: known'
                for metric, value in metrics.items():
                    ds.setncattr(metric, f"{value:.4f}")
            
            logger.info(f"NetCDF保存到: {file_name}")
        except Exception as e:
            logger.error(f"保存NetCDF失败 {file_name}: {e}")
    
    def plot_scatter(self, true_lst, pred_lst, output_dir, file_prefix):
        try:
            plt.figure(figsize=(8, 8))
            plt.scatter(true_lst, pred_lst, alpha=0.5, s=1)
            
            min_val = min(true_lst.min(), pred_lst.min())
            max_val = max(true_lst.max(), pred_lst.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            rmse = np.sqrt(mean_squared_error(true_lst, pred_lst))
            r2 = r2_score(true_lst, pred_lst)
            mae = mean_absolute_error(true_lst, pred_lst)
            
            plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes)
            plt.text(0.05, 0.90, f'$R^2$: {r2:.4f}', transform=plt.gca().transAxes)
            plt.text(0.05, 0.85, f'MAE: {mae:.2f}', transform=plt.gca().transAxes)
            
            plt.title('Predicted LST vs True LST')
            plt.xlabel('True LST (K)')
            plt.ylabel('Predicted LST (K)')
            plt.grid(True)
            
            scatter_path = os.path.join(output_dir, f"scatter_{file_prefix}.png")
            plt.savefig(scatter_path, dpi=300)
            plt.close()
            logger.info(f"散点图保存到: {scatter_path}")
        except Exception as e:
            logger.error(f"生成散点图失败 {file_prefix}: {e}")

if __name__ == "__main__":
    logger.info("开始LST时空插补推理")
    
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    global_finetune_dir = r"G:\CNN_SpatialDownscaling\output\global_finetune"
    local_finetune_dir = r"G:\CNN_SpatialDownscaling\output\local_finetune"
    output_dir = r"G:\CNN_SpatialDownscaling\output\inference"
    
    try:
        inferencer = LSTInference(config_path, global_finetune_dir, local_finetune_dir, output_dir)
        
        # 全年处理
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2018, 12, 31)
        total_hours = 0
        successful_hours = 0
        for date in pd.date_range(start_date, end_date):
            success = inferencer.infer_daily(date, is_debug=False)
            if success:
                successful_hours += 24  # 假设成功处理全天
            total_hours += 24
        
        logger.info(f"全年LST插补完成！成功处理 {successful_hours}/{total_hours} 小时")
        logger.info(f"输出文件在: {output_dir}")
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)