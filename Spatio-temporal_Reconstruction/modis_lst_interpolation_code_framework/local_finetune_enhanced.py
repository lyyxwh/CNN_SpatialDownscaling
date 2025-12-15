"""
局部微调模块 - 增强版
新增功能：输出验证集中样本最多小时的LST预测结果对比（TIF和PNG，600 DPI）
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import gc
import warnings
from collections import Counter
import rasterio
from rasterio.transform import from_bounds
warnings.filterwarnings('ignore')

from data_loader import DataLoader as LSTDataLoader
from preprocess import LSTDataset, DataCollator
from model import LSTTransformer, create_model_config

# 配置日志
log_dir = os.path.join(os.path.dirname(__file__), '../output/local_finetune')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train_local.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LocalFineTuner:
    """局部微调器 - 直接从全局模型加载并进行局部数据微调"""
    
    def __init__(self, config_path, global_pretrain_dir, local_finetune_dir):
        """初始化局部微调器"""
        self.config_path = config_path
        self.global_pretrain_dir = global_pretrain_dir
        self.local_finetune_dir = local_finetune_dir
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.data_params = self.config['data_params']
        self.model_params = self.config['model_params']
        self.finetune_params = self.config['train_params']['local_finetune']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data_loader = LSTDataLoader(config_path)
        
        os.makedirs(local_finetune_dir, exist_ok=True)
        
        logger.info(f"局部微调器初始化完成，使用设备: {self.device}")

    def load_global_pretrain_model(self, target_date):
        """加载全局预训练模型和完整的标准化器"""
        file_prefix = target_date.strftime('%Y%m')
        model_path = os.path.join(self.global_pretrain_dir, f'global_pretrain_{file_prefix}.pt')
        
        if not os.path.exists(model_path):
            logger.error(f"全局预训练模型文件不存在: {model_path}")
            return None, None
        
        logger.info(f"正在加载全局预训练模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        required_keys = ['model_state_dict', 'config', 'scaler_dict']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise KeyError(f"Checkpoint缺少必需的键: {missing_keys}")
        
        pretrain_scaler_dict = checkpoint['scaler_dict']
        model_config = checkpoint['config']
        
        model = LSTTransformer(model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"✓ 已成功加载全局预训练模型")
        logger.info(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, pretrain_scaler_dict

    def find_max_sample_hour_in_validation(self, val_dataset):
        """在验证集中找到样本最多的（日期+小时）组合，并返回该时刻的所有数据"""
        base_dataset = val_dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        
        # 提取时间特征
        month_features = base_dataset.time_features.get('month', None)
        day_features = base_dataset.time_features.get('day', None)
        hour_features = base_dataset.time_features.get('hour', None)
        
        if month_features is None or day_features is None or hour_features is None:
            logger.error("验证集中未找到完整的时间特征（月/日/小时）")
            return None, None
        
        # 获取验证集的索引
        if isinstance(val_dataset, Subset):
            val_indices = val_dataset.indices
        else:
            val_indices = list(range(len(val_dataset)))
        
        # 提取验证集的时间特征
        val_months = month_features[val_indices]
        val_days = day_features[val_indices]
        val_hours = hour_features[val_indices]
        
        # 组合成 (月, 日, 小时) 元组
        time_combinations = list(zip(val_months, val_days, val_hours))
        
        # 统计每个时刻的样本数
        time_counts = Counter(time_combinations)
        max_time, max_count = time_counts.most_common(1)[0]
        
        max_month, max_day, max_hour = max_time
        
        logger.info(f"验证集中样本最多的时刻: {int(max_month):02d}月{int(max_day):02d}日 {int(max_hour):02d}:00")
        logger.info(f"该时刻的样本数: {max_count}")
        
        # 创建掩码，提取该时刻的所有样本索引
        time_mask = np.array([
            (m == max_month) and (d == max_day) and (h == max_hour)
            for m, d, h in time_combinations
        ])
        
        hour_indices = np.array(val_indices)[time_mask]
        
        # 收集该时刻的数据
        hour_data = {
            'month': int(max_month),
            'day': int(max_day),
            'hour': int(max_hour),
            'count': max_count,
            'indices': hour_indices,
            'lon': base_dataset.features['lon'][hour_indices],
            'lat': base_dataset.features['lat'][hour_indices],
            'target': base_dataset.y_raw[hour_indices]  # 原始真实值（未标准化）
        }
        
        return max_time, hour_data

    def export_validation_hour_results(self, model, val_dataset, scaler_dict, target_date, output_dir):
        """输出验证集中样本最多时刻的预测结果（仅对验证集样本进行预测和可视化）"""
        max_time, hour_data = self.find_max_sample_hour_in_validation(val_dataset)
        
        if max_time is None or hour_data is None:
            logger.error("无法找到样本最多的时刻")
            return
        
        max_month = hour_data['month']
        max_day = hour_data['day']
        max_hour = hour_data['hour']
        
        logger.info(f"开始生成 {target_date.year}年{max_month:02d}月{max_day:02d}日 {max_hour:02d}:00 的验证集预测结果")
        
        hour_indices = hour_data['indices']
        
        base_dataset = val_dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        
        model.eval()
        batch_size = 2048
        all_predictions = []
        
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(hour_indices), batch_size), desc="预测验证集样本"):
                end_idx = min(start_idx + batch_size, len(hour_indices))
                batch_indices = hour_indices[start_idx:end_idx]
                
                batch_data = {
                    'num': torch.stack([base_dataset[idx]['num_features'] for idx in batch_indices]).to(self.device),
                    'time': torch.stack([base_dataset[idx]['time_features'] for idx in batch_indices]).to(self.device),
                    'loc': torch.stack([base_dataset[idx]['loc_features'] for idx in batch_indices]).to(self.device),
                    'cat': {
                        k: torch.stack([base_dataset[idx]['cat'][k] for idx in batch_indices]).to(self.device)
                        for k in base_dataset[batch_indices[0]]['cat'].keys()
                    }
                }
                
                outputs = model(batch_data).squeeze(-1).cpu().numpy()
                all_predictions.extend(outputs)
        
        predictions = np.array(all_predictions)
        predictions_denorm = scaler_dict['target'].inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        targets_denorm = hour_data['target']
        lons = hour_data['lon']
        lats = hour_data['lat']
        
        hour_data['predictions'] = predictions_denorm
        
        # 使用完整的日期+小时作为文件名
        file_prefix = f"{target_date.year}{max_month:02d}{max_day:02d}_{max_hour:02d}00"
        tif_pred_path = os.path.join(output_dir, f'LST_prediction_{file_prefix}.tif')
        tif_obs_path = os.path.join(output_dir, f'LST_observation_{file_prefix}.tif')
        png_path = os.path.join(output_dir, f'LST_comparison_{file_prefix}.png')
        
        spatial_shape = self.data_params['local_shape']
        
        self._save_sparse_geotiff(lons, lats, predictions_denorm, spatial_shape, tif_pred_path, 'Predicted LST')
        self._save_sparse_geotiff(lons, lats, targets_denorm, spatial_shape, tif_obs_path, 'Observed LST')
        
        # 构建完整日期用于可视化标题
        from datetime import datetime
        full_datetime = datetime(target_date.year, max_month, max_day)
        
        self._save_comparison_visualization(
            lons, lats, predictions_denorm, targets_denorm, 
            spatial_shape, png_path, full_datetime, max_hour
        )
        
        valid_mask = ~(np.isnan(predictions_denorm) | np.isnan(targets_denorm))
        if np.any(valid_mask):
            rmse = np.sqrt(mean_squared_error(targets_denorm[valid_mask], predictions_denorm[valid_mask]))
            mae = mean_absolute_error(targets_denorm[valid_mask], predictions_denorm[valid_mask])
            r2 = r2_score(targets_denorm[valid_mask], predictions_denorm[valid_mask])
            bias = np.mean(predictions_denorm[valid_mask] - targets_denorm[valid_mask])
            
            logger.info(f"\n验证集样本最多时刻 ({max_month:02d}月{max_day:02d}日 {max_hour:02d}:00) 的预测精度:")
            logger.info(f"  样本数: {hour_data['count']}")
            logger.info(f"  RMSE: {rmse:.4f} K")
            logger.info(f"  MAE: {mae:.4f} K")
            logger.info(f"  R$R^2$: {r2:.4f}")
            logger.info(f"  Bias: {bias:.4f} K")
        
        logger.info(f"\n✓ 验证集预测结果已保存:")
        logger.info(f"  - 预测值GeoTIFF: {tif_pred_path}")
        logger.info(f"  - 观测值GeoTIFF: {tif_obs_path}")
        logger.info(f"  - PNG对比图: {png_path}")

    def _save_sparse_geotiff(self, lons, lats, values, spatial_shape, output_path, description):
        """保存稀疏数据为GeoTIFF格式（只在有数据的位置填充值）"""
        height, width = spatial_shape
        
        lon_range = self.data_params['lon_range']
        lat_range = self.data_params['lat_range']
        
        data_grid = np.full((height, width), np.nan, dtype=np.float32)
        
        lon_res = (lon_range[1] - lon_range[0]) / width
        lat_res = (lat_range[1] - lat_range[0]) / height
        
        col_indices = ((lons - lon_range[0]) / lon_res).astype(int)
        row_indices = ((lat_range[1] - lats) / lat_res).astype(int)
        
        valid_mask = (
            (row_indices >= 0) & (row_indices < height) &
            (col_indices >= 0) & (col_indices < width)
        )
        
        data_grid[row_indices[valid_mask], col_indices[valid_mask]] = values[valid_mask]
        
        transform = from_bounds(
            lon_range[0], lat_range[0], lon_range[1], lat_range[1],
            width, height
        )
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data_grid.dtype,
            crs='EPSG:4326',
            transform=transform,
            compress='lzw',
            nodata=np.nan
        ) as dst:
            dst.write(data_grid, 1)
            dst.set_band_description(1, description)
        
        valid_pixels = np.sum(~np.isnan(data_grid))
        logger.info(f"GeoTIFF已保存: {os.path.basename(output_path)}, "
                   f"尺寸: {height}×{width}, 有效像素: {valid_pixels}")

    def _save_comparison_visualization(self, lons, lats, predictions, observations, 
                                      spatial_shape, output_path, target_date, target_hour):
        """保存预测值与观测值的对比可视化（600 DPI）"""
        height, width = spatial_shape
        
        lon_range = self.data_params['lon_range']
        lat_range = self.data_params['lat_range']
        
        pred_grid = np.full((height, width), np.nan, dtype=np.float32)
        obs_grid = np.full((height, width), np.nan, dtype=np.float32)
        
        lon_res = (lon_range[1] - lon_range[0]) / width
        lat_res = (lat_range[1] - lat_range[0]) / height
        
        col_indices = ((lons - lon_range[0]) / lon_res).astype(int)
        row_indices = ((lat_range[1] - lats) / lat_res).astype(int)
        
        valid_mask = (
            (row_indices >= 0) & (row_indices < height) &
            (col_indices >= 0) & (col_indices < width)
        )
        
        pred_grid[row_indices[valid_mask], col_indices[valid_mask]] = predictions[valid_mask]
        obs_grid[row_indices[valid_mask], col_indices[valid_mask]] = observations[valid_mask]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        all_values = np.concatenate([predictions, observations])
        vmin, vmax = np.nanpercentile(all_values, [2, 98])
        cmap = 'RdYlBu_r'
        
        # 左图：预测值
        im1 = axes[0].imshow(pred_grid, cmap=cmap, vmin=vmin, vmax=vmax, 
                            aspect='auto', interpolation='nearest')
        axes[0].set_title(f'模型预测LST\n{target_date.strftime("%Y-%m-%d")} {target_hour:02d}:00', 
                         fontsize=18, fontweight='bold', pad=15)
        axes[0].set_xlabel('经度方向 (像素)', fontsize=14)
        axes[0].set_ylabel('纬度方向 (像素)', fontsize=14)
        
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('LST (K)', fontsize=14, fontweight='bold')
        cbar1.ax.tick_params(labelsize=12)
        
        valid_pred = predictions[~np.isnan(predictions)]
        stats_text = f'样本数: {len(valid_pred)}\n'
        stats_text += f'最小值: {np.nanmin(valid_pred):.2f} K\n'
        stats_text += f'最大值: {np.nanmax(valid_pred):.2f} K\n'
        stats_text += f'平均值: {np.nanmean(valid_pred):.2f} K\n'
        stats_text += f'标准差: {np.nanstd(valid_pred):.2f} K'
        
        axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                             edgecolor='black', linewidth=1.5))
        
        # 右图：观测值
        im2 = axes[1].imshow(obs_grid, cmap=cmap, vmin=vmin, vmax=vmax, 
                            aspect='auto', interpolation='nearest')
        axes[1].set_title(f'MODIS观测LST\n{target_date.strftime("%Y-%m-%d")} {target_hour:02d}:00', 
                         fontsize=18, fontweight='bold', pad=15)
        axes[1].set_xlabel('经度方向 (像素)', fontsize=14)
        axes[1].set_ylabel('纬度方向 (像素)', fontsize=14)
        
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label('LST (K)', fontsize=14, fontweight='bold')
        cbar2.ax.tick_params(labelsize=12)
        
        valid_mask = ~(np.isnan(predictions) | np.isnan(observations))
        if np.any(valid_mask):
            diff = predictions[valid_mask] - observations[valid_mask]
            rmse = np.sqrt(np.mean(diff**2))
            mae = np.mean(np.abs(diff))
            r2 = r2_score(observations[valid_mask], predictions[valid_mask])
            bias = np.mean(diff)
            
            metrics_text = f'验证样本: {np.sum(valid_mask)}\n'
            metrics_text += f'RMSE: {rmse:.4f} K\n'
            metrics_text += f'MAE: {mae:.4f} K\n'
            metrics_text += f'$R^2$: {r2:.4f}\n'
            metrics_text += f'Bias: {bias:.4f} K'
            
            axes[1].text(0.02, 0.98, metrics_text, transform=axes[1].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                                edgecolor='darkgreen', linewidth=1.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        logger.info(f"对比可视化PNG已保存，分辨率: 600 DPI")

    def fine_tune(self, target_date, is_debug=False):
        """对特定日期的局部数据进行微调"""
        global_model, pretrain_scaler_dict = self.load_global_pretrain_model(target_date)
        if global_model is None:
            raise ValueError("无法加载全局预训练模型")
        
        logger.info(f"开始加载 {target_date.strftime('%Y-%m-%d')} 的局部微调数据")
        local_data = self.data_loader.load_data(target_date, is_global=False, is_pretrain=False)
        
        if local_data is None:
            raise ValueError("无法加载局部数据")
        
        logger.info("创建局部数据集，使用全局标准化器")
        local_dataset = LSTDataset(local_data['features'], local_data['target'],
                                    scaler_dict=pretrain_scaler_dict,
                                    is_train=True, is_global=False,
                                    fit_features=False,
                                    fit_target=False)
        scaler_dict = local_dataset.scaler_dict
   
        total_size = len(local_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(local_dataset, [train_size, val_size])
        
        logger.info(f"数据集划分: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}")
        
        batch_size = self.finetune_params['batch_size']
        embedding_layers = local_dataset.get_embedding_layers()
        collator = DataCollator(embedding_layers)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        
        local_model = LSTTransformer(global_model.config).to(self.device)
        local_model.load_state_dict(global_model.state_dict())
        
        freeze_layers = self.finetune_params['freeze_layers']
        local_model.freeze_layers(freeze_layers)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(local_model.parameters(), lr=self.finetune_params['learning_rate'])
        kl_weight = self.finetune_params['kl_weight']
        
        num_epochs = self.finetune_params['epochs_debug'] if is_debug else self.finetune_params['epochs']
        patience = self.finetune_params['early_stopping_patience']
        
        best_val_loss = float('inf')
        best_model_state = None
        counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_r2': []
        }
        
        file_prefix = target_date.strftime('%Y%m%d')
        output_dir = os.path.join(self.local_finetune_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        logger.info(f"开始局部微调: epochs={num_epochs}, batch_size={batch_size}")
        
        for epoch in range(num_epochs):
            local_model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch（训练）{epoch+1}/{num_epochs}"):
                batch_data = {
                    'num': batch['num_features'].to(self.device),
                    'time': batch['time_features'].to(self.device),
                    'loc': batch['loc_features'].to(self.device),
                    'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                }
                targets = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(batch_data).squeeze(-1)
                
                loss = criterion(outputs, targets)
                
                if global_model is not None and kl_weight > 0:
                    with torch.no_grad():
                        global_outputs = global_model(batch_data).squeeze(-1)
                    kl = criterion(outputs, global_outputs)
                    loss = loss + kl_weight * kl
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(targets)
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            local_model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch（验证）{epoch+1}/{num_epochs}"):
                    batch_data = {
                        'num': batch['num_features'].to(self.device),
                        'time': batch['time_features'].to(self.device),
                        'loc': batch['loc_features'].to(self.device),
                        'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                    }
                    targets = batch['target'].to(self.device)
                    outputs = local_model(batch_data).squeeze(-1)
                    
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * len(targets)
                    
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            
            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            
            target_scaler = pretrain_scaler_dict['target']
            val_preds_denorm = target_scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()
            val_targets_denorm = target_scaler.inverse_transform(val_targets.reshape(-1, 1)).flatten()
            
            val_rmse = np.sqrt(mean_squared_error(val_targets_denorm, val_preds_denorm))
            val_r2 = r2_score(val_targets_denorm, val_preds_denorm)
            
            history['val_rmse'].append(val_rmse)
            history['val_r2'].append(val_r2)

            logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, "
                        f"Val RMSE {val_rmse:.2f}K, Val R² {val_r2:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = local_model.state_dict().copy()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"早停于 epoch {epoch+1}")
                    break
        
        end_time = time.time()
        logger.info(f"局部微调完成，耗时 {(end_time - start_time)/60:.2f} 分钟。")
        
        if best_model_state:
            local_model.load_state_dict(best_model_state)
        
        final_model_path = os.path.join(output_dir, f'local_finetune_{file_prefix}.pt')
        
        torch.save({
            'model_state_dict': local_model.state_dict(),
            'config': local_model.config,
            'scaler_dict': scaler_dict,
            'hyperparams': {
                'freeze_layers': freeze_layers,
                'kl_weight': kl_weight,
                'learning_rate': self.finetune_params['learning_rate'],
                'batch_size': batch_size
            },
            'date': target_date.strftime('%Y%m%d'),
            'best_val_loss': best_val_loss
        }, final_model_path)
        
        logger.info(f"✓ 局部微调模型已保存到: {final_model_path}")
        
        # 输出验证集中样本最多小时的预测结果对比图
        logger.info(f"\n{'='*60}")
        logger.info(f"开始生成验证集样本最多小时的预测结果对比")
        logger.info(f"{'='*60}")
        
        self.export_validation_hour_results(
            local_model, 
            val_dataset,
            scaler_dict, 
            target_date,
            output_dir
        )
        
        self._plot_history(history, file_prefix, output_dir)
        self._plot_scatter(local_model, val_loader, pretrain_scaler_dict, file_prefix, output_dir)
        
        del local_model, train_loader, val_loader, train_dataset, val_dataset, local_data
        gc.collect()
        torch.cuda.empty_cache()

        return final_model_path

    def _plot_history(self, history, file_prefix, output_dir):
        """绘制训练历史曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('训练与验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['val_r2'], label='Validation R²', color='orange', linewidth=2)
        axes[1].set_title('验证集 R²')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('R² Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(history['val_rmse'], label='Validation RMSE', color='green', linewidth=2)
        axes[2].set_title('验证集 RMSE')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('RMSE (K)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        history_path = os.path.join(output_dir, f"local_finetune_{file_prefix}_history.png")
        plt.savefig(history_path, dpi=300)
        logger.info(f"训练历史图已保存: {history_path}")
        plt.close()

    def _plot_scatter(self, model, data_loader, scaler_dict, file_prefix, output_dir):
        """绘制并保存预测-实际散点图"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch_data = {
                    'num': batch['num_features'].to(self.device),
                    'time': batch['time_features'].to(self.device),
                    'loc': batch['loc_features'].to(self.device),
                    'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                }
                targets = batch['target'].to(self.device)
                outputs = model(batch_data).squeeze(-1)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        preds_np = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)
        
        preds_denorm = scaler_dict['target'].inverse_transform(preds_np.reshape(-1, 1)).flatten()
        targets_denorm = scaler_dict['target'].inverse_transform(targets_np.reshape(-1, 1)).flatten()

        plt.figure(figsize=(8, 8))
        plt.scatter(targets_denorm, preds_denorm, alpha=0.5, s=1)
        
        min_val = min(targets_denorm.min(), preds_denorm.min())
        max_val = max(targets_denorm.max(), preds_denorm.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
        r2 = r2_score(targets_denorm, preds_denorm)
        mae = mean_absolute_error(targets_denorm, preds_denorm)
        
        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f'$R^2$: {r2:.4f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f'MAE: {mae:.2f}', transform=plt.gca().transAxes)
        
        plt.title('局部微调：预测-实际散点图')
        plt.xlabel('实际值 (LST)')
        plt.ylabel('预测值 (LST)')
        plt.grid(True)
        
        scatter_path = os.path.join(output_dir, f"local_finetune_{file_prefix}_scatter.png")
        plt.savefig(scatter_path, dpi=300)
        logger.info(f"预测-实际散点图已保存到: {scatter_path}")
        plt.close()


if __name__ == "__main__":
    logger.info("开始局部微调（使用统一标准化）")
    
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    global_pretrain_dir = r"G:\CNN_SpatialDownscaling\output\global_pretrain"
    local_finetune_dir = r"G:\CNN_SpatialDownscaling\output\local_finetune"
    
    try:
        test_date = datetime(2018, 10, 1)
        
        finetuner = LocalFineTuner(config_path, global_pretrain_dir, local_finetune_dir)
        logger.info(f"开始对 {test_date.strftime('%Y-%m-%d')} 进行局部微调")
        
        finetune_model_path = finetuner.fine_tune(test_date, is_debug=False)
        
        if finetune_model_path:
            logger.info(f"✓ 微调完成，模型路径: {finetune_model_path}")
            logger.info(f"✓ LST预测结果对比已自动生成（TIF和PNG格式，600 DPI）")
        else:
            logger.error("✗ 微调失败")
        
    except Exception as e:
        logger.error(f"局部微调过程出错: {e}", exc_info=True)