"""
局部从头训练模块 - 消融实验版本
不使用全局预训练模型，直接在局部数据上从随机初始化开始训练
用于对比验证全局预训练的效果
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
warnings.filterwarnings('ignore')

from data_loader import DataLoader as LSTDataLoader
from preprocess import LSTDataset, DataCollator
from model import LSTTransformer, create_model_config

# 配置日志
log_dir = os.path.join(os.path.dirname(__file__), '../output/local_from_scratch')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'local_from_scratch.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 文件日志
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LocalTrainerFromScratch:
    """局部从头训练器 - 消融实验：不使用全局预训练，直接训练局部模型"""
    
    def __init__(self, config_path, output_dir):
        """
        初始化局部训练器
        
        Args:
            config_path: 配置文件路径
            output_dir: 模型输出目录
        """
        self.config_path = config_path
        self.output_dir = output_dir
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.data_params = self.config['data_params']
        self.model_params = self.config['model_params']
        self.train_params = self.config['train_params']['local_finetune']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data_loader = LSTDataLoader(config_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"局部从头训练器初始化完成，使用设备: {self.device}")
        logger.info("⚠️  消融实验模式：不使用全局预训练模型")

    def train_from_scratch(self, target_date, is_debug=False):
        """
        从头训练局部模型（不使用全局预训练）
        
        Args:
            target_date: 目标日期
            is_debug: 是否调试模式
            
        Returns:
            str: 模型保存路径
        """
        # 1. 加载局部数据
        logger.info(f"开始加载 {target_date.strftime('%Y-%m-%d')} 的局部数据")
        local_data = self.data_loader.load_data(target_date, is_global=False, is_pretrain=False)
        
        if local_data is None:
            raise ValueError("无法加载局部数据")
        
        # 2. 创建数据集 - 重新fit新的scaler
        logger.info("创建局部数据集，使用新的标准化器（从头训练）")
        local_dataset = LSTDataset(
            local_data['features'], 
            local_data['target'],
            scaler_dict=None,  # 不传入scaler，创建新的
            embedding_dims=None,  # 使用默认embedding配置
            is_train=True, 
            is_global=False,
            fit_features=True,  # ✅ 重新fit特征scaler
            fit_target=True     # ✅ 重新fit目标scaler
        )
        scaler_dict = local_dataset.scaler_dict
        
        logger.info(f"✅ 局部数据集scaler_dict键: {list(local_dataset.scaler_dict.keys())}")
        logger.info(f"✅ 局部target scaler统计: mean={scaler_dict['target'].mean_[0]:.2f}K, std={scaler_dict['target'].scale_[0]:.2f}K")
        
        # 3. 数据集划分
        total_size = len(local_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(local_dataset, [train_size, val_size])
        
        logger.info(f"数据集划分: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}")
        
        # 4. 创建数据加载器
        batch_size = self.train_params['batch_size']
        embedding_layers = local_dataset.get_embedding_layers()
        collator = DataCollator(embedding_layers)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        
        # 5. 创建新模型 - 随机初始化
        logger.info("创建新模型（随机初始化，无预训练权重）")
        feature_dims = local_dataset.get_feature_dims()
        
        model_config = create_model_config(
            num_features=feature_dims['num_features'],
            time_features=feature_dims['time_features'],
            loc_features=feature_dims['loc_features'],
            cat_features={name: info['num_classes'] for name, info in feature_dims['embedding_info'].items()},
            d_model=self.model_params['d_model'],
            n_heads=self.model_params['n_heads'],
            n_layers=self.model_params['n_layers'],
            dropout=self.model_params['dropout']
        )
        
        local_model = LSTTransformer(model_config).to(self.device)
        
        logger.info(f"✅ 模型创建成功（随机初始化）")
        trainable_params = sum(p.numel() for p in local_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in local_model.parameters())
        logger.info(f"✅ 可训练参数: {trainable_params:,} / {total_params:,} (100%，无冻结层)")
        
        # 6. 训练设置
        criterion = nn.MSELoss()
        optimizer = optim.Adam(local_model.parameters(), lr=self.train_params['learning_rate'])
        
        num_epochs = self.train_params['epochs_debug'] if is_debug else self.train_params['epochs']
        patience = self.train_params['early_stopping_patience']
        
        best_val_loss = float('inf')
        best_model_state = None
        counter = 0
        
        history = {
            'train_loss': [],
            'train_mse': [],
            'val_loss': [],
            'val_rmse': [],
            'val_r2': []
        }
        
        # 输出目录
        file_prefix = target_date.strftime('%Y%m%d')
        output_dir = os.path.join(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        logger.info(f"开始从头训练: epochs={num_epochs}, batch_size={batch_size}, lr={self.train_params['learning_rate']}")
        logger.info("⚠️  注意：此为消融实验，模型从随机初始化开始训练")
        
        for epoch in range(num_epochs):
            local_model.train()
            
            train_loss = 0.0
            train_mse = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch(训练){epoch+1}/{num_epochs}"):
                batch_data = {
                    'num': batch['num_features'].to(self.device),
                    'time': batch['time_features'].to(self.device),
                    'loc': batch['loc_features'].to(self.device),
                    'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                }
                targets = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(batch_data).squeeze(-1)
                
                # MSE损失
                mse_loss = criterion(outputs, targets)
                loss = mse_loss  # 无蒸馏损失
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(targets)
                train_mse += mse_loss.item() * len(targets)
            
            train_loss /= len(train_loader.dataset)
            train_mse /= len(train_loader.dataset)
            
            history['train_loss'].append(train_loss)
            history['train_mse'].append(train_mse)
            
            # 验证
            local_model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch(验证){epoch+1}/{num_epochs}"):
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
            
            # 计算其他指标 (反标准化后)
            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            
            target_scaler = scaler_dict['target']
            val_preds_denorm = target_scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()
            val_targets_denorm = target_scaler.inverse_transform(val_targets.reshape(-1, 1)).flatten()
            
            val_rmse = np.sqrt(mean_squared_error(val_targets_denorm, val_preds_denorm))
            val_r2 = r2_score(val_targets_denorm, val_preds_denorm)
            
            history['val_rmse'].append(val_rmse)
            history['val_r2'].append(val_r2)

            logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} (MSE {train_mse:.4f}), "
                        f"Val Loss {val_loss:.4f}, Val RMSE {val_rmse:.2f}K, Val R² {val_r2:.4f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}                
                counter = 0
                logger.info(f"  → 新的最佳验证损失，已保存模型状态")
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"早停于 epoch {epoch+1}")
                    break
        
        end_time = time.time()
        logger.info(f"从头训练完成，耗时 {(end_time - start_time)/60:.2f} 分钟。")
        
        # 7. 保存最佳模型
        if best_model_state:
            local_model.load_state_dict(best_model_state)
        
        final_model_path = os.path.join(output_dir, f'local_from_scratch_{file_prefix}.pt')
        
        # 保存完整checkpoint
        torch.save({
            'model_state_dict': local_model.state_dict(),
            'config': local_model.config,
            'scaler_dict': scaler_dict,
            'hyperparams': {
                'learning_rate': self.train_params['learning_rate'],
                'batch_size': batch_size,
                'training_mode': 'from_scratch'  # 标记训练模式
            },
            'date': target_date.strftime('%Y%m%d'),
            'best_val_loss': best_val_loss,
            'final_metrics': {
                'val_rmse': history['val_rmse'][-1],
                'val_r2': history['val_r2'][-1]
            }
        }, final_model_path)
        
        logger.info(f"✅ 从头训练模型已保存到: {final_model_path}")
        logger.info(f"✅ 最终验证指标: RMSE={history['val_rmse'][-1]:.2f}K, R²={history['val_r2'][-1]:.4f}")
        
        # 8. 绘制训练历史和散点图
        self._plot_history(history, file_prefix, output_dir)
        self._plot_scatter(local_model, val_loader, scaler_dict, file_prefix, output_dir)
        
        # 内存清理
        del local_model, train_loader, val_loader, train_dataset, val_dataset, local_data
        gc.collect()
        torch.cuda.empty_cache()

        return final_model_path

    def _plot_history(self, history, file_prefix, output_dir):
        """绘制训练历史曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss曲线
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('训练与验证损失（从头训练）')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MSE曲线
        axes[0, 1].plot(history['train_mse'], label='Train MSE', linewidth=2, color='blue')
        axes[0, 1].set_title('训练MSE损失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # R²曲线
        axes[1, 0].plot(history['val_r2'], label='Validation R²', color='green', linewidth=2)
        axes[1, 0].set_title('验证集 R²')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # RMSE曲线
        axes[1, 1].plot(history['val_rmse'], label='Validation RMSE', color='red', linewidth=2)
        axes[1, 1].set_title('验证集 RMSE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE (K)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        history_path = os.path.join(output_dir, f"local_from_scratch_{file_prefix}_history.png")
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
        
        # 反标准化
        preds_denorm = scaler_dict['target'].inverse_transform(preds_np.reshape(-1, 1)).flatten()
        targets_denorm = scaler_dict['target'].inverse_transform(targets_np.reshape(-1, 1)).flatten()

        # 绘图
        plt.figure(figsize=(8, 8))
        plt.scatter(targets_denorm, preds_denorm, alpha=0.5, s=1)
        
        # 绘制对角线
        min_val = min(targets_denorm.min(), preds_denorm.min())
        max_val = max(targets_denorm.max(), preds_denorm.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
        
        # 计算并标注度量指标
        rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
        r2 = r2_score(targets_denorm, preds_denorm)
        mae = mean_absolute_error(targets_denorm, preds_denorm)

        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f} K', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.90, f'$R^2$: {r2:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.85, f'MAE: {mae:.2f} K', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # 设置图表标题和标签
        plt.title(f'From Scratch Training: {datetime.strptime(file_prefix, "%Y%m%d").strftime("%Y-%m-%d")}', fontsize=14)
        plt.xlabel('Original LST (K)', fontsize=12)
        plt.ylabel('Predicted LST (K)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        scatter_path = os.path.join(output_dir, f"local_from_scratch_{file_prefix}_scatter.png")
        plt.savefig(scatter_path, dpi=300)
        logger.info(f"预测-实际散点图已保存到: {scatter_path}")
        plt.close()


def compare_with_finetune(from_scratch_path, finetune_path):
    """
    比较从头训练和微调模型的性能
    
    Args:
        from_scratch_path: 从头训练模型路径
        finetune_path: 微调模型路径
    """
    logger.info("\n" + "="*80)
    logger.info("消融实验结果对比")
    logger.info("="*80)
    
    # 加载模型检查点
    scratch_ckpt = torch.load(from_scratch_path, map_location='cpu', weights_only=False)
    finetune_ckpt = torch.load(finetune_path, map_location='cpu', weights_only=False)
    
    # 提取指标
    scratch_metrics = scratch_ckpt['final_metrics']
    finetune_metrics = finetune_ckpt['final_metrics']
    
    logger.info("\n从头训练（无预训练）:")
    logger.info(f"  RMSE: {scratch_metrics['val_rmse']:.2f} K")
    logger.info(f"  R²:   {scratch_metrics['val_r2']:.4f}")
    
    logger.info("\n微调训练（使用全局预训练）:")
    logger.info(f"  RMSE: {finetune_metrics['val_rmse']:.2f} K")
    logger.info(f"  R²:   {finetune_metrics['val_r2']:.4f}")
    
    logger.info("\n性能提升:")
    rmse_improve = scratch_metrics['val_rmse'] - finetune_metrics['val_rmse']
    r2_improve = finetune_metrics['val_r2'] - scratch_metrics['val_r2']
    
    logger.info(f"  RMSE降低: {rmse_improve:.2f} K ({100*rmse_improve/scratch_metrics['val_rmse']:.1f}%)")
    logger.info(f"  R²提升:   {r2_improve:.4f} ({100*r2_improve/(1-scratch_metrics['val_r2']):.1f}% of possible improvement)")
    
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    logger.info("开始消融实验：从头训练局部模型（不使用全局预训练）")
    
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    output_dir = r"G:\CNN_SpatialDownscaling\output\local_from_scratch"
    
    try:
        # 批量训练示例
        start_date = 6
        end_date = 6
        
        for day in range(start_date, end_date + 1):
            test_date = datetime(2018, 10, day)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"开始处理日期: {test_date.strftime('%Y-%m-%d')}")
            logger.info(f"{'='*80}\n")
            
            trainer = LocalTrainerFromScratch(config_path, output_dir)
            
            # 调试模式: is_debug=True
            model_path = trainer.train_from_scratch(test_date, is_debug=False)
            
            if model_path:
                logger.info(f"✅ {test_date.strftime('%Y-%m-%d')} 从头训练完成，模型路径: {model_path}")
                
                # 如果存在对应的微调模型，进行对比
                finetune_path = os.path.join(
                    r"G:\CNN_SpatialDownscaling\output\local_finetune",
                    f"local_finetune_{test_date.strftime('%Y%m%d')}.pt"
                )
                if os.path.exists(finetune_path):
                    compare_with_finetune(model_path, finetune_path)
            else:
                logger.error(f"✗ {test_date.strftime('%Y-%m-%d')} 从头训练失败")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"所有日期处理完成!")
        logger.info(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"消融实验过程出错: {e}", exc_info=True)
