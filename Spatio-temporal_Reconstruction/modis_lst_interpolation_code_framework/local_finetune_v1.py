"""
局部微调模块 - v2修复版本 (结构对齐版 + 验证进度条)
关键修复：
1. 结构与 train_global.py 对齐
2. 添加学习率调度器
3. 统一可视化函数调用
4. **新增：验证集进度条显示**
5. **新增：每轮详细打印训练与验证结果**
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
from model import LSTTransformer
from visualization_utils import plot_training_history, plot_hexbin_scatter, plot_model_scatter

# 配置日志
log_dir = os.path.join(os.path.dirname(__file__), '../output/local_finetune')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'local_finetune_05.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 文件日志
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台日志
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Times New Roman','SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LocalFineTuner:
    """局部微调器 - 直接从全局模型加载并进行局部数据微调"""
    
    def __init__(self, config_path, global_pretrain_dir, local_finetune_dir):
        """
        初始化局部微调器
        """
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
        # 构造全局模型路径（按月）
        file_prefix = target_date.strftime('%Y%m')
        model_path = os.path.join(self.global_pretrain_dir, f'global_pretrain_{file_prefix}.pt')
        
        if not os.path.exists(model_path):
            logger.error(f"全局预训练模型文件不存在: {model_path}")
            return None, None, None
        
        # 加载checkpoint
        logger.info(f"正在加载全局预训练模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 获取scaler_dict
        pretrain_scaler_dict = checkpoint['scaler_dict']
        
        # 验证target scaler的合理性
        target_scaler = pretrain_scaler_dict['target']
        logger.info(f"全局Target Scaler信息: Mean={target_scaler.mean_[0]:.2f}K, Std={target_scaler.scale_[0]:.2f}K")
        
        model_config = checkpoint['config']
        
        # 转换embedding_dims格式以兼容LSTDataset
        embedding_dims = {}
        if 'cat_features' in model_config:
            embedding_dims = {
                name: {'embed_dim': dim} 
                for name, dim in model_config['cat_features'].items()
            }
            logger.info(f"✓ 已转换embedding_dims格式，包含 {len(embedding_dims)} 个分类特征")
        
        model = LSTTransformer(model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"✓ 已成功加载全局预训练模型")
        return model, pretrain_scaler_dict, embedding_dims

    def prepare_data(self, target_date, pretrain_scaler_dict, embedding_dims, is_debug=False):
        """
        准备局部微调数据
        关键点：强制使用fit_features=False, fit_target=False
        """
        logger.info(f"开始加载 {target_date.strftime('%Y-%m-%d')} 的局部微调数据")
        local_data = self.data_loader.load_data(target_date, is_global=False, is_pretrain=False)
        
        if local_data is None:
            raise ValueError("无法加载局部数据")
        
        # 创建数据集 - 严格使用全局pretrain_scaler_dict，不进行fit
        logger.info("创建局部数据集, 严格使用全局标准化器")
        local_dataset = LSTDataset(
            local_data['features'], 
            local_data['target'],
            scaler_dict=pretrain_scaler_dict,
            embedding_dims=embedding_dims,
            is_train=True, 
            is_global=False,
            fit_features=False,
            fit_target=False
        )
        # 固定随机种子，保证划分可复现
        split_generator = torch.Generator().manual_seed(42)
        
        # 数据集划分
        if is_debug:
            debug_size = int(0.1 * len(local_dataset))
            local_dataset = Subset(local_dataset, list(range(debug_size)))
            logger.info(f"调试模式: 使用前 {debug_size} 条数据进行微调")
            train_size = int(0.8 * debug_size)
            val_size = debug_size - train_size
        else:
            total_size = len(local_dataset)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
        
        # 替代 random_split 的数据集分割逻辑
        indices = np.arange(len(local_dataset))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]

        train_dataset = Subset(local_dataset, train_indices)
        val_dataset = Subset(local_dataset, val_indices)

        logger.info(f"数据集划分: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}")
        
        # 处理Subset中的Embedding Layer获取
        base_dataset = local_dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        
        embedding_layers = base_dataset.get_embedding_layers()
        collator = DataCollator(embedding_layers)
        
        batch_size = self.finetune_params['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        
        return train_loader, val_loader, base_dataset.scaler_dict

    def train_epoch(self, model, global_model, train_loader, optimizer, criterion, scaler_dict, epoch, kl_weight):
        """
        训练一个epoch
        结构保持V1的tqdm进度条，但逻辑简化以匹配V2（去除每batch的冗余反归一化计算，提高速度）
        """

        model.train()
        if global_model:
            global_model.eval()
                
        total_loss = 0.0
        total_mse_loss = 0.0
        total_kl_loss = 0.0
            
        # 用于后续计算真实指标
        all_preds = []
        all_targets = []
        # 训练进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", dynamic_ncols=True)
            
        for batch in progress_bar:
            batch_data = {
                'num': batch['num_features'].to(self.device),
                'time': batch['time_features'].to(self.device),
                'loc': batch['loc_features'].to(self.device),
                'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
            }
                
            # 【关键修复】确保 targets 和 outputs 都是一维向量 [Batch]
            targets = batch['target'].to(self.device) 
                
            optimizer.zero_grad()
            outputs = model(batch_data).squeeze(-1)   
                
            # 1. MSE Loss (维度必须匹配！)
            mse_loss = criterion(outputs, targets)
                
            # 2. KL Loss
            kl_loss = torch.tensor(0.0, device=self.device)
            if global_model is not None and kl_weight > 0:
                with torch.no_grad():
                    global_outputs = global_model(batch_data).squeeze(-1)
                kl_loss = criterion(outputs, global_outputs)

            # Total Loss
            loss = mse_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
                
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_kl_loss += kl_loss.item()
             
                
            # 收集数据用于计算 Epoch 级指标 (避免 Batch 级 R2 错误)
            # all_preds.extend(outputs.detach().cpu().numpy())
            # all_targets.extend(targets.detach().cpu().numpy())
                
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'MSE': f"{mse_loss.item():.4f}", 'KL': f"{kl_loss.item():.4f}"})
                
        # 计算整个 Epoch 的平均 Loss
        avg_loss = total_loss / len(train_loader)
        avg_mse_loss = total_mse_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
            
        '''       
        # 统一反归一化并计算真实物理指标
        target_scaler = scaler_dict['target']
        preds_denorm = target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
        targets_denorm = target_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()
            
        train_rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
        train_mae = mean_absolute_error(targets_denorm, preds_denorm)
        train_r2 = r2_score(targets_denorm, preds_denorm)'''
            
        return avg_loss, avg_mse_loss, avg_kl_loss  #, train_mae, train_r2, train_rmse
    

    def evaluate(self, model, data_loader, criterion, scaler_dict, desc="Validation"):
        """评估模型性能 (带进度条)"""
        model.eval()
        total_loss = 0.0
        val_preds = []
        val_targets = []
        
        # 验证进度条
        progress_bar = tqdm(data_loader, desc=desc, dynamic_ncols=True)
        
        with torch.no_grad():
            for batch in progress_bar:
                batch_data = {
                    'num': batch['num_features'].to(self.device),
                    'time': batch['time_features'].to(self.device),
                    'loc': batch['loc_features'].to(self.device),
                    'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                }
                targets = batch['target'].to(self.device)
                outputs = model(batch_data).squeeze(-1)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item() * len(targets)
                
                # 收集原始预测值（标准化后的）
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
                progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(data_loader.dataset)
        
        #  统一反归一化计算真实物理量指标
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        target_scaler = scaler_dict['target']
        val_preds_denorm = target_scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()
        val_targets_denorm = target_scaler.inverse_transform(val_targets.reshape(-1, 1)).flatten()
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(val_targets_denorm, val_preds_denorm))
        r2 = r2_score(val_targets_denorm, val_preds_denorm)
        mae = mean_absolute_error(val_targets_denorm, val_preds_denorm)
        mse = mean_squared_error(val_targets_denorm, val_preds_denorm)        
        metrics = {
            'loss': avg_loss,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'true_mse': mse
        }
        return metrics, val_preds_denorm, val_targets_denorm

    def fine_tune(self, target_date, is_debug=False):
        """对特定日期的局部数据进行微调 (主循环)"""
        # 1. 加载全局预训练模型
        global_model, pretrain_scaler_dict, embedding_dims = self.load_global_pretrain_model(target_date)
        if global_model is None:
            raise ValueError("无法加载全局预训练模型")
            
        # 2. 准备数据
        train_loader, val_loader, scaler_dict = self.prepare_data(target_date, pretrain_scaler_dict, embedding_dims, is_debug)
        
        # 3. 初始化局部模型
        local_model = LSTTransformer(global_model.config).to(self.device)
        local_model.load_state_dict(global_model.state_dict())
        
        # 冻结层
        freeze_layers = self.finetune_params['freeze_layers']
        local_model.freeze_layers(freeze_layers)
        logger.info(f"✓ 已冻结前 {freeze_layers} 层")
        
        # 4. 训练设置
        criterion = nn.MSELoss()
        optimizer = optim.Adam(local_model.parameters(), lr=self.finetune_params['learning_rate'])
        
        
        # 使用恒定学习率进行微调
        
        kl_weight = self.finetune_params['kl_weight']
        num_epochs = self.finetune_params['epochs_debug'] if is_debug else self.finetune_params['epochs']
        patience = self.finetune_params['early_stopping_patience']
        
        # 历史记录
        history = {
            'train_loss': [], 'train_mse': [], 'train_kl': [], 'train_mae': [], 'train_r2': [], 'train_true_mse': [],
            'val_loss': [], 'val_rmse': [], 'val_r2': [], 'val_mae': [], 'val_true_mse': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        counter = 0
        
        # 输出目录
        file_prefix = target_date.strftime('%Y%m%d')
        output_dir = self.local_finetune_dir
        
        start_time = time.time()
        logger.info(f"开始局部微调: epochs={num_epochs}, lr={self.finetune_params['learning_rate']}")
        
        for epoch in range(num_epochs):
            # 1. 训练阶段
            train_loss, train_mse, train_kl = self.train_epoch(
                local_model, global_model, train_loader, optimizer, criterion, scaler_dict, epoch, kl_weight
            )#, train_mae, train_r2, train_true_mse
            
            # 2. 验证阶段 (传入描述信息以显示进度条)
            val_metrics, val_preds, val_targets = self.evaluate(
                local_model, val_loader, criterion, scaler_dict, desc=f"Epoch {epoch+1} [Val]"
            )
            val_loss = val_metrics['loss']
            
            # 记录
            history['train_loss'].append(train_loss)
            history['train_mse'].append(train_mse)
            history['train_kl'].append(train_kl)
            '''history['train_mae'].append(train_mae)  
            history['train_r2'].append(train_r2)  
            history['train_true_mse'].append(train_true_mse)  '''
            history['val_loss'].append(val_loss)
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_r2'].append(val_metrics['r2'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_true_mse'].append(val_metrics['true_mse'])
            
            # 3. 打印详细日志 (每一轮都打印)
            logger.info(f"Epoch {epoch+1:02d}/{num_epochs}: "
                        f"Train Loss={train_loss:.6f} (MSE={train_mse:.6f}, KL={train_kl:.6f}) | "
                       # f"Train RMSE={np.sqrt(train_true_mse):.4f}K, Train MSE={train_true_mse:.4f}K, Train MAE={train_mae:.4f}K, Train R²={train_r2:.4f} | "
                        f"Val Loss={val_loss:.6f}, Val MSE={val_metrics['true_mse']:.4f}K, Val MAE={val_metrics['mae']:.4f}K , "
                        f"Val RMSE={val_metrics['rmse']:.4f}K, Val R²={val_metrics['r2']:.4f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
                counter = 0
                logger.info(f"  >>> 新的最佳验证损失: {best_val_loss:.6f} (Epoch {epoch+1})")
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"早停触发: 验证损失连续 {counter} 轮未下降")
                    break
        
        end_time = time.time()
        logger.info(f"局部微调完成，耗时 {(end_time - start_time)/60:.2f} 分钟。")
        
        # 5. 保存模型
        if best_model_state:
            local_model.load_state_dict(best_model_state)
            
        final_model_path = os.path.join(output_dir, f'local_finetune_{file_prefix}.pt')
        torch.save({
            'model_state_dict': local_model.state_dict(),
            'config': local_model.config,
            'scaler_dict': scaler_dict, # 这里的scaler_dict已经是包含全局信息的完整字典
            'hyperparams': {
                'freeze_layers': freeze_layers,
                'kl_weight': kl_weight,
                'learning_rate': self.finetune_params['learning_rate'],
                'batch_size': self.finetune_params['batch_size']
            },
            'date': target_date.strftime('%Y%m%d'),
            'best_val_loss': best_val_loss,
            'final_metrics': val_metrics
        }, final_model_path)
        logger.info(f"✓ 模型已保存: {final_model_path}")
        
        # 6. 统一可视化 (与 train_global 一致)
        # 训练曲线
        plot_training_history(
            history, 
            os.path.join(output_dir, f"local_finetune_{file_prefix}_history.png"),
            title_prefix=f'Local Finetune {file_prefix}'
        )
        # 最终验证集预测（获取完整验证集预测）
        _, all_val_preds, all_val_targets = self.evaluate(local_model, val_loader, criterion, scaler_dict, desc="Final Evaluation")
        
        # Hexbin 密度图
        plot_hexbin_scatter(
            all_val_targets, all_val_preds,
            os.path.join(output_dir, f'local_finetune_{file_prefix}_hexbin.png'),
            title=f'Local Finetune {target_date.strftime("%Y-%m-%d")} ',
            xlabel='Original LST (K)', 
            ylabel='Predicted LST (K)'
        )
        
        # 普通散点图
        plot_model_scatter(local_model, val_loader, pretrain_scaler_dict, 
                           file_prefix,
                           os.path.join(output_dir, f'local_finetune_{file_prefix}_scatter.png'),                           
                           xlabel='Original LST (K)', 
                           ylabel='Predicted LST (K)',
                           dpi=300
        )        
        
        # 清理
        gc.collect()
        torch.cuda.empty_cache()

        
        return final_model_path

if __name__ == "__main__":
    logger.info("开始局部微调")
    
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    global_pretrain_dir = r"G:\CNN_SpatialDownscaling\output\global_pretrain4"
    local_finetune_dir = r"G:\CNN_SpatialDownscaling\output\local_finetune_121"
    
    try:
        finetuner = LocalFineTuner(config_path, global_pretrain_dir, local_finetune_dir)
        
        # 测试日期
        test_dates = [datetime(2018, 5, day) for day in [31]]
        
        for test_date in test_dates:
            logger.info(f"\n{'='*50}")
            logger.info(f"开始处理: {test_date.strftime('%Y-%m-%d')}")
            logger.info(f"{'='*50}")
            
            finetune_model_path = finetuner.fine_tune(test_date, is_debug=False)
            
            if finetune_model_path:
                logger.info(f"✓ 完成: {finetune_model_path}")
            else:
                logger.error("✗ 失败")
                
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)