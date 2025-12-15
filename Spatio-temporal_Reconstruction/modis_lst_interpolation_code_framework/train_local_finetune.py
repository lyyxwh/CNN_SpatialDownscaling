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
from scipy.stats import entropy
import gc

from data_loader import DataLoader as LSTDataLoader
from preprocess import LSTDataset, DataCollator
from model import LSTTransformer, create_model_config


# 配置日志：同时输出到文件和控制台
log_dir = os.path.join(os.path.dirname(__file__), '../output/local_finetune')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train_local.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 文件日志
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

'''# 控制台日志
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
'''
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LocalFineTuner:
    """局部微调器 - 负责每日在局部MODIS数据上微调全局模型"""
    
    def __init__(self, config_path, global_finetune_dir, local_finetune_dir):
        """
        初始化局部微调器
        
        Args:
            config_path (str): 配置文件路径
            global_finetune_dir (str): 全局微调模型目录
            local_finetune_dir (str): 局部微调模型输出目录
        """
        self.config_path = config_path
        self.global_finetune_dir = global_finetune_dir
        self.local_finetune_dir = local_finetune_dir
        os.makedirs(local_finetune_dir, exist_ok=True)
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

       
        # 检查 GPU 可用性
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                self.device = torch.device("cuda:0")
                self.is_dataparallel = True
                logger.info(f"检测到 {num_gpus} 个 GPU，将使用 DataParallel 进行训练。")
            else:
                self.device = torch.device("cuda:0")
                self.is_dataparallel = False
                logger.info(f"检测到单个 GPU，将使用 {self.device} 进行训练。")
        else:
            self.device = torch.device("cpu")
            self.is_dataparallel = False
            logger.warning("未检测到可用的 GPU，将使用 CPU 进行训练。")      
                  
        #self.device = torch.device(self.config.get('device', 'cpu') if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        self.finetune_params = self.config['train_params']['local_finetune']
        self.kl_weight = self.finetune_params.get('kl_weight', 0.1)

    def fine_tune(self, target_date, is_debug=False):
        """
        对指定日期的局部模型进行微调
        
        Args:
            target_date (datetime): 目标日期
            is_debug (bool): 是否为调试模式
            
        Returns:
            str: 保存的微调模型路径，如果失败则返回None
        """
        try:
            date_str = target_date.strftime('%Y%m%d')
            month_str = target_date.strftime('%Y%m')

            # 1. 加载全局微调模型
            global_model_path = os.path.join(self.global_finetune_dir, f"global_finetune_{date_str}.pt")
            pretrain_config_path = os.path.join(self.global_finetune_dir.replace('global_finetune', 'global_pretrain'), f"global_pretrain_{month_str}.pt")

            if not os.path.exists(global_model_path) or not os.path.exists(pretrain_config_path):
                logger.error(f"全局微调模型或预训练指标文件不存在，跳过 {date_str} 的局部微调。")
                return None

            # 从保存的.pt文件中加载模型配置和标准化器信息
            checkpoint = torch.load(pretrain_config_path, map_location=self.device, weights_only=False)
            model_config = checkpoint['config']
            scaler_dict = checkpoint['scaler_dict']
            
            # 修复: 转换 embedding_dims 格式以匹配 LSTDataset 的要求
            embedding_dims = {name: {'embed_dim': dim} for name, dim in model_config['cat_features'].items()}

            # 创建局部模型
            local_model = LSTTransformer(model_config).to(self.device)
            local_model.load_state_dict(torch.load(global_model_path, map_location=self.device))
            logger.info(f"成功加载全局微调模型作为局部模型初始化。")

            # 2. 冻结部分层
            freeze_layers = self.finetune_params.get('freeze_layers', 2)
            for i, (name, param) in enumerate(local_model.named_parameters()):
                if f"transformer_encoder.layers.{i}" in name and i < freeze_layers:
                    param.requires_grad = False
            logger.info(f"已冻结模型的前 {freeze_layers} 层。")

            # 3. 加载微调数据
            logger.info(f"开始加载 {date_str} 的局部微调数据 (±15天)...")
            loader = LSTDataLoader(self.config_path)
            data = loader.load_data(target_date, is_global=False, is_pretrain=False)

            if data is None or data['features'] is None or len(data['features']['lat']) == 0:
                logger.warning(f"跳过 {date_str} 的局部微调，无有效MODIS样本。")
                return None
                
            dataset = LSTDataset(
                data['features'], 
                data['target'], 
                scaler_dict=scaler_dict,
                embedding_dims=embedding_dims,
                is_train=True,
                is_global=False
            )

            # 划分训练集和验证集
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            dataset_indices = list(range(len(dataset)))
            np.random.shuffle(dataset_indices)
            
            # 调试模式下使用少量数据
            if is_debug:
                # 使用总数据的 10% 进行调试
                debug_size = int(0.1 * len(dataset))
                debug_indices = dataset_indices[:debug_size]
                
                # 在调试子集上划分训练和验证集
                debug_train_size = int(0.8 * debug_size)
                train_indices = debug_indices[:debug_train_size]
                val_indices = debug_indices[debug_train_size:]

                train_dataset = Subset(dataset, train_indices)
                val_dataset = Subset(dataset, val_indices)  
                logger.info(f"调试模式: 使用 {debug_size} 个样本 ({len(train_dataset)} 训练, {len(val_dataset)} 验证)")
            else:
                train_indices = dataset_indices[:train_size]
                val_indices = dataset_indices[train_size:]
                train_dataset = Subset(dataset, train_indices)
                val_dataset = Subset(dataset, val_indices)
            
            collator = DataCollator(dataset.get_embedding_layers())
            batch_size = self.finetune_params['batch_size']
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=collator
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=collator
            )
            logger.info(f"加载微调数据成功，训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

            # 4. 微调过程
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=self.finetune_params['learning_rate'])
            mse_loss = nn.MSELoss()
            
            epochs = self.finetune_params['epochs_debug'] if is_debug else self.finetune_params['epochs']
            patience = self.finetune_params['early_stopping_patience']
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_r2': []}

            start_time = time.time()
            for epoch in range(epochs):
                # 训练阶段
                local_model.train()
                total_loss = 0
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (训练)", leave=False):
                    batch_data = {
                        'num': batch['num_features'].to(self.device),
                        'time': batch['time_features'].to(self.device),
                        'loc': batch['loc_features'].to(self.device),
                        'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                    }
                    targets = batch['target'].to(self.device).unsqueeze(1)
                    
                    optimizer.zero_grad()
                    local_outputs = local_model(batch_data)
                    
                    mse = mse_loss(local_outputs, targets)
                    loss = mse + self.kl_weight * torch.zeros(1, device=self.device) # KL散度损失占位符
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_train_loss = total_loss / len(train_loader)
                history['train_loss'].append(avg_train_loss)
                
                # 验证阶段
                local_model.eval()
                total_val_loss, all_val_targets, all_val_outputs = 0, [], []
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (验证)", leave=False):
                        batch_data = {
                            'num': batch['num_features'].to(self.device),
                            'time': batch['time_features'].to(self.device),
                            'loc': batch['loc_features'].to(self.device),
                            'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                        }
                        targets = batch['target'].to(self.device).unsqueeze(1)
                        
                        outputs = local_model(batch_data)
                        loss = mse_loss(outputs, targets)
                        total_val_loss += loss.item()
                        
                        all_val_targets.extend(targets.cpu().numpy())
                        all_val_outputs.extend(outputs.cpu().numpy())
                
                avg_val_loss = total_val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)

                # 将预测结果反标准化
                all_val_targets_inv = dataset.scaler_dict['target'].inverse_transform(np.array(all_val_targets))
                all_val_outputs_inv = dataset.scaler_dict['target'].inverse_transform(np.array(all_val_outputs))

                # 计算RMSE和R2
                val_rmse = np.sqrt(mean_squared_error(all_val_targets_inv, all_val_outputs_inv))
                val_r2 = r2_score(all_val_targets_inv, all_val_outputs_inv)
                
                history['val_rmse'].append(val_rmse)
                history['val_r2'].append(val_r2)
                
                # 打印指标
                elapsed_time = time.time() - start_time
                time_per_epoch = elapsed_time / (epoch + 1)
                remaining_time = time_per_epoch * (epochs - epoch - 1)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, "
                            f"RMSE: {val_rmse:.2f}, R2: {val_r2:.4f}, "
                            f"耗时: {elapsed_time:.1f}s, 预计剩余: {remaining_time:.1f}s")
                
                # Early Stopping 检查
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
                    logger.info(f"验证集损失降低，保存当前最佳模型。")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"早停：验证集损失连续 {patience} 个 epoch 未降低，停止训练。")
                        break

            if best_model_state is not None:
                local_model.load_state_dict(best_model_state)
                logger.info("已加载最佳模型权重。")
                
            # 5. 保存微调模型和评估指标
            local_model_path = os.path.join(self.local_finetune_dir, f"local_finetune_{date_str}.pt")
            torch.save(local_model.state_dict(), local_model_path)
            logger.info(f"局部微调模型已保存到: {local_model_path}")
            
            self.plot_metrics(history, self.local_finetune_dir, date_str)
            self.plot_scatter(all_val_targets_inv, all_val_outputs_inv, self.local_finetune_dir, date_str)

            return local_model_path
        except Exception as e:
            logger.error(f"模型微调或评估发生错误: {e}")
            return None    
        finally:
            logger.info("开始执行模型训练后清理操作...")
            model = locals().get('local_model', None)  # 保证变量存在
            # 1. 释放 GPU 显存
            if torch.cuda.is_available():
                # 清理模型在 GPU 上的所有引用
                if model is not None:
                # 将模型从 GPU 移动到 CPU（如果它还在 GPU 上），然后删除
                    model.cpu()
                    del model
                    logger.info("模型对象已从 GPU 移动到 CPU 并删除。")

                    # 清理 PyTorch 的 CUDA 缓存
                    torch.cuda.empty_cache()
                    logger.info("PyTorch CUDA 缓存已清理 (torch.cuda.empty_cache())。")
                            
                    # 记录当前显存使用情况（可选）
                    current_usage = torch.cuda.memory_allocated() / 1024**3
                    max_usage = torch.cuda.max_memory_allocated() / 1024**3
                    logger.info(f"清理后 GPU 显存分配: {current_usage:.2f} GB, 历史最大分配: {max_usage:.2f} GB")
            else:
                # 如果没有 GPU，只删除模型对象
                if model is not None:
                    del model
                    logger.info("模型对象已删除。")

            # 2. 强制 Python 垃圾回收
            gc.collect()
            logger.info("Python 垃圾收集完成 (gc.collect())。")
        

    def plot_metrics(self, history, output_dir, file_prefix):
        """绘制并保存训练和验证指标图表"""
        plt.figure(figsize=(12, 5))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('局部微调：训练与验证损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 绘制 RMSE 和 R2
        plt.subplot(1, 2, 2)
        plt.plot(history['val_rmse'], label='验证集RMSE', color='orange')
        plt.plot(history['val_r2'], label='验证集$R^2$', color='green')
        plt.title('局部微调：验证集RMSE与$R^2$')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"local_finetune_{file_prefix}_metrics.png")
        plt.savefig(plot_path)
        logger.info(f"局部微调指标图表已保存到: {plot_path}")
        plt.close()
    
    def plot_scatter(self, targets, outputs, output_dir, file_prefix):
        """绘制并保存预测-实际散点图"""
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, outputs, alpha=0.5, s=1)
        
        min_val = min(targets.min(), outputs.min())
        max_val = max(targets.max(), outputs.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        rmse = np.sqrt(mean_squared_error(targets, outputs))
        r2 = r2_score(targets, outputs)
        mae = mean_absolute_error(targets, outputs)
        
        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f'$R^2$: {r2:.4f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f'MAE: {mae:.2f}', transform=plt.gca().transAxes)
        
        plt.title('局部微调：预测-实际散点图')
        plt.xlabel('实际值 (LST)')
        plt.ylabel('预测值 (LST)')
        plt.grid(True)
        
        scatter_path = os.path.join(output_dir, f"local_finetune_{file_prefix}_scatter.png")
        plt.savefig(scatter_path)
        logger.info(f"预测-实际散点图已保存到: {scatter_path}")
        plt.close()

# ===========================
# 独立调试入口
# ===========================
if __name__ == "__main__":
    logger.info("开始调试局部微调模块")
    
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    global_finetune_dir = r"G:\CNN_SpatialDownscaling\output\global_finetune"
    local_finetune_dir = r"G:\CNN_SpatialDownscaling\output\local_finetune"
    
    try:
        # 注意: 运行此调试代码前，请确保已运行 finetune_global.py 产生了对应的全局微调模型文件
        # 或者手动创建文件以供测试
        test_date = datetime(2018, 1, 1)
        
        finetuner = LocalFineTuner(config_path, global_finetune_dir, local_finetune_dir)
        logger.info(f"开始对 {test_date.strftime('%Y-%m-%d')} 进行局部微调")
        
        finetune_model_path = finetuner.fine_tune(test_date, is_debug=False)
        '''
        #训练全年每日模型
        for date in pd.date_range(start='2018-01-01', end='2018-12-31', freq='D'):
            logger.info(f"开始对 {date} 进行局部微调")
            finetuner_date = f"finetuner_"+ date.strftime('%Y%m%d')
            finetuner_date = finetuner.fine_tune(date, is_debug=True)
        '''
        
        if finetune_model_path:
            logger.info("调试模式微调成功，请检查输出目录。")
        else:
            logger.error("调试模式微调失败。")
            
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)