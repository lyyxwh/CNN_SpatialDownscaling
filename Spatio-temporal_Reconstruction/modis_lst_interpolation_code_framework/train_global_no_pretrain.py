import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from datetime import datetime
import logging
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from data_loader import DataLoader as LSTDataLoader
from preprocess import LSTDataset, DataCollator
from model import LSTTransformer, create_model_config

# 配置日志
# 配置日志：同时输出到文件和控制台
log_dir = os.path.join(os.path.dirname(__file__), 'output/global_no_pretrain')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train_global.log')

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

class GlobalTrainer:
    """全局训练器 - 负责在全局CLDAS数据上从零开始训练Transformer模型"""
    
    def __init__(self, config_path, output_dir):
        """
        初始化全局训练器
        
        Args:
            config_path: 配置文件路径
            output_dir: 模型和结果输出目录
        """
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.device = torch.device(self.config.get('device', 'cpu') if torch.cuda.is_available() else "cpu")
        self.train_params = self.config['train_params']['global_finetune'] # 沿用之前的配置参数
        self.model_params = self.config['model_params']

    def train(self, start_date, end_date, is_debug=False):
        """
        在指定日期范围内训练一个全局模型
        
        Args:
            start_date (datetime): 训练数据开始日期
            end_date (datetime): 训练数据结束日期
            is_debug (bool): 是否为调试模式
            
        Returns:
            tuple: (训练好的模型, 评估指标)
        """
        # 1. 加载所有训练数据
        '''
        logger.info(f"开始加载 {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} 的全局训练数据...")
        loader = LSTDataLoader(self.config_path)
        for target_date in tqdm(range(start_date, end_date), desc="加载训练数据"):
            logger.info(f"开始加载 {target_date.strftime('%Y-%m-%d')} 的全局微调数据 (±15天)...")
            data = loader.load_data(target_date, is_global=True, is_pretrain=False)'''
        loader = LSTDataLoader(self.config_path)

        data = loader.load_data(start_date, is_global=True, is_pretrain=False)
        if data is None or data['features'] is None or len(data['features']['lat']) == 0:
            logger.error("无法加载任何全局训练数据，训练中止。")
            return None, None
            
        # 2. 从零开始创建模型配置和数据集

        dataset = LSTDataset(
            data['features'], 
            data['target'], 
            is_train=True
        )
        feature_dims = dataset.get_feature_dims()
        #embedding_dims = dataset.embedding_dims

        # 接着创建模型配置
        # 接着创建模型配置
        model_config = create_model_config(
            num_features=feature_dims['num_features'],
            time_features=feature_dims['time_features'],
            loc_features=feature_dims['loc_features'],
            cat_features=feature_dims.get('cat_features', {}),
            d_model=self.config['model_params']['d_model'],
            n_heads=self.config['model_params']['n_heads'],
            n_layers=self.config['model_params']['n_layers'][1],
            dropout=self.config['model_params']['dropout'][0]
        )
        # 将嵌入维度添加到模型配置中
        model_config['embedding_dims'] = dataset.embedding_dims

        model = LSTTransformer(model_config).to(self.device)
        logger.info(f"成功创建新模型，参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 获取数据集在初始化时生成的标准化器
        scaler_dict = dataset.scaler_dict

        # 划分训练集和验证集 (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        dataset_indices = list(range(len(dataset)))
        np.random.shuffle(dataset_indices)


        # 调试模式下使用少量数据
        if is_debug:
            # 使用总数据的 10% 进行调试
            logger.info("调试模式: 使用总数据的 10% 进行训练和验证")
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
            #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_indices = dataset_indices[:train_size]
            val_indices = dataset_indices[train_size:]
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices) 
            logger.info(f"加载全局训练数据成功，训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")       
            
        collator = DataCollator(dataset.get_embedding_layers())
        batch_size = self.model_params['batch_size'][0]
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

        # 3. 训练过程
        optimizer = optim.Adam(model.parameters(), lr=self.model_params['learning_rate'][1])
        criterion = nn.MSELoss()
        
        epochs = self.train_params['epochs_debug'] if is_debug else self.train_params['epochs']
        patience = self.train_params['early_stopping_patience']
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        # 打印实际超参数信息
        logger.info(f"实际训练超参数: n_layers={self.config['model_params']['n_layers'][1]},"
                    f"dropout={self.config['model_params']['dropout'][0]}, batch_size={batch_size},"
                     f" learning_rate={optimizer.param_groups[0]['lr']}, patience={patience}")

        history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_r2': []}

        start_time = time.time()
        for epoch in range(epochs):
            # 训练阶段
            model.train()
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
                outputs = model(batch_data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            model.eval()
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
                    
                    outputs = model(batch_data)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
                    
                    all_val_targets.extend(targets.cpu().numpy())
                    all_val_outputs.extend(outputs.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            # 将预测结果反标准化
            all_val_targets_inv = scaler_dict['target'].inverse_transform(np.array(all_val_targets))
            all_val_outputs_inv = scaler_dict['target'].inverse_transform(np.array(all_val_outputs))

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
                best_model_state = model.state_dict()
                logger.info(f"验证集损失降低，保存当前最佳模型。")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停：验证集损失连续 {patience} 个 epoch 未降低，停止训练。")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info("已加载最佳模型权重。")
            
        # 4. 保存模型、标准化器和历史记录
        output_path = os.path.join(self.output_dir, f"global_trained_{start_date.strftime('%Y%m%d%H')}.pt")
        
        # 保存整个模型（含参数、配置和标准化器）
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model_config,
            'scaler_dict': scaler_dict,
            'history': history,
        }, output_path)
        
        logger.info(f"全局训练模型已保存到: {output_path}")
        
        self.plot_metrics(history, self.output_dir)
        self.plot_scatter(all_val_targets_inv, all_val_outputs_inv, self.output_dir)

        return model, history

    def plot_metrics(self, history, output_dir):
        """绘制并保存训练和验证指标图表"""
        plt.figure(figsize=(12, 5))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('全局训练：训练与验证损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 绘制 RMSE 和 R2
        plt.subplot(1, 2, 2)
        plt.plot(history['val_rmse'], label='验证集RMSE', color='orange')
        plt.plot(history['val_r2'], label='验证集$R^2$', color='green')
        plt.title('全局训练：验证集RMSE与$R^2$')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"global_training_metrics.png")
        plt.savefig(plot_path)
        logger.info(f"全局训练指标图表已保存到: {plot_path}")
        plt.close()
    
    def plot_scatter(self, targets, outputs, output_dir):
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
        
        plt.title('全局训练：预测-实际散点图')
        plt.xlabel('实际值 (LST)')
        plt.ylabel('预测值 (LST)')
        plt.grid(True)
        
        scatter_path = os.path.join(output_dir, f"global_training_scatter.png")
        plt.savefig(scatter_path)
        logger.info(f"预测-实际散点图已保存到: {scatter_path}")
        plt.close()

# ===========================
# 独立调试入口
# ===========================
if __name__ == "__main__":
    logger.info("开始调试全局训练模块 (无预训练)")
    
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    output_dir = r"G:\CNN_SpatialDownscaling\output\global_trained"
    
    try:
        trainer = GlobalTrainer(config_path, output_dir)
        start_date = datetime(2018, 1, 15)
        end_date = datetime(2018, 1, 15)
        
        logger.info(f"测试训练 {start_date.strftime('%Y-%m-%d')} 日的全局模型")
        
        # 在调试模式下，使用少量数据和较少轮次进行测试
        model, metrics = trainer.train(start_date, end_date, is_debug=True)
        
        if model:
            logger.info("调试模式训练成功，请检查输出目录。")
        else:
            logger.error("调试模式训练失败。")
            
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)