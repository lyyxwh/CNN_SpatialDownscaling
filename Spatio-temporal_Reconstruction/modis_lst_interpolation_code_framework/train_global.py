import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from datetime import datetime
import logging
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import gc

from data_loader import DataLoader as LSTDataLoader
from preprocess import LSTDataset, DataCollator
from model import LSTTransformer, create_model_config
from visualization_utils import  plot_training_history, plot_hexbin_scatter, plot_scatter, plot_hyperparameter_search

# 配置日志：同时输出到文件和控制台
log_dir = os.path.join(os.path.dirname(__file__), '../output/global_pretrain')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train_global_pretrain2.log')

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

class GlobalPretrainer:
    """全局预训练器 - 负责在全局CLDAS数据上预训练Transformer模型"""
    
    def __init__(self, config_path, output_dir):
        """
        初始化全局预训练器
        
        Args:
            config_path: 配置文件路径
            output_dir: 模型和结果输出目录
        """
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 初始化数据加载器
        self.data_loader = LSTDataLoader(config_path)
        
        # 设备配置
       
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

        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 模型配置参数
        self.model_params = self.config['model_params']
        self.best_hyperparams = {}  # 存储每月最优超参数
        
    def prepare_data(self, target_date, debug_mode=False):
        """
        准备指定月份的全局预训练数据
        
        Args:
            target_date: 目标日期（将使用该日期所在的整个月）
            debug_mode: 是否使用小批量数据进行调试
            
        Returns:
            train_dataset, val_dataset, test_dataset, scaler_dict, train_loader, val_loader, test_loader
        """
        logger.info(f"开始准备 {target_date.strftime('%Y-%m')} 月的全局预训练数据")
        
        # 定义缓存文件路径
        cache_dir = os.path.join(self.output_dir, 'cached_data')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_path = os.path.join(cache_dir, f"preprocessed_{target_date.strftime('%Y-%m')}.pt")
        
        # 尝试从缓存文件加载数据
        dataset, scaler_dict = LSTDataset.load_processed_data(cache_file_path)
        
        if dataset is None:
            # 如果缓存文件不存在，则加载原始数据并进行处理
            logger.info("未找到缓存文件，开始加载原始数据并进行预处理")
            data = self.data_loader.load_data(target_date, is_global=True, is_pretrain=True)
            if data is None:
                raise ValueError(f"无法加载 {target_date.strftime('%Y-%m')} 月的数据")
            
            logger.info(f"原始数据加载成功: {len(data['features']['lat'])} 个样本")
            
            # 创建数据集并进行处理
            dataset = LSTDataset(data['features'], data['target'], is_train=True, is_global=True)
            scaler_dict = dataset.scaler_dict
            
            # 保存处理后的数据以供下次使用
            #dataset.save_processed_data(cache_file_path)
        
        # 固定随机种子，保证划分可复现
        split_generator = torch.Generator().manual_seed(42)

        # 如果是调试模式，只使用10%的数据，且采样可复现
        if debug_mode:
            logger.info("调试模式开启，使用10%的数据进行调试")
            total_size = int(len(dataset) * 0.1)
            indices = torch.randperm(len(dataset), generator=split_generator)[:total_size]
            dataset = Subset(dataset, indices)

        # 按8:1:1划分数据集，划分可复现
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=split_generator
        )
        
        logger.info(f"数据集划分完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")

        # 创建数据加载器
        batch_size = self.model_params['batch_size']  # 默认使用第一个batch_size
        # 获取底层dataset以访问get_embedding_layers
        base_dataset = dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        try:
            embedding_layers = base_dataset.get_embedding_layers()
        except AttributeError:
            logger.warning("get_embedding_layers 未在数据集定义，使用空嵌入层")
            embedding_layers = {}
        collator = DataCollator(embedding_layers)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

        return train_dataset, val_dataset, test_dataset, scaler_dict, train_loader, val_loader, test_loader

    def create_model(self, train_dataset, n_layers, dropout):
        """
        根据训练数据特征维度创建模型
        
        Args:
            train_dataset: 训练数据集
            n_layers: Transformer层数
            dropout: Dropout率
            
        Returns:
            model: 创建的模型
        """
        # 获取底层dataset
        base_dataset = train_dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        try:
            feature_dims = base_dataset.get_feature_dims()
        except AttributeError:
            logger.error("get_feature_dims 未在数据集定义")
            raise
        
        num_features = feature_dims['num_features']
        time_features = feature_dims['time_features']
        loc_features = feature_dims['loc_features']
        cat_features = {name: info['num_classes'] for name, info in feature_dims['embedding_info'].items()}
        
        logger.info(f"特征维度: 数值={num_features}, 时间={time_features}, "
                    f"地理={loc_features}, 分类={cat_features}")
        
        # 创建模型配置
        config = create_model_config(
            num_features=num_features,
            time_features=time_features,
            loc_features=loc_features,
            cat_features=cat_features,
            d_model=512,
            n_heads=8,
            n_layers=n_layers,
            dropout=dropout
        )
        
        model = LSTTransformer(config).to(self.device)
        return model
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch, scaler_dict=None):
        """
        训练一个epoch
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            epoch: 当前epoch
            scaler_dict: 标准化器字典（包含target scaler）
            
        Returns:
            avg_loss: 平均训练损失（标准化）
            avg_true_loss: 平均训练损失（反标准化，K）
        """
        model.train()
        total_loss = 0.0
        total_true_loss = 0.0
        
        # 修正：用于收集整个epoch的预测值和目标值，但这里我们只在最后进行反标准化
        all_raw_predictions = []
        all_raw_targets = []
        
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移动到设备
            batch_data = {
                'num': batch['num_features'].to(self.device),
                'time': batch['time_features'].to(self.device),
                'loc': batch['loc_features'].to(self.device),
                'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
            }
            target = batch['target'].to(self.device).unsqueeze(1)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 收集原始的、未反标准化的预测值和目标值
            all_raw_predictions.extend(output.detach().cpu().numpy())
            all_raw_targets.extend(target.detach().cpu().numpy())
            
            # 计算每个批次的反标准化损失，仅用于进度条显示
            if scaler_dict and 'target' in scaler_dict:
                predictions = output.detach().cpu().numpy().reshape(-1, 1)
                targets = target.detach().cpu().numpy().reshape(-1, 1)
                target_scaler = scaler_dict['target']
                predictions_inv = target_scaler.inverse_transform(predictions).flatten()
                targets_inv = target_scaler.inverse_transform(targets).flatten()
                true_loss = mean_squared_error(targets_inv, predictions_inv)
                total_true_loss += true_loss
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'true_mse': true_loss
                })

        end_time = time.time()
        avg_loss = total_loss / len(train_loader)
        avg_true_loss = total_true_loss / len(train_loader) if scaler_dict and 'target' in scaler_dict else None

        # 修正：在所有批次处理完后，对整体数据进行一次性反标准化，并计算最终指标
        if scaler_dict and 'target' in scaler_dict:
            targets_inv = scaler_dict['target'].inverse_transform(np.array(all_raw_targets))
            outputs_inv = scaler_dict['target'].inverse_transform(np.array(all_raw_predictions))
            
            final_r2 = r2_score(targets_inv, outputs_inv)
            final_rmse = np.sqrt(mean_squared_error(targets_inv, outputs_inv))
            final_mae = mean_absolute_error(targets_inv, outputs_inv)
            
            log_message = f"Epoch {epoch+1} 完成，耗时: {end_time - start_time:.2f} 秒, 平均损失(标准化): {avg_loss:.6f}, " \
                        f"R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}"
        else:
            log_message = f"Epoch {epoch+1} 完成，耗时: {end_time - start_time:.2f} 秒, 平均损失(标准化): {avg_loss:.6f}"
        
        if avg_true_loss is not None:
            log_message += f", 平均损失(K): {avg_true_loss:.6f}" 
            
        logger.info(log_message)
        
        return avg_loss, avg_true_loss
    
    def evaluate(self, model, data_loader, criterion, scaler_dict=None):
        """
        评估模型性能
        
        Args:
            model: 模型
            data_loader: 数据加载器
            criterion: 损失函数
            scaler_dict: 标准化器字典（包含target scaler）
            
        Returns:
            metrics: 包含各种评估指标的字典
        """
        model.eval()
        total_loss = 0.0
        total_true_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch_data = {
                    'num': batch['num_features'].to(self.device),
                    'time': batch['time_features'].to(self.device),
                    'loc': batch['loc_features'].to(self.device),
                    'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                }
                target = batch['target'].to(self.device).unsqueeze(1)
                
                output = model(batch_data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                
                # 计算反标准化损失
                if scaler_dict and 'target' in scaler_dict:
                    predictions = output.cpu().numpy()
                    targets = target.cpu().numpy()
                    target_scaler = scaler_dict['target']
                    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                    targets = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
                    true_loss = mean_squared_error(targets, predictions)
                    total_true_loss += true_loss
                    all_predictions.extend(predictions)
                    all_targets.extend(targets)
        
        # 计算评估指标
        avg_loss = total_loss / len(data_loader)
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        if scaler_dict and 'target' in scaler_dict:
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)
        else:
            mse = rmse = mae = r2 = None
        
        metrics = {
            'loss': avg_loss,  # 标准化后的损失
            'true_mse': mse,   # 反标准化后的MSE (K)
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'avg_true_loss': total_true_loss / len(data_loader) if scaler_dict and 'target' in scaler_dict else None
        }
        
        return metrics
    
    def hyperparameter_search(self, train_dataset, val_dataset, month, num_epochs=30):
        """
        超参数搜索
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            month: 当前月份
            num_epochs: 搜索时的训练轮数
            
        Returns:
            best_params: 最优超参数
        """
        logger.info(f"开始 {month} 月的超参数搜索")
        
        # 超参数搜索空间
        search_space = {
            'n_layers': self.model_params['n_layers'],
            'learning_rate': self.model_params['learning_rate'],
            'batch_size': self.model_params['batch_size'],
            'dropout': self.model_params['dropout']
        }
        
        best_val_loss = float('inf')
        best_params = {}


        
        for n_layers in search_space['n_layers']:
            for lr in search_space['learning_rate']:
                for batch_size in search_space['batch_size']:
                    for dropout in search_space['dropout']:
                        logger.info(f"测试超参数: layers={n_layers}, lr={lr}, "
                                    f"batch_size={batch_size}, dropout={dropout}")
                        
                        # 创建数据加载器
                        base_dataset = train_dataset
                        while isinstance(base_dataset, Subset):
                            base_dataset = base_dataset.dataset
                        try:
                            embedding_layers = base_dataset.get_embedding_layers()
                        except AttributeError:
                            logger.warning("get_embedding_layers 未在数据集定义，使用空嵌入层")
                            embedding_layers = {}
                        collator = DataCollator(embedding_layers)
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
                        
                        # 创建模型
                        model = self.create_model(train_dataset, n_layers, dropout)
                        
                        # 创建优化器和损失函数
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        criterion = nn.MSELoss()
                        
                        # 获取scaler_dict
                        base_dataset = train_dataset
                        while isinstance(base_dataset, Subset):
                            base_dataset = base_dataset.dataset
                        scaler_dict = base_dataset.scaler_dict
                        #dataset = LSTDataset(data['features'], data['target'], is_train=True)
                        #scaler_dict = dataset.scaler_dict
                        # 训练
                        for epoch in range(num_epochs):   
                            train_loss, train_true_loss = self.train_epoch(model, train_loader, optimizer, criterion, epoch, scaler_dict)
                        
                        # 评估
                        val_metrics = self.evaluate(model, val_loader, criterion, scaler_dict)
                        val_loss = val_metrics['loss']
                        val_true_mse = val_metrics['true_mse']
                        
                        logger.info(f"验证损失(标准化): {val_loss:.6f}, 验证MSE(K): {val_true_mse:.6f}, R²: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = {
                                'n_layers': n_layers,
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'dropout': dropout
                            }
                            
                        # 清理GPU内存
                        del model, optimizer
                        torch.cuda.empty_cache()

        logger.info(f"{month} 月最优超参数: {best_params}, 验证损失: {best_val_loss:.6f}, 验证MSE(K): {val_true_mse:.6f}, R²: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
        return best_params


    def train_model(self, target_date, num_epochs=None, debug_mode=False):
        """
        训练指定月份的全局预训练模型
        
        Args:
            target_date: 目标日期
            num_epochs: 训练轮数，如果为None则使用配置文件中的默认值
            debug_mode: 是否使用小批量数据进行调试
            
        Returns:
            model: 训练好的模型
            metrics: 训练和验证指标
        """
        month = target_date.month
        month_str = target_date.strftime('%Y%m')
        
        if num_epochs is None:
            num_epochs = self.model_params['epochs_pretrain']
        
        logger.info(f"开始训练 {target_date.strftime('%Y-%m')} 月的全局预训练模型")
        
        # 准备数据
        train_dataset, val_dataset, test_dataset, scaler_dict, train_loader, val_loader, test_loader = self.prepare_data(target_date, debug_mode)
          # 获取完整的scaler_dict（包含所有特征的scaler）
        base_dataset = train_dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        
        complete_scaler_dict = base_dataset.scaler_dict  # 这个包含所有scaler
        
        logger.info(f"完整scaler_dict包含的键: {list(complete_scaler_dict.keys())}")
        hyperparam_search_epochs = 4 if  debug_mode else num_epochs
        
        # 超参数搜索
        best_params ={
                                'n_layers': 6,
                                'learning_rate': 0.0001,
                                'batch_size': 512,
                                'dropout': 0.1
                            }  #self.hyperparameter_search(train_dataset, val_dataset, month, num_epochs=hyperparam_search_epochs)
        self.best_hyperparams[month_str] = best_params

        # 创建数据加载器
        base_dataset = train_dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        try:
            embedding_layers = base_dataset.get_embedding_layers()
        except AttributeError:
            logger.warning("get_embedding_layers 未在数据集定义，使用空嵌入层")
            embedding_layers = {}
        collator = DataCollator(embedding_layers)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collator)        
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collator)
        # 超参数搜索结果图
        #self.visualize_hyperparameter_search_results(month, best_params)
        # 使用最优超参数创建最终模型
        model = self.create_model(train_dataset, best_params['n_layers'], best_params['dropout'])
        
        # 创建优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()
        
        # 学习率调度器：使用余弦退火
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 训练历史记录
        train_losses = []
        train_true_losses = []
        val_losses = []
        val_true_mses = []
        val_r2_scores = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3  # 早停容忍度
        
        logger.info(f"开始训练，使用超参数: {best_params}")
        logger.info(f"预计总训练时间: {num_epochs * (len(train_loader) * 0.01 + 0.5):.2f} 秒 (估算)")
        
        best_model_state = None
        best_epoch = -1
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss, train_true_loss = self.train_epoch(model, train_loader, optimizer, criterion, epoch, scaler_dict)
            train_losses.append(train_loss)
            if train_true_loss is not None:
                train_true_losses.append(train_true_loss)

            # 验证阶段
            val_metrics = self.evaluate(model, val_loader, criterion, scaler_dict)
            val_loss = val_metrics['loss']
            val_true_mse = val_metrics['true_mse']
            val_r2 = val_metrics['r2']

            val_losses.append(val_loss)
            if val_true_mse is not None:
                val_true_mses.append(val_true_mse)
            val_r2_scores.append(val_r2)

            # 学习率调度
            scheduler.step()

            log_message = f"Epoch {epoch+1}/{num_epochs}: Train Loss(标准化)={train_loss:.6f}, Val Loss(标准化)={val_loss:.6f}"
            if train_true_loss is not None and val_true_mse is not None:
                log_message += f", Train MSE(K)={train_true_loss:.6f}, Val MSE(K)={val_true_mse:.6f}"
            log_message += f", Val RMSE={val_metrics['rmse']:.4f}, Val R²={val_r2:.4f}"
            logger.info(log_message)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 记录最佳模型参数
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                logger.info(f"Epoch {epoch+1}: 新的最佳验证损失: {best_val_loss:.6f}")
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"验证损失连续 {patience} 个epoch未改善，提前停止训练")
                    best_model_state = model.state_dict()
                    best_epoch = epoch
                    break

        # 训练结束后保存最佳模型
        if best_model_state is not None:
            model_path = os.path.join(self.output_dir, f'global_pretrain_{month_str}.pt')
            
            # 保存完整的checkpoint
            checkpoint = {
                'model_state_dict': best_model_state,
                'config': model.config,
                'scaler_dict': complete_scaler_dict,  # 使用完整的scaler_dict
                'hyperparams': best_params,
                'epoch': best_epoch,
                'val_loss': best_val_loss
            }
            
            torch.save(checkpoint, model_path)
            logger.info(f"最终保存最佳模型到: {model_path}")
            logger.info(f"保存的scaler包含 {len(complete_scaler_dict)} 个键")
        
        # 另外保存一个可读的scaler配置文件（用于调试和验证）
        scaler_config_path = os.path.join(self.output_dir, f'pretrain_scaler_config_{month_str}.json')
        scaler_config = {}
        
        for key, scaler in complete_scaler_dict.items():
            if hasattr(scaler, 'mean_'):
                scaler_config[key] = {
                    'mean': float(scaler.mean_[0]) if hasattr(scaler.mean_, '__getitem__') else float(scaler.mean_),
                    'scale': float(scaler.scale_[0]) if hasattr(scaler.scale_, '__getitem__') else float(scaler.scale_),
                    'type': 'StandardScaler'
                }
            elif hasattr(scaler, 'num_classes'):
                scaler_config[key] = {
                    'num_classes': int(scaler.num_classes) if hasattr(scaler, 'num_classes') else 'N/A',
                    'embed_dim': int(scaler.embed_dim) if hasattr(scaler, 'embed_dim') else 'N/A',
                    'type': 'Embedding'
                }
        
        with open(scaler_config_path, 'w', encoding='utf-8') as f:
            json.dump(scaler_config, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Scaler配置已保存到: {scaler_config_path}")
        
        # 最终测试
        test_metrics = self.evaluate(model, test_loader, criterion, scaler_dict)
        logger.info(f"测试集性能: MSE(K)={test_metrics['true_mse']:.6f}, "
                    f"RMSE={test_metrics['rmse']:.6f}, MAE={test_metrics['mae']:.6f}, "
                    f"R²={test_metrics['r2']:.4f}")
        # 获取测试集的实际值和预测值
        all_test_targets = []
        all_test_outputs = []
        # 重新运行测试集评估以捕获所有预测
        with torch.no_grad():
            for batch in test_loader:
                # ... 前向传播和数据处理 ...
                batch_data = {
                    'num': batch['num_features'].to(self.device),
                    'time': batch['time_features'].to(self.device),
                    'loc': batch['loc_features'].to(self.device),
                    'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                }
                outputs = model(batch_data).cpu().numpy()
                targets = batch['target'].cpu().numpy()

                # 反标准化
                targets_inv = scaler_dict['target'].inverse_transform(targets.reshape(-1, 1)).flatten()
                outputs_inv = scaler_dict['target'].inverse_transform(outputs.reshape(-1, 1)).flatten()

                all_test_targets.extend(targets_inv)
                all_test_outputs.extend(outputs_inv)       
        
        # 保存训练历史和结果
        metrics = {
            'train_losses': train_losses,
            'train_true_losses': train_true_losses,
            'val_losses': val_losses,
            'val_true_mses': val_true_mses,
            'val_r2_scores': val_r2_scores,
            'test_metrics': test_metrics,
            'best_hyperparams': best_params
        }
        
        # 保存超参数配置
        hyperparams_path = os.path.join(self.output_dir, f'pretrain_hyperparams_{month_str}.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, val_losses, val_r2_scores, val_true_mses, month_str)
        plot_training_history(
            {'train_losses': train_losses, 'val_losses': val_losses, 
            'val_r2_scores': val_r2_scores, 'val_true_mses': val_true_mses},
            os.path.join(self.output_dir, f'training_curves_{month_str}.png'),
            title_prefix=f'{month_str} Global Pretrain'
        )
        plot_hexbin_scatter(np.array(all_test_targets), np.array(all_test_outputs),
                        os.path.join(self.output_dir, f"global_pretrain_{month_str}_hexbin.png"),
                        title='Global_Pretrain '+month_str,
                        xlabel='Reference LST (K)',
                        ylabel='Predicted LST (K)',
                        figsize=(8, 8), gridsize=200)
        plot_scatter(np.array(all_test_targets), np.array(all_test_outputs),
                        os.path.join(self.output_dir, f"global_pretrain_{month_str}_scatter.png"),
                        month_str,dpi=300)

        return model, metrics
   
    def plot_training_curves(self, train_losses, val_losses, val_r2_scores, val_true_mses, month_str):
        """
        绘制训练曲线
        
        Args:
            train_losses: 训练损失列表（标准化）
            val_losses: 验证损失列表（标准化）
            val_r2_scores: 验证R²分数列表
            val_true_mses: 验证MSE列表（反标准化，K）
            month_str: 月份字符串
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        
        # 标准化损失曲线
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='训练损失(标准化)', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='验证损失(标准化)', linewidth=2)
        ax1.set_title(f'{month_str}月全局预训练 - 标准化损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss (Standardized)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 反标准化MSE曲线
        if val_true_mses:
            ax2.plot(epochs, val_true_mses, 'g-', label='验证MSE(K)', linewidth=2)
            ax2.set_title(f'{month_str}月全局预训练 - 反标准化MSE曲线')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MSE (K)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # R²曲线
        ax3.plot(epochs, val_r2_scores, 'm-', label='验证R$^2$', linewidth=2)
        ax3.set_title(f'{month_str}月全局预训练 - R$^2$曲线')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('R$^2$ Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(self.output_dir, f'training_curves_{month_str}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线保存到: {plot_path}")


    def train_yearly_models(self, year=2018):
        """
        训练一整年的全局预训练模型
        
        Args:
            year: 目标年份
        """
        logger.info(f"开始训练 {year} 年全部12个月的全局预训练模型")
        
        all_metrics = {}
        
        for month in range(8, 13):
            target_date = datetime(year, month, 15)  # 使用每月15日作为代表日期

            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"开始训练第 {month} 月模型")
                logger.info(f"{'='*50}")
                
                model, metrics = self.train_model(target_date, num_epochs=100, debug_mode=False)
                all_metrics[f"{month:02d}"] = metrics
                
                logger.info(f"第 {month} 月模型训练完成")
                
            except Exception as e:
                logger.error(f"第 {month} 月模型训练失败: {str(e)}")
                continue
            finally:
                    logger.info("开始执行模型训练后清理操作...")
    
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
        
        # 保存所有月份的训练结果
        results_path = os.path.join(self.output_dir, f'yearly_results_{year}.json')
        with open(results_path, 'w') as f:
            # 将numpy数组转换为列表以便JSON序列化
            serializable_metrics = {}
            for month, metrics in all_metrics.items():
                serializable_metrics[month] = {
                    'train_losses': metrics['train_losses'],
                    'train_true_losses': metrics['train_true_losses'],
                    'val_losses': metrics['val_losses'],
                    'val_true_mses': metrics['val_true_mses'],
                    'val_r2_scores': metrics['val_r2_scores'],
                    'test_metrics': {k: float(v) if isinstance(v, np.ndarray) else v 
                                   for k, v in metrics['test_metrics'].items()},
                    'best_hyperparams': metrics['best_hyperparams']
                }
            json.dump(serializable_metrics, f, indent=4)
        
        logger.info(f"年度训练结果保存到: {results_path}")
        logger.info(f"{year} 年全局预训练完成!")

# ===========================
# 独立调试入口
# ===========================
if __name__ == "__main__":
    logger.info("开始调试全局预训练模块")
    
    # 配置路径
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    output_dir = r"G:\CNN_SpatialDownscaling\output\global_pretrain4"
    
    try:
        # 创建预训练器
        pretrainer = GlobalPretrainer(config_path, output_dir)
        for month in [1, 7]:#range(3, 4):
            target_date = datetime(2018, month, 15)
            logger.info(f"\n{'='*50}")
            logger.info(f"开始训练第 {month} 月模型")
            logger.info(f"{'='*50}")
            
            model, metrics = pretrainer.train_model(target_date, num_epochs=100, debug_mode=False)
            logger.info(f"第 {month} 月模型训练完成")
            
            #清理内存和显存
            logger.info("开始执行模型训练后清理操作...")

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
        '''    
        # 测试单月训练
        target_date = datetime(2018, 2, 15)
        logger.info(f"测试训练 {target_date.strftime('%Y-%m')} 月的全局预训练模型")
        
        # 在调试模式下，使用少量数据和较少轮次进行测试
        model, metrics = pretrainer.train_model(target_date, num_epochs=100, debug_mode=False)
        
        logger.info("单月模型训练测试完成")
        
        
        # 如果需要训练全年模型，取消下面的注释
        logger.info("开始训练全年模型")
        pretrainer.train_yearly_models(2018)
        ''' 
    except Exception as e:
        logger.error(f"调试失败: {str(e)}")
        import traceback
        traceback.print_exc()