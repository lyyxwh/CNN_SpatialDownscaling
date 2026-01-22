"""
局部微调模块 - v2修复版本
关键修复：
1. 确保使用全局预训练模型的完整scaler_dict
2. 添加embedding_dims格式转换
3. 实现真正的知识蒸馏
4. 增强日志记录和验证
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
from visualization_utils import  plot_training_history, plot_hexbin_scatter, plot_model_scatter

# 配置日志
log_dir = os.path.join(os.path.dirname(__file__), '../output/local_finetune')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'local_finetune_0115.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 文件日志
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Times New Roman','SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LocalFineTuner:
    """局部微调器 - 直接从全局模型加载并进行局部数据微调"""
    
    def __init__(self, config_path, global_pretrain_dir, local_finetune_dir):
        """
        初始化局部微调器
        
        Args:
            config_path: 配置文件路径
            global_pretrain_dir: 全局预训练模型目录
            local_finetune_dir: 局部微调模型输出目录
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
        """
        加载全局预训练模型和完整的标准化器
        
        Args:
            target_date: 目标日期
            
        Returns:
            model: 加载的模型
            pretrain_scaler_dict: 完整的标准化器字典
        """
        # 构造全局模型路径（按月）
        file_prefix = target_date.strftime('%Y%m')
        
        model_path = os.path.join(self.global_pretrain_dir, f'global_pretrain_{file_prefix}.pt')
        
        if not os.path.exists(model_path):
            logger.error(f"全局预训练模型文件不存在: {model_path}")
            return None, None
        
        # 加载checkpoint
        logger.info(f"正在加载全局预训练模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 验证checkpoint结构
        required_keys = ['model_state_dict', 'config', 'scaler_dict']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise KeyError(f"Checkpoint缺少必需的键: {missing_keys}")
        
        # 获取scaler_dict
        pretrain_scaler_dict = checkpoint['scaler_dict']
        
        # 详细验证scaler_dict内容
        logger.info(f"加载的scaler_dict包含 {len(pretrain_scaler_dict)} 个键:")
        
        # 统计不同类型的scaler
        standard_scalers = []
        embedding_configs = []
        
        for key, value in pretrain_scaler_dict.items():
            if hasattr(value, 'mean_') and hasattr(value, 'scale_'):
                # StandardScaler
                standard_scalers.append(key)
                logger.info(f"  {key}: StandardScaler (mean={value.mean_[0]:.3f}, std={value.scale_[0]:.3f})")
            elif key == 'cat_features':
                # 分类特征配置
                embedding_configs.append(key)
                logger.info(f"  {key}: 分类特征配置 (包含 {len(value)} 个特征)")
                for cat_name, cat_info in value.items():
                    logger.info(f"    - - {cat_name}: {cat_info['num_classes']} 类, 嵌入维度={cat_info['embed_dim']}")
            else:
                logger.info(f"  {key}: {type(value).__name__}")
        
        logger.info(f"StandardScaler总数: {len(standard_scalers)}")
        logger.info(f"分类特征配置数: {len(embedding_configs)}")
        
        # 验证必需的scaler
        required_scalers = ['target', 'cat_features']
        missing_scalers = [s for s in required_scalers if s not in pretrain_scaler_dict]
        if missing_scalers:
            raise ValueError(f"scaler_dict缺少必需的标准化器: {missing_scalers}")
        
        # 验证target scaler的合理性
        target_scaler = pretrain_scaler_dict['target']
        target_mean = target_scaler.mean_[0]
        target_std = target_scaler.scale_[0]
        
        logger.info(f"目标变量scaler验证:")
        logger.info(f"  均值: {target_mean:.2f} K")
        logger.info(f"  标准差: {target_std:.2f} K")
        
        # CLDAS/MODIS地表温度的合理范围检查
        if not (250 < target_mean < 310):
            logger.warning(f"⚠️  Target scaler的均值 {target_mean:.2f}K 超出合理范围 [250K, 310K]")
        if not (5 < target_std < 30):
            logger.warning(f"⚠️  Target scaler的标准差 {target_std:.2f}K 超出合理范围 [5K, 30K]")
        
        # 创建模型
        model_config = checkpoint['config']
        logger.info(f"模型配置: {model_config}")
        
        # 【修复1】: 转换embedding_dims格式以兼容LSTDataset
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
        logger.info(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, pretrain_scaler_dict, embedding_dims

    def fine_tune(self, target_date, is_debug=False):
        """
        对特定日期的局部数据进行微调
        
        Args:
            target_date: 目标日期
            is_debug: 是否调试模式
            
        Returns:
            str: 模型保存路径
        """
        # 1. 加载全局预训练模型和scaler
        global_model, pretrain_scaler_dict, embedding_dims = self.load_global_pretrain_model(target_date)
        if global_model is None:
            raise ValueError("无法加载全局预训练模型")
        
        # 2. 加载局部数据
        logger.info(f"开始加载 {target_date.strftime('%Y-%m-%d')} 的局部微调数据")
        local_data = self.data_loader.load_data(target_date, is_global=False, is_pretrain=False)
        
        if local_data is None:
            raise ValueError("无法加载局部数据")
        
        # 3. 创建数据集 - 使用全局的pretrain_scaler_dict
        logger.info("创建局部数据集,使用全局标准化器")
        logger.info(f"全局标准化器键: {list(pretrain_scaler_dict.keys())}")
        
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
        scaler_dict = local_dataset.scaler_dict
    
        # 验证使用的scaler是全局的
        logger.info(f"✓ 局部数据集scaler_dict键: {list(local_dataset.scaler_dict.keys())}")
        logger.info(f"✓ 确认使用全局target scaler (mean={scaler_dict['target'].mean_[0]:.2f}K)")
        
        # 固定随机种子，保证划分可复现
        split_generator = torch.Generator().manual_seed(42)
        # 4. 数据集划分
        if is_debug:
            # 调试模式下使用10%的数据集
            debug_size = int(0.1 * len(local_dataset))
            local_dataset = Subset(local_dataset, list(range(debug_size)))
            logger.info(f"调试模式: 使用前 {debug_size} 条数据进行微调")
            train_size = int(0.8 * debug_size)
            val_size = debug_size - train_size
        
        else:
            total_size = len(local_dataset)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(local_dataset, [train_size, val_size], generator=split_generator)
        
        logger.info(f"数据集划分: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}")
        
        # 5. 创建数据加载器
        batch_size = self.finetune_params['batch_size']
        
        # 🔧 关键修复: 访问底层dataset以获取embedding_layers
        base_dataset = local_dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        
        embedding_layers = base_dataset.get_embedding_layers()
        collator = DataCollator(embedding_layers)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        
        # 6. 创建局部模型 - 从全局模型复制
        local_model = LSTTransformer(global_model.config).to(self.device)
        local_model.load_state_dict(global_model.state_dict())
        
        # 冻结层
        freeze_layers = self.finetune_params['freeze_layers']
        local_model.freeze_layers(freeze_layers)
        
        logger.info(f"✓ 已冻结前 {freeze_layers} 层")
        trainable_params = sum(p.numel() for p in local_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in local_model.parameters())
        logger.info(f"✓ 可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
        
        # 7. 训练设置
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
            'train_mse': [],
            'train_kl': [],
            'train_rmse': [],
            'train_r2': [],
            'val_loss': [],
            'val_rmse': [],
            'val_r2': []
        }
        
        # 输出目录
        file_prefix = target_date.strftime('%Y%m%d')
        output_dir = os.path.join(self.local_finetune_dir)#, file_prefix
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        logger.info(f"开始局部微调: epochs={num_epochs}, batch_size={batch_size}, lr={self.finetune_params['learning_rate']}, kl_weight={kl_weight}")
        
        for epoch in range(num_epochs):
            local_model.train()
            global_model.eval()
            
            train_loss = 0.0
            train_mse = 0.0
            train_kl = 0.0
            
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
                
                # 知识蒸馏
                kl_loss = torch.tensor(0.0, device=self.device)
                if global_model is not None and kl_weight > 0:
                    with torch.no_grad():
                        global_outputs = global_model(batch_data).squeeze(-1)
                    kl_loss = criterion(outputs, global_outputs)
                
                # 总损失
                loss = mse_loss + kl_weight * kl_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(targets)
                train_mse += mse_loss.item() * len(targets)
                train_kl += kl_loss.item() * len(targets)
            
            train_loss /= len(train_loader.dataset)
            train_mse /= len(train_loader.dataset)
            train_kl /= len(train_loader.dataset)
            
            history['train_loss'].append(train_loss)
            history['train_mse'].append(train_mse)
            history['train_kl'].append(train_kl)
            
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
            
            target_scaler = pretrain_scaler_dict['target']
            val_preds_denorm = target_scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()
            val_targets_denorm = target_scaler.inverse_transform(val_targets.reshape(-1, 1)).flatten()
            
            val_rmse = np.sqrt(mean_squared_error(val_targets_denorm, val_preds_denorm))
            val_r2 = r2_score(val_targets_denorm, val_preds_denorm)
            
            history['val_rmse'].append(val_rmse)
            history['val_r2'].append(val_r2)

            logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} (MSE {train_mse:.4f} + KL {train_kl:.4f}), "
                        f"Val Loss {val_loss:.4f}, Val RMSE {val_rmse:.2f}K, Val R² {val_r2:.4f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}                
                counter = 0
                logger.info(f"  → 新的最佳验证损失: {best_val_loss:.6f},已保存模型状态")
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"早停于 epoch {epoch+1}")
                    break
            

        
        end_time = time.time()
        logger.info(f"局部微调完成，耗时 {(end_time - start_time)/60:.2f} 分钟。")
        
        # 8. 保存最佳模型
        if best_model_state:
            local_model.load_state_dict(best_model_state)
        
        final_model_path = os.path.join(output_dir, f'local_finetune_{file_prefix}.pt')
        
        # 保存完整checkpoint
        torch.save({
            'model_state_dict': local_model.state_dict(),
            'config': local_model.config,
            'scaler_dict': scaler_dict,  # 包含全局特征Scaler + 局部targetScaler,  # 保存完整的scaler_dict
            'hyperparams': {
                'freeze_layers': freeze_layers,
                'kl_weight': kl_weight,
                'learning_rate': self.finetune_params['learning_rate'],
                'batch_size': batch_size
            },
            'date': target_date.strftime('%Y%m%d'),
            'best_val_loss': best_val_loss,
            'final_metrics': {
                'val_rmse': history['val_rmse'][-1],
                'val_r2': history['val_r2'][-1]
            }
        }, final_model_path)
        
        logger.info(f"✓ 局部微调模型已保存到: {final_model_path}")
        logger.info(f"✓ 保存的scaler_dict包含 {len(scaler_dict)} 个键")
        logger.info(f"✓ 最终验证指标: RMSE={history['val_rmse'][-1]:.2f}K, R²={history['val_r2'][-1]:.4f}")
        
        # 9. 绘制训练历史和散点图
        plot_training_history(
            history, 
            os.path.join(output_dir, f"local_finetune_{file_prefix}_history1.png")
        )
        plot_hexbin_scatter(val_targets_denorm, 
                            val_preds_denorm, 
                            os.path.join(output_dir, f'local_finetune_{file_prefix}_hexbin.png'),
                            title=f'Local Fine-tuning {target_date.strftime("%Y-%m-%d")}',
                            xlabel='Original LST (K)', 
                            ylabel='Predicted LST (K)'
        )
        # 如果需要更多点，建议使用 plot_model_scatter 时传入小样本或改成避免再次推理。当前用采样数据生成密度图，避免OOM。
        plot_model_scatter(local_model, val_loader, pretrain_scaler_dict, 
                           file_prefix,
                           os.path.join(output_dir, f'local_finetune_{file_prefix}_scatter.png'),                           
                           xlabel='Original LST (K)', 
                           ylabel='Predicted LST (K)',
                           dpi=300
        )
        
        # 内存清理
        del local_model, train_loader, val_loader, train_dataset, val_dataset, local_data
        gc.collect()
        torch.cuda.empty_cache()

        return final_model_path


if __name__ == "__main__":
    logger.info("开始局部微调（使用统一标准化）")
    
    # 请根据您的实际路径修改以下配置
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    global_pretrain_dir = r"G:\CNN_SpatialDownscaling\output\global_pretrain4"
    local_finetune_dir = r"G:\CNN_SpatialDownscaling\output\local_finetune_ValidationData"
    
    try:
        # 示例日期
        #test_date = datetime(2018, 1, 15)
        
        for day in [6]:#range(1, 32):  # 1 到 31 日

            test_date = datetime(2018, 10, day)

            finetuner = LocalFineTuner(config_path, global_pretrain_dir, local_finetune_dir)
            logger.info(f"开始对 {test_date.strftime('%Y-%m-%d')} 进行局部微调")
            
            # 调试模式
            finetune_model_path = finetuner.fine_tune(test_date, is_debug=False)
            
            if finetune_model_path:
                logger.info(f"✓ 调试完成，模型路径: {finetune_model_path}")
            else:
                logger.error("✗ 微调失败")
        
    except Exception as e:
        logger.error(f"局部微调过程出错: {e}", exc_info=True)