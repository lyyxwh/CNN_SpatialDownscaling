"""
模型加载与验证数据Hexbin图绘制脚本
用于加载训练好的模型,在验证集上进行推断,并绘制hexbin散点图
"""
import os
import json
import torch
import numpy as np
from datetime import datetime
import logging
from torch.utils.data import DataLoader, Subset

from data_loader import DataLoader as LSTDataLoader
from preprocess import LSTDataset, DataCollator
from model import LSTTransformer
from visualization_utils import plot_hexbin_scatter, plot_density_scatter

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器 - 加载模型并在验证集上绘制hexbin图"""
    
    def __init__(self, config_path, model_path, output_dir):
        """
        初始化评估器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型文件路径(.pt)
            output_dir: 输出目录
        """
        self.config_path = config_path
        self.model_path = model_path
        self.output_dir = output_dir
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化数据加载器
        self.data_loader = LSTDataLoader(config_path)
    
    def load_model(self):
        """加载训练好的模型"""
        logger.info(f"正在加载模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # 获取模型配置和scaler
        model_config = checkpoint['config']
        self.scaler_dict = checkpoint['scaler_dict']
        
        # 创建模型并加载权重
        model = LSTTransformer(model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"✓ 模型加载成功")
        logger.info(f"  模型类型: {model_config.get('n_layers', 'N/A')}层Transformer")
        logger.info(f"  d_model: {model_config.get('d_model', 'N/A')}")
        
        return model
    
    def prepare_validation_data(self, target_date, is_global=True, val_ratio=0.1):
        """
        准备验证数据
        
        Args:
            target_date: 目标日期
            is_global: 是否为全局数据
            val_ratio: 验证集比例
            
        Returns:
            val_loader: 验证集数据加载器
        """
        logger.info(f"准备验证数据: {target_date.strftime('%Y-%m-%d')}, 全局={is_global}")
        
        # 加载数据
        data = self.data_loader.load_data(target_date, is_global=is_global, is_pretrain=is_global)
        
        if data is None:
            raise ValueError(f"无法加载数据: {target_date.strftime('%Y-%m-%d')}")
        
        # 创建数据集(使用已有的scaler,不进行fit)
        dataset = LSTDataset(
            data['features'], 
            data['target'],
            scaler_dict=self.scaler_dict,
            is_train=False,
            is_global=is_global,
            fit_features=False,  # 不fit特征
            fit_target=False     # 不fit目标
        )
        
        # 划分验证集(使用与训练时相同的随机种子)
        total_size = len(dataset)
        val_size = int(val_ratio * total_size)
        train_size = total_size - val_size
        
        # 使用固定种子确保可复现
        torch.manual_seed(42)
        indices = torch.randperm(total_size)
        val_indices = indices[train_size:train_size + val_size]
        
        val_dataset = Subset(dataset, val_indices)
        
        logger.info(f"验证集大小: {len(val_dataset)}")
        
        # 创建数据加载器
        base_dataset = dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        
        embedding_layers = base_dataset.get_embedding_layers()
        collator = DataCollator(embedding_layers)
        
        batch_size = self.config['model_params'].get('batch_size', 512)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collator
        )
        
        return val_loader
    
    def evaluate_and_plot(self, model, val_loader, month, title_prefix="Model Validation"):
        """
        在验证集上评估模型并绘制hexbin图
        
        Args:
            model: 训练好的模型
            val_loader: 验证集数据加载器
            title_prefix: 图表标题前缀
            
        Returns:
            metrics: 评估指标字典
        """
        logger.info("开始模型评估...")
        
        all_predictions = []
        all_targets = []
        
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # 准备输入数据
                batch_data = {
                    'num': batch['num_features'].to(self.device),
                    'time': batch['time_features'].to(self.device),
                    'loc': batch['loc_features'].to(self.device),
                    'cat': {k: v.to(self.device) for k, v in batch['cat'].items()}
                }
                
                # 前向传播
                outputs = model(batch_data).squeeze(-1)
                targets = batch['target'].to(self.device)
                
                # 收集预测和目标(标准化状态)
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 合并所有批次
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        logger.info(f"收集了 {len(predictions)} 个样本")
        
        # 反标准化
        if 'target' in self.scaler_dict:
            target_scaler = self.scaler_dict['target']
            predictions_denorm = target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
            targets_denorm = target_scaler.inverse_transform(
                targets.reshape(-1, 1)
            ).flatten()
            
            logger.info(f"反标准化完成")
            logger.info(f"  预测值范围: [{predictions_denorm.min():.2f}, {predictions_denorm.max():.2f}] K")
            logger.info(f"  真实值范围: [{targets_denorm.min():.2f}, {targets_denorm.max():.2f}] K")
        else:
            predictions_denorm = predictions
            targets_denorm = targets
            logger.warning("未找到target scaler,使用原始值")
        
        # 绘制hexbin图
        model_name = os.path.basename(self.model_path).replace('.pt', '')
        output_path = os.path.join(
            self.output_dir, 
            f"{model_name}_validation_hexbin.png"
        )
        date_str = model_name.split('_')[-1]
        metrics={}
        metrics['rmse'] = np.sqrt(np.mean((predictions_denorm - targets_denorm) ** 2))
        metrics['mae'] = np.mean(np.abs(predictions_denorm - targets_denorm))
        metrics['r2'] = 1 - np.sum((targets_denorm - predictions_denorm) ** 2) / np.sum((targets_denorm - np.mean(targets_denorm)) ** 2)
        metrics['bias'] = np.mean(predictions_denorm - targets_denorm)

        if month == 1:
            xlim=(230,310)
            ylim=(230,310)
        else:
            xlim=(270,340)
            ylim=(270,340)

        plot_hexbin_scatter(
            y_true=targets_denorm,
            y_pred=predictions_denorm,
            output_path=output_path,
            title=f'{title_prefix} - {date_str}',
            xlabel='Original LST (K)',
            ylabel='Predicted LST (K)',
            figsize=(8, 8),
            gridsize=200
        )

        plot_density_scatter(
            y_true=targets_denorm, 
            y_pred=predictions_denorm,
            output_path=output_path.replace('.png', '_density.png'),
            title=f'{title_prefix} - {date_str}',
            xlabel='Original LST (K)',
            ylabel='Predicted LST (K)',
            xlim=xlim,
            ylim=ylim
        )
        
        logger.info(f"✓ Hexbin图已保存: {output_path}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f} K")
        logger.info(f"  MAE: {metrics['mae']:.4f} K")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  Bias: {metrics['bias']:.4f} K")
        
        return metrics


def main():
    """主函数"""
    # ===== 配置参数 =====
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    
    # 全局预训练模型路径示例
    model_dir = r"G:\CNN_SpatialDownscaling\output"
    
    #target_date = datetime(2018, 1, 15)  # 与模型训练数据对应
    for is_global in [True, False]:
 
    #is_global = False  # true为全局模型
        for month in [1, 7]:
            for day in [1, 15]:
                target_date = datetime(2018, month, day)
                print(target_date)
                if is_global:
                    model_path = os.path.join(model_dir,"global_pretrain4", f"global_pretrain_{target_date.strftime('%Y%m')}.pt")
                else:
                    model_path = os.path.join(model_dir,"local_finetune_121", f"local_finetune_{target_date.strftime('%Y%m%d')}.pt")
                # 或者使用局部微调模型
                # model_path = r"G:\CNN_SpatialDownscaling\output\local_finetune_121\local_finetune_20181026.pt"
                # target_date = datetime(2018, 10, 26)
                # is_global = False  # 局部模型
                
                output_dir = r"G:\CNN_SpatialDownscaling\output\model_evaluation"
                
                # ===== 执行评估 =====
                try:
                    # 创建评估器
                    evaluator = ModelEvaluator(config_path, model_path, output_dir)
                    
                    # 加载模型
                    model = evaluator.load_model()
                    val_ratio = 0.1 if is_global else 0.2    # 全局模型使用10%的验证集，局部模型使用20%的验证集
                    # 准备验证数据
                    val_loader = evaluator.prepare_validation_data(
                        target_date=target_date,
                        is_global=is_global,
                        val_ratio=val_ratio
                    )
                    
                    # 评估并绘图
                    title = "Global Pretrain" if is_global else "Local Finetune"
                    metrics = evaluator.evaluate_and_plot(
                        model=model,
                        val_loader=val_loader,
                        title_prefix=title,
                        month=month
                    )
                    
                    logger.info("\n" + "="*50)
                    logger.info("评估完成!")
                    logger.info(f"模型: {os.path.basename(model_path)}")
                    logger.info(f"RMSE: {metrics['rmse']:.4f} K")
                    logger.info(f"R²: {metrics['r2']:.4f}")
                    logger.info("="*50)
                    model = None  # 释放模型内存
                    torch.cuda.empty_cache()  # 清理GPU缓存
        
                except Exception as e:
                    logger.error(f"评估失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()