"""
SHAP特征贡献分析模块
用于分析多源数据对LST预测的贡献度
支持全局和局部模型的特征重要性分析
"""
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from tqdm import tqdm
import shap
from torch.utils.data import DataLoader, Subset
import warnings
import sys
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 将 package 目录直接加入搜索路径，兼容模块内使用的顶级导入（如 from data_loader import ...）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modis_lst_interpolation_code_framework')))

from modis_lst_interpolation_code_framework.data_loader import DataLoader as LSTDataLoader
from modis_lst_interpolation_code_framework.preprocess import LSTDataset, DataCollator
from modis_lst_interpolation_code_framework.model import LSTTransformer

# 配置日志
log_dir = os.path.join(os.path.dirname(__file__), '../output/shap_analysis')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'shap_analysis.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

class SHAPAnalyzer:
    """SHAP特征贡献分析器"""
    
    def __init__(self, config_path, model_dir, output_dir):
        """
        初始化SHAP分析器
        
        Args:
            config_path: 配置文件路径
            model_dir: 模型目录(全局或局部)
            output_dir: 分析结果输出目录
        """
        self.config_path = config_path
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = LSTDataLoader(config_path)
        
        # 定义特征分组(按数据源)
        self.feature_groups = {
            # 【修改】strd -> net_radiation_flux
            'ERA5': ['t2m', 'ssrd', 'net_radiation_flux', 'rh', 'd2m', 'vpd'],
            'SMAP': ['sm_surface_wetness'],
            'NDVI': ['_1_km_16_days_NDVI'],
            'DEM': ['dem', 'slope', 'aspect'],
            # 【修改】移除 month
            'Time': ['day', 'hour', 'doy'],
            'Location': ['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos'],
            'LandCover': ['clcd']
        }
        
        # 特征中文名映射
        self.feature_names_zh = {
            't2m':  '2m temperature', # '2米温度',
            'ssrd':  'Surface solar radiation downwards',#'短波辐射(太阳短波辐射向下通量)',
            # 【修改】strd -> net_radiation_flux
            'net_radiation_flux': 'Surface Net Radiation', #'地表净辐射',
            'rh': 'Relative humidity',# '相对湿度',
            'd2m': '2m dewpoint temperature',#'2米露点温度',
            'vpd': 'Vapor pressure deficit',# '水汽压差',
            'sm_surface_wetness': 'Surface soil moisture',#'表层土壤湿度',
            '_1_km_16_days_NDVI': 'NDVI',
            'dem': 'Elevation',#'高程',
            'slope': 'Slope',#'坡度',
            'aspect': 'Aspect',#'坡向',
            'clcd': 'Land cover',# '土地覆盖',
            # 【修改】移除 month
            'day': 'Day',
            'hour': 'Hour',
            'doy': 'Day of Year',
            'lat_sin': 'Latitude (sin)',
            'lat_cos': 'Latitude (cos)',
            'lon_sin': 'Longitude (sin)',
            'lon_cos': 'Longitude (cos)'
        }
        
        logger.info(f"SHAP分析器初始化完成,使用设备: {self.device}")
    
    def load_model(self, target_date, is_global=True):
        """
        加载模型
        
        Args:
            target_date: 目标日期
            is_global: 是否为全局模型
            
        Returns:
            model, scaler_dict, feature_names
        """
        if is_global:
            date_str = target_date.strftime('%Y%m')
            model_path = os.path.join(self.model_dir, f'global_pretrain_{date_str}.pt')
        else:
            date_str = target_date.strftime('%Y%m%d')
            model_path = os.path.join(self.model_dir, f'local_finetune_{date_str}.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        logger.info(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model = LSTTransformer(checkpoint['config']).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler_dict = checkpoint['scaler_dict']
        
        # 获取特征名称列表
        feature_names = self._get_feature_names(checkpoint['config'])
        
        logger.info(f"模型加载成功,特征数量: {len(feature_names)}")
        
        return model, scaler_dict, feature_names
    
    def _get_feature_names(self, model_config):
        """从模型配置中提取特征名称"""
        feature_names = []
        
        # 数值特征(按照训练时的顺序)
        # 【修改】strd -> net_radiation_flux
        num_features = ['t2m', 'ssrd', 'net_radiation_flux', 'rh', 'd2m', 'vpd',
                       '_1_km_16_days_NDVI', 'dem', 'slope', 'aspect']
        feature_names.extend(num_features)
        
        # 时间特征(正余弦编码后)
        # 【修改】移除 month_sin, month_cos
        time_features = ['day_sin', 'day_cos',
                        'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos']
        feature_names.extend(time_features)
        
        # 地理特征
        loc_features = ['lon_sin', 'lon_cos', 'lat_sin', 'lat_cos']
        feature_names.extend(loc_features)
        
        # 分类特征(嵌入后的维度)
        if 'cat_features' in model_config:
            for cat_name in ['sm_surface_wetness', 'clcd']:
                if cat_name in model_config['cat_features']:
                    # 简化处理,只用特征名表示
                    feature_names.append(cat_name)
        
        return feature_names
    
    def prepare_data(self, target_date, is_global=True, sample_size=5000):
        """
        准备分析数据
        
        Args:
            target_date: 目标日期
            is_global: 是否为全局数据
            sample_size: 采样大小
            
        Returns:
            dataset, dataloader
        """
        logger.info(f"准备{'全局' if is_global else '局部'}数据")
        
        # 加载数据
        data = self.data_loader.load_data(
            target_date, 
            is_global=is_global, 
            is_pretrain=is_global
        )
        
        if data is None:
            raise ValueError("数据加载失败")
        
        # 创建数据集
        dataset = LSTDataset(
            data['features'], 
            data['target'],
            is_train=False,
            is_global=is_global
        )
        
        # 随机采样
        if len(dataset) > sample_size:
            indices = np.random.choice(len(dataset), int(len(dataset) * 0.1), replace=False) # 采样10%
            dataset = Subset(dataset, indices)
            logger.info(f"从{len(data['features']['lat'])}个样本中采样{len(indices)}个")
        
        # 创建数据加载器
        base_dataset = dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        
        embedding_layers = base_dataset.get_embedding_layers()
        collator = DataCollator(embedding_layers)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collator)
        
        return dataset, dataloader
    
    def create_model_wrapper(self, model, scaler_dict):
        """
        创建模型包装器,用于SHAP分析
        
        Args:
            model: PyTorch模型
            scaler_dict: 标准化器字典
            
        Returns:
            wrapped_model: 包装后的模型函数
        """
        def wrapped_model(batch_data_array):
            """
            SHAP兼容的模型包装函数
            
            Args:
                batch_data_array: numpy数组,形状为(batch_size, n_features)
                
            Returns:
                predictions: numpy数组,形状为(batch_size,)
            """
            # 将numpy数组转换为模型输入格式
            batch_size = batch_data_array.shape[0]
            
            # 假设特征顺序:数值特征 + 时间特征 + 地理特征 + 分类特征
            num_features = 10  # ERA5(6) + NDVI(1) + DEM(3)
            # 【修改】Time features now 6 (day, hour, doy * 2), removed month
            time_features = 6  
            loc_features = 4   # 经纬度的sin/cos编码
            
            with torch.no_grad():
                # 分离不同类型的特征
                num_data = torch.tensor(
                    batch_data_array[:, :num_features], 
                    dtype=torch.float32
                ).to(self.device)
                
                time_data = torch.tensor(
                    batch_data_array[:, num_features:num_features+time_features],
                    dtype=torch.float32
                ).to(self.device)
                
                loc_data = torch.tensor(
                    batch_data_array[:, num_features+time_features:num_features+time_features+loc_features],
                    dtype=torch.float32
                ).to(self.device)
                
                # 分类特征(简化处理,使用众数)
                cat_data = {}
                if batch_data_array.shape[1] > num_features + time_features + loc_features:
                    cat_features_array = batch_data_array[:, num_features+time_features+loc_features:]
                    
                    # SMAP
                    if cat_features_array.shape[1] > 0:
                        cat_data['sm_surface_wetness'] = torch.tensor(
                            cat_features_array[:, 0].astype(np.int64),
                            dtype=torch.long
                        ).to(self.device)
                    
                    # CLCD
                    if cat_features_array.shape[1] > 1:
                        cat_data['clcd'] = torch.tensor(
                            cat_features_array[:, 1].astype(np.int64),
                            dtype=torch.long
                        ).to(self.device)
                
                # 构建模型输入
                batch_data = {
                    'num': num_data,
                    'time': time_data,
                    'loc': loc_data,
                    'cat': cat_data
                }
                
                # 模型推理
                outputs = model(batch_data).squeeze(-1)
                
                # 反标准化
                if 'target' in scaler_dict:
                    predictions = scaler_dict['target'].inverse_transform(
                        outputs.cpu().numpy().reshape(-1, 1)
                    ).flatten()
                else:
                    predictions = outputs.cpu().numpy()
            
            return predictions
        
        return wrapped_model
    
    def extract_features_from_batch(self, batch):
        """
        从批次数据中提取特征矩阵
        
        Args:
            batch: DataLoader返回的批次数据
            
        Returns:
            features: numpy数组,形状为(batch_size, n_features)
        """
        num_features = batch['num_features'].cpu().numpy()
        time_features = batch['time_features'].cpu().numpy()
        loc_features = batch['loc_features'].cpu().numpy()
        
        # 拼接特征
        features = np.concatenate([num_features, time_features, loc_features], axis=1)
        
        # 添加分类特征
        if 'cat' in batch:
            cat_features_list = []
            for cat_name in ['sm_surface_wetness', 'clcd']:
                if cat_name in batch['cat']:
                    cat_features_list.append(
                        batch['cat'][cat_name].cpu().numpy().reshape(-1, 1)
                    )
            
            if cat_features_list:
                cat_features = np.concatenate(cat_features_list, axis=1)
                features = np.concatenate([features, cat_features], axis=1)
        
        return features
    
    def compute_shap_values(self, model, scaler_dict, dataloader, max_samples=1000):
        """
        计算SHAP值
        
        Args:
            model: 模型
            scaler_dict: 标准化器
            dataloader: 数据加载器
            max_samples: 最大样本数
            
        Returns:
            shap_values: SHAP值数组
            feature_data: 特征数据数组
        """
        logger.info("开始计算SHAP值...")
        
        # 提取特征数据
        all_features = []
        all_targets = []
        
        for batch in dataloader:
            features = self.extract_features_from_batch(batch)
            targets = batch['target'].cpu().numpy()
            
            all_features.append(features)
            all_targets.append(targets)
            
            if len(all_features) * features.shape[0] >= max_samples:
                break
        
        feature_data = np.vstack(all_features)[:max_samples]
        target_data = np.concatenate(all_targets)[:max_samples]
        
        logger.info(f"特征数据形状: {feature_data.shape}")
        
        # 创建模型包装器
        wrapped_model = self.create_model_wrapper(model, scaler_dict)
        
        # 使用KernelExplainer(适用于任何模型)
        logger.info("初始化SHAP KernelExplainer...")
        
        # 选择背景数据(使用k-means采样)
        background_size = min(100, len(feature_data))
        background = shap.kmeans(feature_data, background_size)
        
        explainer = shap.KernelExplainer(wrapped_model, background)
        
        # 计算SHAP值
        logger.info("计算SHAP值(这可能需要一些时间)...")
        shap_values = explainer.shap_values(
            feature_data,
            nsamples=100,  # 减少采样次数以加快速度
            silent=False
        )
        
        logger.info(f"SHAP值计算完成,形状: {shap_values.shape}")
        
        return shap_values, feature_data, target_data
    
    def analyze_feature_importance(self, shap_values, feature_names):
        """
        分析特征重要性
        
        Args:
            shap_values: SHAP值数组
            feature_names: 特征名称列表
            
        Returns:
            importance_df: 特征重要性DataFrame
        """
        # 计算每个特征的平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_abs_shap
        })
        
        # 添加中文名称(使用替换sin/cos后的名称)
        importance_df['Feature_ZH'] = importance_df['Feature'].map(
            lambda x: self.feature_names_zh.get(x, x)
        )
        
        # 添加数据源分组
        def get_source(feature):
            for source, features in self.feature_groups.items():
                if any(f in feature for f in features):
                    return source
            return 'Other'
        
        importance_df['Source'] = importance_df['Feature'].apply(get_source)
        
        # ⭐ 聚合sin/cos编码的同一特征
        # 将sin和cos的重要性加和,只保留一个条目
        aggregated_rows = []
        processed_features = set()
        
        for idx, row in importance_df.iterrows():
            feature = row['Feature']
            
            # 如果已经处理过,跳过
            if feature in processed_features:
                continue
            
            # 检查是否是sin/cos编码的特征
            if feature.endswith('_sin'):
                base_feature = feature.replace('_sin', '')
                cos_feature = base_feature + '_cos'
                
                # 查找对应的cos特征
                cos_row = importance_df[importance_df['Feature'] == cos_feature]
                
                if not cos_row.empty:
                    # 合并sin和cos的重要性
                    combined_importance = row['Importance'] + cos_row.iloc[0]['Importance']
                    
                    aggregated_rows.append({
                        'Feature': base_feature,
                        'Importance': combined_importance,
                        'Feature_ZH': row['Feature_ZH'],  # 使用已映射的名称
                        'Source': row['Source']
                    })
                    
                    # 标记这两个特征已处理
                    processed_features.add(feature)
                    processed_features.add(cos_feature)
                else:
                    # 没有对应的cos,保留原样
                    aggregated_rows.append(row.to_dict())
                    processed_features.add(feature)
            
            elif feature.endswith('_cos'):
                # 如果是cos但sin还没处理,说明没有对应的sin
                if feature.replace('_cos', '_sin') not in importance_df['Feature'].values:
                    aggregated_rows.append(row.to_dict())
                    processed_features.add(feature)
            
            else:
                # 不是sin/cos编码的特征,直接保留
                aggregated_rows.append(row.to_dict())
                processed_features.add(feature)
        
        # 创建新的DataFrame
        importance_df_agg = pd.DataFrame(aggregated_rows)
        importance_df_agg = importance_df_agg.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        logger.info(f"特征聚合完成: 原始{len(importance_df)}个特征 -> 聚合后{len(importance_df_agg)}个特征")
        
        return importance_df_agg
    
    def analyze_source_contribution(self, importance_df):
        """
        分析数据源贡献度
        
        Args:
            importance_df: 特征重要性DataFrame
            
        Returns:
            source_df: 数据源贡献DataFrame
        """
        # 按数据源聚合
        source_contribution = importance_df.groupby('Source')['Importance'].sum()
        source_contribution = source_contribution.sort_values(ascending=False)
        
        source_df = pd.DataFrame({
            'Source': source_contribution.index,
            'Contribution': source_contribution.values,
            'Percentage': source_contribution.values / source_contribution.values.sum() * 100
        })
        
        logger.info("\n数据源贡献度:")
        logger.info(source_df.to_string())
        
        return source_df
    
    def plot_feature_importance(self, importance_df, output_path, top_n=20):
        """
        绘制特征重要性图(优化版 - 已聚合sin/cos特征)
        
        Args:
            importance_df: 特征重要性DataFrame(已聚合)
            output_path: 输出路径
            top_n: 显示前N个特征
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 选择Top N特征
        top_features = importance_df.head(top_n)
        
        # 数据源颜色映射
        source_colors = {
            'ERA5': '#FF6B6B',
            'SMAP': '#4ECDC4',
            'NDVI': '#95E1D3',
            'DEM': '#F38181',
            'Time': '#AA96DA',
            'Location': '#FCBAD3',
            'LandCover': '#FFFFD2',
            'Other': '#E8E8E8'
        }
        
        colors = [source_colors.get(source, '#E8E8E8') for source in top_features['Source']]
        
        # 绘制水平柱状图
        bars = ax.barh(
            range(len(top_features)), 
            top_features['Importance'], 
            color=colors,
            edgecolor='white',
            linewidth=0.5
        )
        
        # 设置y轴
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature_ZH'], fontsize=11)
        ax.invert_yaxis()
        
        # 设置x轴
        ax.set_xlabel('Mean absolute SHAP value', fontsize=12, fontweight='bold')
        ax.set_title(f'Feature Importance Ranking (Top {top_n})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 添加网格
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, top_features['Importance'])):
            ax.text(
                val, i, f' {val:.4f}',
                va='center', ha='left', fontsize=9, color='#333333'
            )
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=source, edgecolor='white')
            for source, color in source_colors.items()
            if source in top_features['Source'].values
        ]
        
        ax.legend(
            handles=legend_elements, 
            loc='lower right',
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10
        )
        
        # 调整布局
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"特征重要性图已保存: {output_path}")
        logger.info(f"Top 5特征: {', '.join(top_features['Feature_ZH'].head(5).tolist())}")
    
    def plot_source_contribution(self, source_df, output_path):
        """
        绘制数据源贡献饼图
        
        Args:
            source_df: 数据源贡献DataFrame
            output_path: 输出路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 饼图
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', 
                 '#AA96DA', '#FCBAD3', '#FFFFD2']
        
        wedges, texts, autotexts = ax1.pie(
            source_df['Contribution'],
            labels=source_df['Source'],
            autopct='%1.1f%%',
            colors=colors[:len(source_df)],
            startangle=90
        )
        
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax1.set_title('数据源贡献度分布', fontsize=14, fontweight='bold')
        
        # 柱状图
        bars = ax2.bar(
            range(len(source_df)),
            source_df['Contribution'],
            color=colors[:len(source_df)]
        )
        
        ax2.set_xticks(range(len(source_df)))
        ax2.set_xticklabels(source_df['Source'], rotation=45, ha='right')
        ax2.set_ylabel('贡献度(平均绝对SHAP值)', fontsize=12)
        ax2.set_title('数据源贡献度对比', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{source_df.iloc[i]["Percentage"]:.1f}%',
                ha='center',
                va='bottom'
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"数据源贡献图已保存: {output_path}")
    
    def plot_shap_summary(self, shap_values, feature_data, feature_names, output_path):
        """
        绘制SHAP摘要图(优化版 - 聚合sin/cos特征)
        
        Args:
            shap_values: SHAP值数组
            feature_data: 特征数据数组
            feature_names: 特征名称列表
            output_path: 输出路径
        """
        # ⭐ 步骤1: 聚合sin/cos编码的特征
        aggregated_shap = []
        aggregated_data = []
        aggregated_names = []
        processed_indices = set()
        
        for idx, feature in enumerate(feature_names):
            if idx in processed_indices:
                continue
            
            if feature.endswith('_sin'):
                base_feature = feature.replace('_sin', '')
                cos_feature = base_feature + '_cos'
                
                # 查找对应的cos特征索引
                try:
                    cos_idx = feature_names.index(cos_feature)
                    
                    # 合并sin和cos的SHAP值(取绝对值后相加,保留符号)
                    sin_shap = shap_values[:, idx]
                    cos_shap = shap_values[:, cos_idx]
                    
                    # 计算合并后的SHAP值(使用L2范数)
                    combined_shap = np.sqrt(sin_shap**2 + cos_shap**2) * np.sign(sin_shap)
                    
                    # 合并特征值(使用平均值)
                    combined_data = (feature_data[:, idx] + feature_data[:, cos_idx]) / 2
                    
                    aggregated_shap.append(combined_shap)
                    aggregated_data.append(combined_data)
                    aggregated_names.append(self.feature_names_zh.get(feature, base_feature))
                    
                    processed_indices.add(idx)
                    processed_indices.add(cos_idx)
                    
                except ValueError:
                    # 没有对应的cos特征
                    aggregated_shap.append(shap_values[:, idx])
                    aggregated_data.append(feature_data[:, idx])
                    aggregated_names.append(self.feature_names_zh.get(feature, feature))
                    processed_indices.add(idx)
            
            elif feature.endswith('_cos'):
                # 检查是否已经在sin处理时被处理
                if idx not in processed_indices:
                    aggregated_shap.append(shap_values[:, idx])
                    aggregated_data.append(feature_data[:, idx])
                    aggregated_names.append(self.feature_names_zh.get(feature, feature))
                    processed_indices.add(idx)
            
            else:
                # 非sin/cos特征,直接添加
                aggregated_shap.append(shap_values[:, idx])
                aggregated_data.append(feature_data[:, idx])
                aggregated_names.append(self.feature_names_zh.get(feature, feature))
                processed_indices.add(idx)
        
        # 转换为numpy数组
        aggregated_shap = np.column_stack(aggregated_shap)
        aggregated_data = np.column_stack(aggregated_data)
        
        logger.info(f"特征聚合: {len(feature_names)} -> {len(aggregated_names)}")
        
        # ⭐ 步骤2: 按重要性排序并选择Top 20
        mean_abs_shap = np.abs(aggregated_shap).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-20:][::-1]
        
        final_shap = aggregated_shap[:, top_indices]
        final_data = aggregated_data[:, top_indices]
        final_names = [aggregated_names[i] for i in top_indices]
        
        # ⭐ 步骤3: 绘制优化的SHAP摘要图
        plt.figure(figsize=(12, 10))
        
        # 使用自定义绘图而非shap.summary_plot以获得更好的控制
        for i in range(len(final_names)):
            y_pos = len(final_names) - 1 - i
            
            # 按特征值排序以获得渐变效果
            sorted_idx = np.argsort(final_data[:, i])
            x_vals = final_shap[sorted_idx, i]
            colors = final_data[sorted_idx, i]
            
            # 绘制散点
            plt.scatter(
                x_vals, 
                [y_pos] * len(x_vals),
                c=colors,
                cmap='RdYlBu_r',
                alpha=0.6,
                s=15,
                edgecolors='none',
                rasterized=True
            )
        
        # 设置y轴标签
        plt.yticks(range(len(final_names)), final_names, fontsize=11)
        plt.xlabel('SHAP value (impact on model output)', fontsize=12)
        plt.title('Feature Importance Summary', fontsize=14, fontweight='bold', pad=20)
        
        # 添加零线
        plt.axvline(x=0, color='#999999', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r')
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Feature value', fontsize=11)
        cbar.ax.tick_params(labelsize=10)
        
        # 调整布局
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP摘要图已保存: {output_path}")
        logger.info(f"显示特征: {', '.join(final_names[:5])}...")
    
    def run_analysis(self, target_date, is_global=True, max_samples=1000):
        """
        运行完整的SHAP分析
        
        Args:
            target_date: 目标日期
            is_global: 是否为全局模型
            max_samples: 最大样本数
            
        Returns:
            analysis_results: 分析结果字典
        """
        date_str = target_date.strftime('%Y%m' if is_global else '%Y%m%d')
        model_type = 'global' if is_global else 'local'
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始SHAP分析: {date_str} ({model_type})")
        logger.info(f"{'='*60}\n")
        
        # 1. 加载模型
        model, scaler_dict, feature_names = self.load_model(target_date, is_global)
        
        # 2. 准备数据
        dataset, dataloader = self.prepare_data(target_date, is_global, max_samples)
        
        # 3. 计算SHAP值
        shap_values, feature_data, target_data = self.compute_shap_values(
            model, scaler_dict, dataloader, max_samples
        )
        
        # 4. 分析特征重要性
        importance_df = self.analyze_feature_importance(shap_values, feature_names)
        
        # 5. 分析数据源贡献
        source_df = self.analyze_source_contribution(importance_df)
        
        # 6. 保存结果
        output_prefix = f"{model_type}_{date_str}"
        
        # 保存CSV
        importance_df.to_csv(
            os.path.join(self.output_dir, f'{output_prefix}_feature_importance.csv'),
            index=False,
            encoding='utf-8-sig'
        )
        
        source_df.to_csv(
            os.path.join(self.output_dir, f'{output_prefix}_source_contribution.csv'),
            index=False,
            encoding='utf-8-sig'
        )
        
        # 保存SHAP值
        np.save(
            os.path.join(self.output_dir, f'{output_prefix}_shap_values.npy'),
            shap_values
        )
        
        # 7. 可视化
        self.plot_feature_importance(
            importance_df,
            os.path.join(self.output_dir, f'{output_prefix}_feature_importance.png')
        )
        
        self.plot_source_contribution(
            source_df,
            os.path.join(self.output_dir, f'{output_prefix}_source_contribution.png')
        )
        
        self.plot_shap_summary(
            shap_values,
            feature_data,
            feature_names,
            os.path.join(self.output_dir, f'{output_prefix}_shap_summary.png')
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SHAP分析完成: {date_str}")
        logger.info(f"结果已保存到: {self.output_dir}")
        logger.info(f"{'='*60}\n")
        
        return {
            'importance_df': importance_df,
            'source_df': source_df,
            'shap_values': shap_values,
            'feature_data': feature_data
        }


def main():
    """主函数"""
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    
    # 全局模型分析
    global_model_dir = r"G:\CNN_SpatialDownscaling\output\global_pretrain"
    global_output_dir = r"G:\CNN_SpatialDownscaling\output\shap_analysis\global"
    
    # 局部模型分析
    local_model_dir = r"G:\CNN_SpatialDownscaling\output\local_finetune"
    local_output_dir = r"G:\CNN_SpatialDownscaling\output\shap_analysis\local"
    
    try:
        # 分析全局模型(2018年1月)
        logger.info("="*80)
        logger.info("开始分析全局预训练模型")
        logger.info("="*80)
        
        global_analyzer = SHAPAnalyzer(config_path, global_model_dir, global_output_dir)
        global_results = global_analyzer.run_analysis(
            target_date=datetime(2018, 1, 15),
            is_global=True,
            max_samples=1000
        )
        
        # 分析局部模型(2018年1月1日)
        logger.info("\n" + "="*80)
        logger.info("开始分析局部微调模型")
        logger.info("="*80)
        
        local_analyzer = SHAPAnalyzer(config_path, local_model_dir, local_output_dir)
        local_results = local_analyzer.run_analysis(
            target_date=datetime(2018, 1, 1),
            is_global=False,
            max_samples=1000
        )
        
        # 对比分析
        logger.info("\n" + "="*80)
        logger.info("全局模型 vs 局部模型 - 数据源贡献对比")
        logger.info("="*80)
        
        logger.info("\n全局模型:")
        logger.info(global_results['source_df'].to_string())
        
        logger.info("\n局部模型:")
        logger.info(local_results['source_df'].to_string())
        
    except Exception as e:
        logger.error(f"分析过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



