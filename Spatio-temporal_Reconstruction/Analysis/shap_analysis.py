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
logging.getLogger('shap').setLevel(logging.WARNING)


# --- 学术风格全局配置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 统一学术配色
SOURCE_COLORS = {
            'ERA5': '#FF6B6B',
            'SMAP': '#4ECDC4',
            'NDVI': '#95E1D3',
            'DEM': '#F38181',
            'Time': '#AA96DA',
            'Location': '#FCBAD3',
            'LandCover': '#FFFFD2',
            'Other': '#E8E8E8'
}



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
      
        self.feature_display_names = {
            't2m': '2m Temperature',# '2米温度',
            'ssrd': 'Surface Solar Radiation',# '短波辐射(太阳短波辐射向下通量)',
            'strd': 'Surface Thermal Radiation',#'地表向下热辐射',
            'rh': 'Relative Humidity',# '相对湿度',
            'd2m': 'Dewpoint Temperature', # '露点温度',
            'vpd': 'Vapor Pressure Deficit', # '水汽压差',
            'sm_surface_wetness': 'Soil Moisture', # '表层土壤湿度',
            '_1_km_16_days_NDVI': 'NDVI', # '归一化植被指数',
            'dem': 'Elevation', # '高程',
            'slope': 'Slope', # '坡度',
            'aspect': 'Aspect', # '坡向',
            'clcd': 'Land Cover', # '土地覆盖',
            'day': 'Day', # '天',
            'hour': 'Hour',
            'doy': 'Day of Year',
            'lat': 'Latitude',
            'lon': 'Longitude'
        }
         # 定义特征分组(按数据源)，用于可视化
        self.source_groups = {
            'ERA5': ['t2m', 'ssrd', 'strd', 'rh', 'd2m', 'vpd'],
            'SMAP': ['sm_surface_wetness'],
            'NDVI': ['_1_km_16_days_NDVI'],
            'DEM': ['dem', 'slope', 'aspect'],
            'Time': ['day', 'hour', 'doy'],
            'Location': ['lat', 'lon'],
            'LandCover': ['clcd']
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
        # 【修改】strd -> strd
        num_features = ['t2m', 'ssrd', 'strd', 'rh', 'd2m', 'vpd',
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

        total_len = len(dataset)
        # 直接采样最终需要的数量，而不是采样10%
        actual_sample_count = min(sample_size, total_len)
        
        # 随机生成固定数量的索引
        indices = np.random.choice(total_len, actual_sample_count, replace=False)
        dataset = Subset(dataset, indices)
        
        logger.info(f"数据总量: {total_len}, 随机抽取解释样本: {actual_sample_count} 个")

        # 创建数据加载器
        base_dataset = dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset

        # 注意：shuffle 设为 True 确保样本分布均匀
        embedding_layers = base_dataset.get_embedding_layers()
        collator = DataCollator(embedding_layers)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=collator)
        
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
        
        for batch_idx, batch in enumerate(dataloader):
            features = self.extract_features_from_batch(batch)
            targets = batch['target'].cpu().numpy()
            
            all_features.append(features)
            all_targets.append(targets)
            
            if len(all_features) * features.shape[0] >= max_samples:
                break
            
            # 调整日志记录频率
            if batch_idx % 100 == 0:  # 每处理 100 个批次记录一次日志
                logger.info(f"已处理 {batch_idx} 个批次，共 {len(dataloader)} 个批次")

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
        logger.info("开始计算 SHAP 值 (串行循环 10,000 次，每步 Batch=100)")
        shap_values = explainer.shap_values(
            feature_data,
            nsamples=500,#500,  # 减少采样次数以加快速度
            silent=False
        )
        
        logger.info(f"SHAP值计算完成,形状: {shap_values.shape}")
        
        return shap_values, feature_data, target_data

    def _get_clean_name(self, feature_name):
            """去除编码后缀并映射到显示名称"""
            base = feature_name.replace('_sin', '').replace('_cos', '')
            return self.feature_display_names.get(base, base)
    
    def _get_source(self, base_name):
            """辅助方法：根据基础特征名获取所属数据源"""
            for src, feats in self.source_groups.items():
                if base_name in feats:
                    return src
            return 'Other'
    
    def analyze_feature_importance(self, shap_values, feature_names):
        """
        分析特征重要性
        
        Args:
            shap_values: SHAP值数组
            feature_names: 特征名称列表
            
        Returns:
            importance_df: 特征重要性DataFrame
        """
        # 1. 提取不含后缀的基础特征列表
        base_features = []
        seen = set()
        for f in feature_names:
            base = f.replace('_sin', '').replace('_cos', '').replace('_lat', '').replace('_lon', '')
            # 处理 lat_sin/lat_cos 等特殊后缀
            base = base.replace('lat', 'Latitude').replace('lon', 'Longitude') if 'lat' in f or 'lon' in f else base
            if base not in seen:
                base_features.append(base)
                seen.add(base)
        
        # 特别修正：地理位置通常合并为 Latitude 和 Longitude
        # 脚本中原特征名为 lon_sin, lon_cos 等，此处需确保匹配逻辑        
        importance_results = []
        for base in base_features:
            # 查找所有属于该基础特征的原始列索引
            # 兼容处理：如 base 为 'Latitude'，匹配 'lat_sin', 'lat_cos'
            search_key = 'lat' if base == 'Latitude' else ('lon' if base == 'Longitude' else base)
            cols = [i for i, f in enumerate(feature_names) if search_key in f]
            
            if not cols: continue
                
            # 核心修正点：先求和合并物理意义，再计算平均绝对值
            combined_shap_per_sample = shap_values[:, cols].sum(axis=1)
            mean_abs_importance = np.abs(combined_shap_per_sample).mean()
            
            importance_results.append({
                'Base_Feature': base,
                'Importance': mean_abs_importance,
                'Display_Name': self._get_clean_name(search_key),
                'Source': self._get_source(search_key)
            })
        
        importance_df = pd.DataFrame(importance_results)
        return importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
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
        # 修正 top_n 以防特征总数不足
        top_n = min(top_n, len(importance_df))
        top_df = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [SOURCE_COLORS.get(s, '#E8E8E8') for s in top_df['Source']]
        
        bars = ax.barh(top_df['Display_Name'], top_df['Importance'], 
                      color=colors,  alpha=0.8)
        
        ax.invert_yaxis()
        ax.set_xlabel('Mean Absolute SHAP Value (Impact on LST)', fontsize=14)
        ax.set_title(f'Feature Importance Ranking (Top {top_n})', fontsize=16, fontweight='bold', family='serif')
        
        # 移除多余边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=SOURCE_COLORS[s], label=s, ) 
                          for s in top_df['Source'].unique()]
        ax.legend(handles=legend_elements, loc='lower right', frameon=False)

        plt.savefig(output_path, dpi=600)
        plt.close()
    
    def plot_feature_importance_bar(self, importance_df, output_path, top_n=20):
        """
        绘制特征重要性条形图

        Args:
            importance_df: 特征重要性DataFrame(已聚合)
            output_path: 输出路径
            top_n: 显示前N个特征
        """
        # 修正 top_n 以防特征总数不足
        top_n = min(top_n, len(importance_df))
        top_df = importance_df.head(top_n)

        # 设置画布大小
        fig, ax = plt.subplots(figsize=(10, 8))

        # 提取颜色
        colors = [SOURCE_COLORS.get(source, '#E8E8E8') for source in top_df['Source']]

        # 绘制条形图
        bars = ax.barh(top_df['Display_Name'], top_df['Importance'], color='lightpink', alpha=0.8)

        # 反转Y轴，使得重要性最高的特征在最上方
        ax.invert_yaxis()

        # 设置标签和标题
        ax.set_xlabel('Mean Absolute SHAP Value', fontsize=14)
        #ax.set_title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')

        # 添加数值标签到每个条形图上
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.3f}',
                    ha='left', va='center', fontsize=10, color='black')

        # 移除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 添加网格线
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()

        logger.info(f"特征重要性条形图已保存: {output_path}")
    
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
        
        ax1.set_title('数据源贡献度分布', fontsize=16, fontweight='bold')
        
        # 柱状图
        bars = ax2.bar(
            range(len(source_df)),
            source_df['Contribution'],
            color=colors[:len(source_df)]
        )
        
        ax2.set_xticks(range(len(source_df)))
        ax2.set_xticklabels(source_df['Source'], rotation=45, ha='right')
        ax2.set_ylabel('贡献度(平均绝对SHAP值)', fontsize=14)
        ax2.set_title('数据源贡献度对比', fontsize=16, fontweight='bold')
        
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
    
    def plot_shap_summary(self, shap_values, feature_data, feature_names, output_path, top_n=17):
        """
        绘制SHAP摘要图(优化版 - 聚合sin/cos特征)
        
        Args:
            shap_values: SHAP值数组
            feature_data: 特征数据数组
            feature_names: 特征名称列表
            output_path: 输出路径
        """
        unique_bases = []
        seen = set()
        for f in feature_names:
            base = f.replace('_sin', '').replace('_cos', '')
            if base not in seen: 
                unique_bases.append(base)
                seen.add(base)

        agg_shaps = []
        agg_datas = []
        for base in unique_bases:
            cols = [i for i, f in enumerate(feature_names) if f.startswith(base)]
            # X轴：SHAP值直接求和 (代数和反映总拉力)
            agg_shaps.append(shap_values[:, cols].sum(axis=1))
            # 颜色：特征原值取均值作为代表性数值
            agg_datas.append(feature_data[:, cols].mean(axis=1))
            
        agg_shaps = np.array(agg_shaps).T
        agg_datas = np.array(agg_datas).T
        
        # 排序逻辑：按平均绝对值
        mean_shaps = np.abs(agg_shaps).mean(axis=0)
        actual_top_n = min(top_n, len(unique_bases)) 
        top_idx = np.argsort(mean_shaps)[-actual_top_n:]
        
        # 3. 绘图 (简化散点逻辑，确保与 Beeswarm 风格一致)
        # 准备绘图数据
        plot_shaps = agg_shaps[:, top_idx]
        plot_datas = agg_datas[:, top_idx]
        plot_features = [unique_bases[i] for i in top_idx]
        display_names = [self._get_clean_name(f) for f in plot_features]

        # 4. 创建 SHAP Explanation 对象
        expl = shap.Explanation(
            values=plot_shaps,
            data=plot_datas,
            feature_names=display_names
        )

        # 5. 绘制蜂群图
        plt.figure(figsize=(10, 8))
        plt.set_cmap('coolwarm')  # 替换为您想要的颜色映射
        shap.plots.beeswarm(expl, max_display=actual_top_n, show=False, color_bar=True)
        
        # 绘制左侧边框
        ax = plt.gca()
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(1)  # 确保边框可见
        # 左侧边框对应绘制刻度线
        ax.tick_params(axis='y', which='both', direction='out', length=4)

        plt.axvline(0, color='black', lw=0.8, alpha=0.7)
        plt.yticks(range(actual_top_n), [self._get_clean_name(unique_bases[i]) for i in top_idx], fontsize=12)
        plt.xlabel('SHAP value (impact on model output)', fontsize=14)
        #plt.title('Feature Contribution Summary (Aggregated)', fontsize=16, fontweight='bold')
        
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()

    def plot_overlay_summary(self, shap_values, feature_data, feature_names, output_path, top_n=17):
        """
        实现参考图效果：背景条形图(重要性)与前景散点图(分布)叠加
        """
        # 1. 特征聚合 (核心修正：代数和)
        unique_bases = []
        for f in feature_names:
            base = f.replace('_sin', '').replace('_cos', '')
            if base not in unique_bases: unique_bases.append(base)
            
        agg_shaps_list = []
        agg_datas_list = []
        for base in unique_bases:
            cols = [i for i, f in enumerate(feature_names) if f.startswith(base)]
            agg_shaps_list.append(shap_values[:, cols].sum(axis=1)) # 物理合并
            agg_datas_list.append(feature_data[:, cols].mean(axis=1))
            
        agg_shaps = np.array(agg_shaps_list).T
        agg_datas = np.array(agg_datas_list).T
        
        # 2. 计算重要性 (修正：|A+B| 的均值)
        mean_abs_shaps = np.abs(agg_shaps).mean(axis=0)
        actual_top_n = min(top_n, len(unique_bases))
        total_imp = mean_abs_shaps.sum()
        top_idx = np.argsort(mean_abs_shaps)[-min(top_n, len(unique_bases)):]

        # 2. 创建画布
        fig, ax_main = plt.subplots(figsize=(10, 8))
        # 创建双 X 轴：ax_bar 用于顶部的均值重要性条形图
        ax_bar = ax_main.twiny() 

        y_pos = np.arange(actual_top_n)

        # 设置统一的背景颜色
        UNIFORM_BAR_COLOR = "#DEADAD"  # 标准学术浅灰色
        LABEL_COLOR = "#232323"        # 深灰色文字，确保易读性
        
        # 3. 绘制背景条形图 (对应顶部轴)
        # 使用对应数据源的颜色，但降低透明度(alpha)作为背景
        for i, idx in enumerate(top_idx):
            base_name = unique_bases[idx]
            source = self._get_source(base_name)
            color = SOURCE_COLORS.get(source, '#E8E8E8')
            
            ax_bar.barh(i, mean_abs_shaps[idx], color='lightpink', alpha=0.4, 
                    height=0.7, edgecolor='none', zorder=1)
            # 在条形图左侧添加数值标注 (类似参考图: 0.137 (28.8%))
            percentage = (mean_abs_shaps[idx] / mean_abs_shaps.sum()) * 100
            label_text = f"{mean_abs_shaps[idx]:.3f} ({percentage:.1f}%)"
            # 优化：使用轴坐标变换 (transform)，x=0.01 表示紧贴左侧 Y 轴
            # ha='left' 确保文字向右延伸
            ax_main.text(0.01, i, label_text, va='center', ha='left', 
                        fontsize=10, family='Times New Roman', fontweight='bold',
                        color='gray', # 文字颜色同步，增强一致性
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'), #文字增加背景值，避免遮挡
                        transform=ax_main.get_yaxis_transform(), zorder=5)

        # 4. 绘制前景散点图 (对应底部轴)
        for i, idx in enumerate(top_idx):
            shaps = agg_shaps[:, idx]
            vals = agg_datas[:, idx]
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min()) if vals.max() != vals.min() else vals
            
            jitter = np.random.normal(0, 0.12, size=len(shaps))

            sc = ax_main.scatter(shaps, i + jitter, c=norm_vals, cmap='coolwarm', 
                                s=15, alpha=0.6, edgecolors='none', rasterized=True, zorder=3)
            
        # 5. 坐标轴美化
        ax_main.axvline(0, color='gray', lw=1, alpha=0.5, zorder=2)
        
        # 设置底部轴 (SHAP Value分布)
        ax_main.set_xlabel('SHAP value (impact on model output)', fontsize=12, labelpad=10)
        ax_main.set_yticks(y_pos)
        ax_main.set_yticklabels([self._get_clean_name(unique_bases[i]) for i in top_idx], fontsize=11)
        
        # 设置顶部轴 (Mean |SHAP| 重要性)
        ax_bar.set_xlabel('Mean (|SHAP| value)', fontsize=12, labelpad=10)
        ax_bar.spines['top'].set_visible(True)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['left'].set_visible(False)

        # 隐藏主图不需要的边框
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)

        # 6. 添加颜色条
        cb = plt.colorbar(sc, ax=ax_main, shrink=0.6, aspect=25, pad=0.08)
        cb.set_label('Feature value', fontsize=10)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['Low', 'High'])
        cb.outline.set_visible(False)

        plt.title('Combined Feature Importance & Distribution Plot', fontsize=14, fontweight='bold', pad=40)
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()   

    def plot_beeswarm_with_bar(self, shap_values, feature_data, feature_names, output_path, top_n=17):
        """
        修复版：严格匹配参考图样式的复合 SHAP 摘要图
        功能：背景粉色条形图(Mean |SHAP|) + 前景蜂群图(Distribution) + 数值标注
        """
        # 1. 健壮性检查：处理 shap_values 为列表的情况 (常见于 KernelExplainer)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # 2. 定义统一的特征聚合逻辑 (匹配 sin/cos 和 lat/lon)
        def get_base_name(name):
            base = name.replace('_sin', '').replace('_cos', '')
            # 统一 lat/lon 的基础名，确保与 analyze_feature_importance 一致
            if base == 'lat': return 'lat'
            if base == 'lon': return 'lon'
            return base

        # 提取聚合后的基础特征列表
        base_features = []
        seen = set()
        for f in feature_names:
            base = get_base_name(f)
            if base not in seen:
                base_features.append(base)
                seen.add(base)

        agg_shaps_list = []
        agg_datas_list = []
        for base in base_features:
            # 精确匹配：匹配基础名或基础名开头的编码特征
            cols = [i for i, f in enumerate(feature_names) if f == base or f.startswith(base + '_')]
            # SHAP值局部可加性：代数求和
            agg_shaps_list.append(shap_values[:, cols].sum(axis=1))
            # 特征原值：取均值 (对于sin/cos，均值仅用于区分高低，物理意义受限但符合SHAP着色逻辑)
            agg_datas_list.append(feature_data[:, cols].mean(axis=1))

        agg_shaps = np.array(agg_shaps_list).T  # (samples, n_base_features)
        agg_datas = np.array(agg_datas_list).T
        
        # 3. 计算重要性并排序
        mean_abs_shaps = np.abs(agg_shaps).mean(axis=0)
        total_importance = mean_abs_shaps.sum()
        
        # 获取 Top N (按重要性从小到大排序，因为 y_pos 从下往上)
        actual_top_n = min(top_n, len(base_features))
        top_idx = np.argsort(mean_abs_shaps)[-actual_top_n:] 

        # 准备绘图数据
        plot_shaps = agg_shaps[:, top_idx]
        plot_datas = agg_datas[:, top_idx]
        plot_features = [base_features[i] for i in top_idx]
        plot_importance = mean_abs_shaps[top_idx]
        plot_percentages = (plot_importance / total_importance) * 100
        display_names = [self._get_clean_name(f) for f in plot_features]

        # 4. 创建画布和双坐标系
        fig = plt.figure(figsize=(12, 0.5 * actual_top_n + 3), dpi=300)
        ax = fig.add_subplot(111)  # 主坐标系：用于散点 (底部X轴)
        ax_bar = ax.twiny()        # 孪生坐标系：用于条形图 (顶部X轴)

        ax_bar.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_alpha(0)       

        y_pos = np.arange(actual_top_n)

        # 5. 绘制背景条形图 (顶部轴)
        xlim_bar = plot_importance.max() * 1.15
        ax_bar.barh(y=y_pos, width=plot_importance, height=0.6, 
                    color='lightpink', alpha=0.4, edgecolor='none', zorder=0)
        
        ax_bar.set_xlim(0, xlim_bar)
        ax_bar.set_xlabel('Mean (|SHAP| value)', fontsize=12, labelpad=12)
        
        '''# 添加文本标签 (例如: 2.443 (22.6%))
        for i, (v, p) in enumerate(zip(plot_importance, plot_percentages)):
            ax_bar.text(x=xlim_bar * 0.01, y=i, 
                        s=f"{v:.3f} ({p:.1f}%)",
                        va='center', ha='left', fontsize=9, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))'''

        # 6. 绘制蜂群图 (核心修复：设置 max_display)
        expl = shap.Explanation(
            values=plot_shaps, 
            data=plot_datas,
            feature_names=display_names
        )
        
        plt.sca(ax)
        # 必须指定 max_display=actual_top_n，否则默认只显示前10个
        shap.plots.beeswarm(expl, max_display=actual_top_n, show=False, color_bar=True)

        # 6. 样式精调与文字防压盖处理
        # 自动调整 X 轴范围，为左侧文字留出空间
        x_min, x_max = ax.get_xlim()
        # 稍微向左拉伸范围，确保文字不会紧贴 Y 轴
        ax.set_xlim(x_min - abs(x_min)*0.05, x_max) 
        
        # 【关键修复】：在所有绘图完成后绘制文字，并设置极高的 zorder 和不透明背景
        for i, (v, p) in enumerate(zip(plot_importance, plot_percentages)):
            label_text = f"{v:.3f} ({p:.1f}%)"
            # 使用 ax 坐标系放置文字。x=x_min 表示放在绘图区最左侧
            ax.text(x=x_min, y=i, s=label_text, 
                    va='center', ha='left', 
                    fontsize=10, fontweight='bold', family='Times New Roman',
                    color='gray',
                    zorder=100,  # 确保在散点之上
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))

        # 设置坐标轴标签
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=12, family='serif')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=11, family='serif')
        # 左侧边框对应绘制刻度线
        ax.tick_params(axis='y', which='both', direction='out', length=4)
        '''# 移除多余边框
        for a in [ax, ax_bar]:
            a.spines['right'].set_visible(False)
            if a == ax: a.spines['top'].set_visible(False)'''
        
        # 移除边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        #ax_bar.spines['left'].set_visible(False)

        # 7. 调整 Colorbar 字体
        for child in fig.get_children():
            if isinstance(child, plt.Axes) and child not in [ax, ax_bar]:
                child.set_ylabel('Feature value', fontsize=11, family='serif')
                child.tick_params(labelsize=10)

        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ 已生成 SHAP 图: {output_path}")

    def run_analysis(self, target_date, is_global=True, max_samples=10000000):
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
        
        self.plot_feature_importance_bar(
            importance_df,
            os.path.join(self.output_dir, f'{output_prefix}_feature_importance_bar.png')
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

        self.plot_overlay_summary(
            shap_values,
            feature_data,
            feature_names,
            os.path.join(self.output_dir, f'{output_prefix}_shap_overlay_summary.png')
        )
        
        self.plot_beeswarm_with_bar(
            shap_values,
            feature_data,
            feature_names,
            os.path.join(self.output_dir, f'{output_prefix}_shap_enhanced_beeswarm.png')
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
    global_model_dir = r"G:\CNN_SpatialDownscaling\output\global_pretrain4"
    global_output_dir = r"G:\CNN_SpatialDownscaling\output\shap_analysis\global_1"
    
    # 局部模型分析
    local_model_dir = r"G:\CNN_SpatialDownscaling\output\local_finetune_121"
    local_output_dir = r"G:\CNN_SpatialDownscaling\output\shap_analysis\local_1" \
    ""
    
    try:
        # 分析全局模型(2018年1月)
        logger.info("="*80)
        logger.info("开始分析全局预训练模型")
        logger.info("="*80)

        for month in [1, 7]:
            logger.info(f"\n{'-'*30}\n分析全局模型 - 2018年{month}月\n{'-'*30}")


            target_date = datetime(2018, month, 15)

            global_analyzer = SHAPAnalyzer(config_path, global_model_dir, global_output_dir)
            global_results = global_analyzer.run_analysis(
                target_date=target_date,
                is_global=True,
                max_samples=100000
            )
            
            # 分析局部模型
            logger.info("\n" + "="*80)
            logger.info("开始分析局部微调模型")
            logger.info("="*80)
            
            local_analyzer = SHAPAnalyzer(config_path, local_model_dir, local_output_dir)
            local_results = local_analyzer.run_analysis(
                target_date=target_date,
                is_global=False,
                max_samples=100000
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



