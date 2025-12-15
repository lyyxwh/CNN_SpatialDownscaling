import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
from datetime import datetime
from data_loader import DataLoader as LSTDataLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================
# 新增: 统一LST标准化器
# ============================================
class UnifiedLSTScaler:
    """
    统一的LST标准化器
    将所有LST数据（CLDAS和MODIS）标准化到固定范围[-1, 1]
    
    优势:
    - 全局预训练和局部微调使用相同的标准化基准
    - 避免不同数据源的分布偏移
    - 模型学到的是真实的温度变化规律
    """
    def __init__(self, temp_min=220.0, temp_max=340.0, scale_range=(-1, 1)):
        """
        Args:
            temp_min: LST最小值(K)，默认220K (约-53°C)
            temp_max: LST最大值(K)，默认340K (约67°C)
            scale_range: 标准化后的范围，默认[-1, 1]
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.scale_min, self.scale_max = scale_range
        
        logger.info(f"初始化统一LST标准化器: 原始范围[{temp_min}, {temp_max}]K -> 标准化范围{scale_range}")
    
    def fit_transform(self, data):
        """标准化（训练时调用，实际上不需要fit）"""
        return self.transform(data)
    
    def transform(self, data):
        """
        Min-Max标准化到指定范围
        公式: scaled = (x - min) / (max - min) * (scale_max - scale_min) + scale_min
        """
        data = np.asarray(data).astype(np.float32)
        
        # 裁剪到合理范围（防止异常值）
        data_clipped = np.clip(data, self.temp_min, self.temp_max)
        
        # Min-Max标准化
        normalized = (data_clipped - self.temp_min) / (self.temp_max - self.temp_min)
        scaled = normalized * (self.scale_max - self.scale_min) + self.scale_min
        
        return scaled.reshape(-1, 1) if data.ndim == 1 else scaled
    
    def inverse_transform(self, scaled_data):
        """
        反标准化
        公式: x = (scaled - scale_min) / (scale_max - scale_min) * (max - min) + min
        """
        scaled_data = np.asarray(scaled_data).astype(np.float32)
        
        # 反Min-Max标准化
        normalized = (scaled_data - self.scale_min) / (self.scale_max - self.scale_min)
        original = normalized * (self.temp_max - self.temp_min) + self.temp_min
        
        return original.reshape(-1, 1) if scaled_data.ndim == 1 else original
    
    @property
    def mean_(self):
        """兼容StandardScaler接口（用于日志输出）"""
        return np.array([(self.temp_max + self.temp_min) / 2])
    
    @property
    def scale_(self):
        """兼容StandardScaler接口（用于日志输出）"""
        return np.array([(self.temp_max - self.temp_min) / 2])


# ============================================
# 新增: 特征标准化管理器
# ============================================
class FeatureScalerManager:
    """
    特征标准化管理器
    为不同类型的特征提供合适的标准化方法
    """
    def __init__(self):
        self.scalers = {}
        # 不需要标准化的特征
        self.no_scale_features = [
            '_1_km_16_days_NDVI',  # NDVI已经在[-1,1]
            'lon', 'lat',  # 原始经纬度（不作为模型输入）
            'lon_sin', 'lon_cos', 'lat_sin', 'lat_cos',  # 三角函数已归一化
        ]
        # 地理特征列表
        self.geo_features = ['dem', 'slope', 'aspect', 'dem_1km', 'slope_1km', 'aspect_1km']
    
    def fit_transform(self, features_dict, is_train=True):
        """
        标准化所有数值特征
        
        Args:
            features_dict: 特征字典
            is_train: 是否为训练模式（fit新scaler）
        
        Returns:
            standardized_features: 标准化后的特征字典
        """
        standardized_features = {}
        
        for key, values in features_dict.items():
            if key in self.no_scale_features:
                standardized_features[key] = values
                logger.debug(f"特征 {key} 跳过标准化")
                continue
            
            if is_train:
                # 训练模式：fit新scaler
                scaler = StandardScaler()
                scaler.fit(values.reshape(-1, 1))
                self.scalers[key] = scaler
                standardized_features[key] = scaler.transform(values.reshape(-1, 1)).flatten()
                logger.info(f"特征 {key}: fit并标准化, mean={scaler.mean_[0]:.3f}, std={scaler.scale_[0]:.3f}")
            else:
                # 推理模式：使用已有scaler
                if key in self.scalers:
                    scaler = self.scalers[key]
                    standardized_features[key] = scaler.transform(values.reshape(-1, 1)).flatten()
                    logger.info(f"特征 {key}: 使用已有scaler标准化")
                else:
                    logger.warning(f"特征 {key} 无scaler，使用原始值")
                    standardized_features[key] = values
        
        return standardized_features


class EmbeddingLayer(nn.Module):
    """嵌入层,用于处理分类变量"""
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        return self.embedding(x)


class LSTDataset(Dataset):
    def __init__(self, features, target, scaler_dict=None, embedding_dims=None, is_train=True, is_global=True,
                 fit_features=True, fit_target=True):
        """
        初始化LST数据集（使用统一标准化）
        
        Args:
            features: 特征字典
            target: 目标变量字典
            scaler_dict: 标准化器字典(如果为None则创建新的,如果提供则使用现有的)
            embedding_dims: 嵌入维度字典
            is_train: 是否为训练模式
            is_global: 是否为全局数据
            fit_features: 是否fit特征scaler（训练时True，微调时False）
            fit_target: 是否fit目标scaler（训练时True，微调时False）
        """
        self.fit_features = fit_features
        self.fit_target = fit_target
        self.is_global = is_global
        self.features = features
        self.target = target
        self.is_train = is_train
        self.embedding_dims = embedding_dims or {
            'clcd': {'embed_dim': 8},
            'sm_surface_wetness': {'embed_dim': 5}
        }
        
        # 初始化scaler_dict
        if scaler_dict is None:
            self.scaler_dict = {}
            logger.info("创建新的scaler_dict")
        else:
            self.scaler_dict = scaler_dict.copy()
            logger.info(f"使用传入的scaler_dict，包含 {len(scaler_dict)} 个键")
        
        # ============================================
        # 1. 合并为 DataFrame，便于行过滤
        # ============================================
        target_var = list(target.keys())[0]
        df = pd.DataFrame(features)
        df[target_var] = target[target_var]

        # 去除含 NaN 的行
        before_rows = len(df)
        df = df.dropna()
        after_rows = len(df)
        logger.info(f"已删除含 NaN 的样本行: {before_rows - after_rows} 行，剩余 {after_rows} 行")

        # 数据修剪(仅在全局训练时)
        if self.is_train and self.is_global and after_rows > 0:
            target_values = df[target_var].values
            
            lower_bound = np.nanpercentile(target_values, 0.1)
            upper_bound = np.nanpercentile(target_values, 99.9)
            
            trim_mask = (target_values >= lower_bound) & (target_values <= upper_bound)
            df = df[trim_mask]
            
            trimmed_rows = after_rows - len(df)
            after_trim_rows = len(df)
            
            logger.info(f"【训练数据修剪】: 删除了最小/最大 0.1% 的数据，共 {trimmed_rows} 行。新样本数: {after_trim_rows} 行。")
            logger.info(f"修剪边界: {target_var} 范围 [{lower_bound:.2f}, {upper_bound:.2f}] (原 {after_rows} 行)")
            after_rows = after_trim_rows

        # 拆分回 features / target
        self.features = {k: df[k].values for k in features.keys()}
        self.target = {target_var: df[target_var].values}

        # ============================================
        # 2. 分类 / 时间 / 数值 / 地理特征区分
        # ============================================
        self.num_features = {}
        self.cat_features = {}
        self.time_features = {}
        self.loc_features = {}

        ignored_vars = ["lon", "lat"]
        categorical_vars = ["clcd", "sm_surface_wetness"]
        
        for k, v in self.features.items():
            if k in ignored_vars:
                logger.debug(f"特征 {k} 为原始经纬度，已忽略，不作为模型输入特征。")
                continue
            if k in categorical_vars or k in self.embedding_dims:
                self.cat_features[k] = v.astype(np.int64)
                if len(v) > 0:
                    logger.info(f"分类变量 {k}: 唯一值数量={len(np.unique(v))}, 取值范围=[{np.min(v)}, {np.max(v)}]")
                else:
                    logger.info(f"分类变量 {k}: 数据为空")
            elif k in ["month", "day", "hour", "doy"]:
                self.time_features[k] = v.astype(np.int64)
            elif k in ["lon_sin", "lon_cos", "lat_sin", "lat_cos"]:
                self.loc_features[k] = v.astype(np.float32)
            else:
                self.num_features[k] = v.astype(np.float32)

        # ============================================
        # 3. 数值特征标准化（使用FeatureScalerManager）
        # ============================================
        if 'feature_manager' not in self.scaler_dict:
            self.feature_manager = FeatureScalerManager()
            if self.is_train and self.fit_features:
                # 训练模式且需要fit：创建新的feature_manager
                self.num_features = self.feature_manager.fit_transform(self.num_features, is_train=True)
                self.scaler_dict['feature_manager'] = self.feature_manager
                logger.info("已创建并保存FeatureScalerManager")
            else:
                # 不fit特征，但还是要用空的manager（后续会从scaler_dict获取）
                logger.info("训练模式但不fit特征，将使用传入的scaler")
        else:
            # 使用已有的feature_manager
            self.feature_manager = self.scaler_dict['feature_manager']
            self.num_features = self.feature_manager.fit_transform(self.num_features, is_train=False)
            logger.info("使用传入的FeatureScalerManager进行特征标准化")

        # ============================================
        # 4. 时间特征正余弦编码
        # ============================================
        time_encoded = []
        for k, v in self.time_features.items():
            if k == "month": T = 12
            elif k == "day": T = 31
            elif k == "hour": T = 24
            elif k == "doy": T = 365
            else: continue
            time_encoded.append(np.sin(2 * np.pi * v / T))
            time_encoded.append(np.cos(2 * np.pi * v / T))
        self.X_time = torch.tensor(np.stack(time_encoded, axis=1), dtype=torch.float32) if time_encoded else torch.zeros((len(self.target[target_var]), 0))

        # ============================================
        # 5. 地理特征 (直接拼接)
        # ============================================
        if self.loc_features:
            loc_arr = np.stack(list(self.loc_features.values()), axis=1)
            self.X_loc = torch.tensor(loc_arr, dtype=torch.float32)
        else:
            self.X_loc = torch.zeros((len(self.target[target_var]), 0))

        # ============================================
        # 6. 数值特征拼接
        # ============================================
        if self.num_features:
            num_arr = np.stack(list(self.num_features.values()), axis=1)
            self.X_num = torch.tensor(num_arr, dtype=torch.float32)
        else:
            self.X_num = torch.zeros((len(self.target[target_var]), 0))

        # ============================================
        # 7. 分类变量处理(用于嵌入编码)
        # ============================================
        self.X_cat = {}
        self.embedding_info = {}
        
        for k, v in self.cat_features.items():
            unique_vals = np.unique(v)
            num_classes = len(unique_vals)
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            mapped_v = np.array([val_to_idx[val] for val in v])
            
            self.X_cat[k] = torch.tensor(mapped_v, dtype=torch.long)
            
            # 获取或使用默认嵌入维度
            if k in self.embedding_dims:
                embed_dim = self.embedding_dims[k].get('embed_dim', min(50, (num_classes + 1) // 2))
            else:
                embed_dim = min(50, (num_classes + 1) // 2)
            
            self.embedding_info[k] = {
                'num_classes': num_classes,
                'embed_dim': embed_dim,
                'val_to_idx': val_to_idx,
                'unique_vals': unique_vals
            }
            
            logger.info(f"分类变量 {k}: 类别数={num_classes}, 嵌入维度={embed_dim}, 原始取值={unique_vals}")

        # 在训练模式下保存embedding信息到scaler_dict
        if self.is_train:
            self.scaler_dict['cat_features'] = self.embedding_info
            logger.info(f"已保存 {len(self.embedding_info)} 个分类特征的embedding信息到scaler_dict")

        # ============================================
        # 8. 目标变量处理（关键修改：使用统一标准化器）
        # ============================================
        self.y_raw = self.target[target_var].astype(np.float32)
        
        # 创建或使用统一的LST标准化器
        if self.is_train and self.fit_target:
            # 训练模式且需要fit：创建新的UnifiedLSTScaler
            self.unified_lst_scaler = UnifiedLSTScaler(220, 340, (-1, 1))
            self.scaler_dict['target'] = self.unified_lst_scaler
            logger.info(f"目标 {target_var} (统一标准化): 范围[220, 340]K -> [-1, 1]")
        else:
            # 推理模式或不fit目标：使用传入的scaler
            if 'target' in self.scaler_dict:
                target_scaler = self.scaler_dict['target']
                # 检查是否为UnifiedLSTScaler
                if not isinstance(target_scaler, UnifiedLSTScaler):
                    logger.warning("⚠️ 检测到旧的StandardScaler，替换为UnifiedLSTScaler")
                    self.unified_lst_scaler = UnifiedLSTScaler(220, 340, (-1, 1))
                    self.scaler_dict['target'] = self.unified_lst_scaler
                else:
                    self.unified_lst_scaler = target_scaler
                    logger.info(f"使用传入的UnifiedLSTScaler: [{target_scaler.temp_min}, {target_scaler.temp_max}]K")
            else:
                logger.warning("scaler_dict中无target scaler，创建新的UnifiedLSTScaler")
                self.unified_lst_scaler = UnifiedLSTScaler(220, 340, (-1, 1))
                self.scaler_dict['target'] = self.unified_lst_scaler
        
        # 标准化目标变量
        self.y = self.unified_lst_scaler.transform(self.y_raw).flatten()
        self.y = torch.tensor(self.y, dtype=torch.float32)
        
        # 输出统计信息
        logger.info(f"目标变量统计: 原始范围[{self.y_raw.min():.2f}, {self.y_raw.max():.2f}]K, "
                   f"标准化后范围[{self.y.min():.3f}, {self.y.max():.3f}]")
        
        # 最终汇总
        logger.info(f"scaler_dict 最终包含的键: {list(self.scaler_dict.keys())}")
        logger.info(f"LSTDataset 初始化完成: 数值特征 {self.X_num.shape[1]}, "
                   f"时间特征 {self.X_time.shape[1]}, 地理特征 {self.X_loc.shape[1]}, "
                   f"分类特征 {list(self.X_cat.keys())}, 样本数 {len(self.y)}")

    def save_processed_data(self, file_path):
        """保存处理后的数据集到文件,包括特征和标准化器。"""
        data_to_save = {
            'features': self.features,
            'target': self.target,
            'scaler_dict': self.scaler_dict,
            'embedding_dims': self.embedding_dims,
            'embedding_info': self.embedding_info
        }
        torch.save(data_to_save, file_path)
        logger.info(f"处理后的数据已保存到: {file_path}")

    @classmethod
    def load_processed_data(cls, file_path):
        """从文件加载处理后的数据集。"""
        if not os.path.exists(file_path):
            return None, None
        
        logger.info(f"从文件加载处理后的数据: {file_path}")
        data_loaded = torch.load(file_path, weights_only=False)
        
        features = data_loaded['features']
        target = data_loaded['target']
        scaler_dict = data_loaded['scaler_dict']
        embedding_dims = data_loaded.get('embedding_dims', {})
        
        return cls(features, target, scaler_dict=scaler_dict, embedding_dims=embedding_dims, is_train=False), scaler_dict

    def get_embedding_layers(self):
        """
        返回嵌入层字典,用于模型构建
        
        Returns:
            dict: 包含各分类变量嵌入层的字典
        """
        embedding_layers = {}
        for k, info in self.embedding_info.items():
            embedding_layers[k] = EmbeddingLayer(
                num_embeddings=info['num_classes'],
                embedding_dim=info['embed_dim']
            )
        return embedding_layers

    def get_feature_dims(self):
        """
        返回各类特征的维度信息
        
        Returns:
            dict: 包含各类特征维度的字典
        """
        return {
            'num_features': self.X_num.shape[1],
            'time_features': self.X_time.shape[1],
            'loc_features': self.X_loc.shape[1],
            'embedding_info': self.embedding_info
        }

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "num_features": self.X_num[idx],
            "time_features": self.X_time[idx],
            "loc_features": self.X_loc[idx],
            "cat": {k: v[idx] for k, v in self.X_cat.items()},
            "target": self.y[idx]
        }


class DataCollator:
    """数据整理器,用于批处理时合并特征"""
    
    def __init__(self, embedding_layers):
        self.embedding_layers = embedding_layers
    
    def __call__(self, batch):
        """
        将批次数据整理为模型输入格式
        
        Args:
            batch: 批次数据列表
        
        Returns:
            dict: 整理后的批次数据
        """
        num_features = torch.stack([item['num_features'] for item in batch])
        time_features = torch.stack([item['time_features'] for item in batch])
        loc_features = torch.stack([item['loc_features'] for item in batch])
        cat_features = {k: torch.stack([item['cat'][k] for item in batch]) for k in self.embedding_layers}
        targets = torch.stack([item['target'] for item in batch])
        
        return {
            'num_features': num_features,
            'time_features': time_features,
            'loc_features': loc_features,
            'cat': cat_features,
            'target': targets
        }


# ===========================
# 独立调试入口
# ===========================
if __name__ == "__main__":
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    loader = LSTDataLoader(config_path)
    target_date = datetime(2018, 10, 15)
    
    logger.info("\n--- 调试全局预训练数据加载 ---")
    data = loader.load_data(target_date, is_global=True, is_pretrain=True)
    if data is not None:
        logger.info(f"全局预训练数据加载成功。特征数量: {len(data['features'])},样本总数: {len(data['features']['lat'])}")
    
    if data is not None:
        embedding_dims = {
            'clcd': {'embed_dim': 4},
            'sm_surface_wetness': {'embed_dim': 3}
        }
        
        # 创建训练数据集
        dataset = LSTDataset(data["features"], data["target"], 
                           embedding_dims=embedding_dims, is_train=True, is_global=True)
        
        # 验证scaler_dict完整性
        logger.info(f"\n训练数据集的scaler_dict包含的键: {list(dataset.scaler_dict.keys())}")
        
        # 验证UnifiedLSTScaler
        target_scaler = dataset.scaler_dict['target']
        logger.info(f"Target Scaler类型: {type(target_scaler).__name__}")
        if isinstance(target_scaler, UnifiedLSTScaler):
            logger.info(f"✅ 使用统一标准化: [{target_scaler.temp_min}, {target_scaler.temp_max}]K -> [{target_scaler.scale_min}, {target_scaler.scale_max}]")
        
        # 获取嵌入层
        embedding_layers = dataset.get_embedding_layers()
        
        # 获取特征维度信息
        feature_dims = dataset.get_feature_dims()
        logger.info(f"特征维度信息: {feature_dims}")
        
        # 创建数据整理器和数据加载器
        collator = DataCollator(embedding_layers)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=collator)

        # 测试批次处理
        for batch in dataloader:
            logger.info(f"批次数据: 数值特征 {batch['num_features'].shape}, "
                       f"时间特征 {batch['time_features'].shape}, "
                       f"地理特征 {batch['loc_features'].shape}, "
                       f"分类特征 {dict((k, v.shape) for k, v in batch['cat'].items())}, "
                       f"目标 {batch['target'].shape}")
            
            for name, layer in embedding_layers.items():
                logger.info(f"嵌入层 {name}: 类别数={layer.embedding.num_embeddings}, 维度={layer.embedding.embedding_dim}")
            
            break