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
import gc

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
        初始化LST数据集
        
        🔧 关键修复：严格控制fit_features和fit_target参数
        
        Args:
            features: 特征字典
            target: 目标变量字典
            scaler_dict: 标准化器字典(如果为None则创建新的,如果提供则使用现有的)
            embedding_dims: 嵌入维度字典
            is_train: 是否为训练模式(True时fit新scaler,False时使用传入的scaler)
            is_global: 是否为全局数据
            fit_features: 🔧 是否fit特征scaler（新增参数）
            fit_target: 🔧 是否fit目标scaler（新增参数）
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
            self.scaler_dict = scaler_dict.copy()  # 复制以避免修改原始dict
            logger.info(f"使用传入的scaler_dict,包含 {len(scaler_dict)} 个键")

        # ----------------
        # 1. 合并为 DataFrame,便于行过滤
        # ----------------
        target_var = list(target.keys())[0]
        df = pd.DataFrame(features)
        df[target_var] = target[target_var]

        # 去除含 NaN 的行
        before_rows = len(df)
        df = df.dropna()
        after_rows = len(df)
        logger.info(f"已剔除含 NaN 的样本行: {before_rows - after_rows} 行,剩余 {after_rows} 行")

        # 数据修剪(仅在全局训练时)
        if self.is_train and self.is_global and after_rows > 0:
            target_values = df[target_var].values
            
            lower_bound = np.nanpercentile(target_values, 0.1)
            upper_bound = np.nanpercentile(target_values, 99.9)
            
            trim_mask = (target_values >= lower_bound) & (target_values <= upper_bound)
            df = df[trim_mask]
            
            trimmed_rows = after_rows - len(df)
            after_trim_rows = len(df)
            
            logger.info(f"【训练数据修剪】: 剔除了最小/最大 0.1% 的数据,共 {trimmed_rows} 行。新样本数: {after_trim_rows} 行。")
            logger.info(f"修剪边界: {target_var} 范围 [{lower_bound:.2f}, {upper_bound:.2f}] (原 {after_rows} 行)")
            after_rows = after_trim_rows

        # 拆分回 features / target
        self.features = {k: df[k].values for k in features.keys()}
        self.target = {target_var: df[target_var].values}

        # 清理不需要的内存
        del df
        # ----------------
        # 2. 分类 / 时间 / 数值 / 地理特征区分
        # ----------------
        self.num_features = {}
        self.cat_features = {}
        self.time_features = {}
        self.loc_features = {}

        ignored_vars = ["lon", "lat"]
        categorical_vars = ["clcd", "sm_surface_wetness"]
        
        for k, v in self.features.items():
            if k in ignored_vars:
                logger.debug(f"特征 {k} 为原始经纬度,已忽略,不作为模型输入特征。")
                continue
            if k in categorical_vars or k in self.embedding_dims:
                self.cat_features[k] = v.astype(np.int64)
                if len(v) > 0:
                    logger.info(f"分类变量 {k}: 唯一值数量={len(np.unique(v))}, 取值范围=[{np.min(v)}, {np.max(v)}]")
                else:
                    logger.info(f"分类变量 {k}: 数据为空")
            elif k in ["day", "hour", "doy"]:
                self.time_features[k] = v.astype(np.int64)
            elif k in ["lon_sin", "lon_cos", "lat_sin", "lat_cos"]:
                self.loc_features[k] = v.astype(np.float32)
            else:
                self.num_features[k] = v.astype(np.float32)
        # 清理不需要的内存
        del features   

        # ----------------
        # 3. 🔧 关键修复：数值特征标准化（严格遵守fit_features参数）
        # ----------------
        no_standardize_features = ['_1_km_16_days_NDVI', 
                                    'lon', 'lat',
                                    'dem', 'slope', 'aspect',
                                    'dem_1km', 'slope_1km', 'aspect_1km']
        
        logger.info(f"开始对数值特征进行标准化(除 {no_standardize_features} 外)...")
        logger.info(f"🔧 fit_features={self.fit_features}, fit_target={self.fit_target}")
        
        geo_features_standardized = []

        for k, v in self.num_features.items():
            if k in no_standardize_features:
                logger.info(f"特征 {k} 跳过标准化处理。")
                continue
            
            # 记录地理特征
            if k in ['dem', 'slope', 'aspect', 'dem_1km', 'slope_1km', 'aspect_1km']:
                geo_features_standardized.append(k)
            
            # 🔧 关键修复：严格按照fit_features参数决定是否fit
            if self.fit_features:
                # 需要fit新的scaler
                scaler = StandardScaler()
                self.num_features[k] = scaler.fit_transform(v.reshape(-1, 1)).flatten()
                self.scaler_dict[k] = scaler
                logger.info(f"特征 {k} (fit模式): 已fit并标准化, mean={scaler.mean_[0]:.3f}, std={scaler.scale_[0]:.3f}")
            else:
                # 不fit，必须使用传入的scaler
                scaler = self.scaler_dict.get(k)
                if scaler:
                    self.num_features[k] = scaler.transform(v.reshape(-1, 1)).flatten()
                    logger.info(f"特征 {k} (use模式): 使用传入scaler标准化, mean={scaler.mean_[0]:.3f}, std={scaler.scale_[0]:.3f}")
                else:
                    logger.warning(f"特征 {k} 无scaler, 使用原始值!")
        
        if geo_features_standardized:
            logger.info(f"【地理特征检查】:以下地理特征已确认进行 StandardScaler 标准化: {', '.join(geo_features_standardized)}")
        else:
            logger.warning("【地理特征检查】:未发现明显的 DEM/Slope/Aspect 特征进行标准化,请检查特征名称是否正确!")
        # 清理不需要的内存
        del geo_features_standardized

        # ----------------
        # 4. 时间特征正余弦编码
        # ----------------
        time_encoded = []
        for k, v in self.time_features.items():
            if k == "day": T = 31
            elif k == "hour": T = 24
            elif k == "doy": T = 365
            else: continue
            time_encoded.append(np.sin(2 * np.pi * v / T).astype(np.float32))
            time_encoded.append(np.cos(2 * np.pi * v / T).astype(np.float32))
        self.X_time = torch.tensor(np.stack(time_encoded, axis=1), dtype=torch.float32) if time_encoded else torch.zeros((len(self.target[target_var]), 0))
        # 清理不需要的内存
        del time_encoded

        # ----------------
        # 5. 地理特征 (直接拼接)
        # ----------------
        if self.loc_features:
            loc_arr = np.stack(list(self.loc_features.values()), axis=1)
            self.X_loc = torch.tensor(loc_arr, dtype=torch.float32)
        else:
            self.X_loc = torch.zeros((len(self.target[target_var]), 0))
        # 清理不需要的内存
        del loc_arr 
        # ----------------
        # 6. 数值特征拼接
        # ----------------
        if self.num_features:
            num_arr = np.stack(list(self.num_features.values()), axis=1)
            self.X_num = torch.tensor(num_arr, dtype=torch.float32)
        else:
            self.X_num = torch.zeros((len(self.target[target_var]), 0))
        # 清理不需要的内存
        del num_arr

        # ----------------
        # 7. 分类变量处理 - 智能选择方法
        # ----------------
        if self.cat_features:
            total_samples = len(list(self.cat_features.values())[0])
            
            if total_samples > 50000000:
                logger.info(f"⚠️ 数据量大 ({total_samples:,})，分块处理")
                self.X_cat, self.embedding_info = self._process_categorical_chunked(30000000)
            else:
                logger.info(f"数据量适中 ({total_samples:,})，pandas处理")
                self.X_cat, self.embedding_info = self._process_categorical_pandas()
        else:
            self.X_cat, self.embedding_info = {}, {}

        if self.is_train:
            self.scaler_dict['cat_features'] = self.embedding_info
        # 清理不需要的内存
        del total_samples
        '''        
        # ----------------
        # 7. 分类变量处理(用于嵌入编码)
        # ----------------
        self.X_cat = {}
        self.embedding_info = {}
        
        for k, v in self.cat_features.items():           
            unique_vals, inverse_indices = np.unique(v, return_inverse=True)
            num_classes = len(unique_vals)
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            mapped_v = inverse_indices
            
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
        
            # 显式删除大型临时变量
            del unique_vals, inverse_indices, val_to_idx, mapped_v

        # 在训练模式下保存embedding信息到scaler_dict
        if self.is_train:
            self.scaler_dict['cat_features'] = self.embedding_info
            logger.info(f"已保存 {len(self.embedding_info)} 个分类特征的embedding信息到scaler_dict")'''

        # ----------------
        # 8. 🔧 关键修复：目标变量处理（严格遵守fit_target参数）
        # ----------------
        target_var = list(self.target.keys())[0]
        self.y_raw = self.target[target_var].astype(np.float32)

        if self.fit_target:
            # 需要fit新的target scaler
            self.target_scaler = StandardScaler()
            self.y = self.target_scaler.fit_transform(self.y_raw.reshape(-1, 1)).flatten()
            self.scaler_dict['target'] = self.target_scaler
            logger.info(f"目标 {target_var} (fit模式): 已fit并标准化, mean={self.target_scaler.mean_[0]:.2f}K, std={self.target_scaler.scale_[0]:.2f}K")
        else:
            # 不fit，必须使用传入的target scaler
            self.target_scaler = self.scaler_dict.get('target')
            if self.target_scaler:
                self.y = self.target_scaler.transform(self.y_raw.reshape(-1, 1)).flatten()
                logger.info(f"目标 {target_var} (use模式): 使用传入scaler标准化, mean={self.target_scaler.mean_[0]:.2f}K, std={self.target_scaler.scale_[0]:.2f}K")
                
                # 🔍 额外验证：检查标准化后的分布
                logger.info(f"  标准化后分布: mean={self.y.mean():.3f}, std={self.y.std():.3f}, "
                           f"range=[{self.y.min():.3f}, {self.y.max():.3f}]")
            else:
                self.y = self.y_raw
                logger.warning("无目标scaler, 使用原始目标值")
        
        self.y = torch.tensor(self.y, dtype=torch.float32)

        # 最终汇总
        logger.info(f"scaler_dict 最终包含的键: {list(self.scaler_dict.keys())}")
        logger.info(f"LSTDataset 初始化完成: 数值特征 {self.X_num.shape[1]}, "
                    f"时间特征 {self.X_time.shape[1]}, 地理特征 {self.X_loc.shape[1]}, "
                    f"分类特征 {list(self.X_cat.keys())}, 样本数 {len(self.y)}")

    def _process_categorical_chunked(self, chunk_size=30000000):
            """分块处理分类特征 - 每块3000万样本"""
            X_cat = {}
            embedding_info = {}
            
            for k, v in self.cat_features.items():
                num_samples = len(v)
                logger.info(f"处理分类变量 {k}，样本数: {num_samples:,}，分块处理中...")
                
                # 阶段1：收集唯一值
                unique_set = set()
                for i in range(0, num_samples, chunk_size):
                    chunk_end = min(i + chunk_size, num_samples)
                    unique_set.update(np.unique(v[i:chunk_end]).tolist())
                
                unique_vals = np.array(sorted(unique_set))
                num_classes = len(unique_vals)
                val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                del unique_set
                gc.collect()
                
                # 阶段2：映射索引
                mapped_array = np.empty(num_samples, dtype=np.int64)
                for i in range(0, num_samples, chunk_size):
                    chunk_end = min(i + chunk_size, num_samples)
                    chunk = v[i:chunk_end]
                    mapped_array[i:chunk_end] = [val_to_idx[val] for val in chunk]
                
                X_cat[k] = torch.tensor(mapped_array, dtype=torch.long)
                
                embed_dim = self.embedding_dims.get(k, {}).get('embed_dim', min(50, (num_classes + 1) // 2))
                
                embedding_info[k] = {
                    'num_classes': num_classes,
                    'embed_dim': embed_dim,
                    'val_to_idx': val_to_idx,
                    'unique_vals': unique_vals
                }
                
                logger.info(f"✅ {k}: 类别数={num_classes}")
                del mapped_array, unique_vals
                gc.collect()
            
            return X_cat, embedding_info
    
    def _process_categorical_pandas(self):
        """使用 pandas 快速处理"""
        import pandas as pd
        X_cat, embedding_info = {}, {}
        
        for k, v in self.cat_features.items():
            cat_series = pd.Categorical(v)
            X_cat[k] = torch.tensor(cat_series.codes.astype(np.int64), dtype=torch.long)
            
            num_classes = len(cat_series.categories)
            embed_dim = self.embedding_dims.get(k, {}).get('embed_dim', min(50, (num_classes + 1) // 2))
            
            embedding_info[k] = {
                'num_classes': num_classes,
                'embed_dim': embed_dim,
                'val_to_idx': {int(val): idx for idx, val in enumerate(cat_series.categories)},
                'unique_vals': cat_series.categories.to_numpy()
            }
            del cat_series
            gc.collect()
        
        return X_cat, embedding_info
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
    target_date = datetime(2018, 10, 6)
    
    logger.info("\n--- 调试全局预训练数据加载 ---")
    data = loader.load_data(target_date, is_global=False, is_pretrain=False)
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