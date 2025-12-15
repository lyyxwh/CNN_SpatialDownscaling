"""
LST时空插值器 - 重构版（更新：直接剔除含NaN的样本）
插值逻辑：
1. 原始LST存在 → 保留原始值
2. 仅LST缺失，其他特征完整 → 模型预测
3. CLCD和LST同时缺失 → 设为nodata
4. LST缺失但CLCD存在（其他特征可能缺失）→ 空间插值（仅使用步骤1、2的有效值）
"""
import os
import json
import torch
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader as LSTDataLoader
from model import LSTTransformer

# 配置日志
log_dir = os.path.join(os.path.dirname(__file__), '../output/interpolation')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'interpolation.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 文件日志
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LSTInterpolator:
    """LST时空插值器 - 新逻辑"""
    
    def __init__(self, config_path, local_finetune_dir):
        """初始化插值器"""
        self.config_path = config_path
        self.local_finetune_dir = local_finetune_dir
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        self.data_loader = LSTDataLoader(config_path)
        self.model_cache = {}
        self.scaler_cache = {}
        
    def load_model(self, target_date):
        """加载局部微调模型及其scaler"""
        date_str = target_date.strftime('%Y%m%d')
        cache_key = f"local_{date_str}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key], self.scaler_cache[cache_key]
        
        model_path = os.path.join(self.local_finetune_dir, f"local_finetune_{date_str}.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint['config']
        scaler_dict = checkpoint['scaler_dict']
        
        model = LSTTransformer(model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.model_cache[cache_key] = model
        self.scaler_cache[cache_key] = scaler_dict
        
        logger.info(f"✓ 模型加载完成: {model_path}")
        
        return model, scaler_dict

    def interpolate_hourly(self, target_date, target_hour):
        """
        插值指定小时 - 新逻辑
        
        插值顺序：
        1. 保留原始LST值
        2. 仅LST缺失 → 模型预测
        3. CLCD和LST同时缺失 → 设为nodata
        4. LST缺失但CLCD存在 → 空间插值
        """
        logger.info(f"插值 {target_date.strftime('%Y-%m-%d')} {target_hour:02d}:00")
        
        # 步骤1: 加载数据
        hourly_data = self._load_hourly_data(target_date, target_hour)
        if hourly_data is None:
            raise ValueError("数据加载失败")
        
        spatial_shape = hourly_data['spatial_shape']
        features = hourly_data['features']
        observed_lst = hourly_data['target']
        
        # 转为2D
        observed_lst_2d = observed_lst.reshape(spatial_shape)
        
        # 步骤2: 像元分类
        pixel_classes = self._classify_pixels(features, observed_lst, spatial_shape)
        
        logger.info(f"\n像元分类统计:")
        logger.info(f"  第1类（保留原始LST）: {pixel_classes['type1_count']} 像元")
        logger.info(f"  第2类（模型预测）: {pixel_classes['type2_count']} 像元")
        logger.info(f"  第3类（永久nodata）: {pixel_classes['type3_count']} 像元")
        logger.info(f"  第4类（空间插值）: {pixel_classes['type4_count']} 像元")
        logger.info(f"  总像元数: {spatial_shape[0] * spatial_shape[1]}")
        
        # 载入局部模型（提前加载以便对第1类也进行预测用于评估）
        try:
            model, scaler_dict = self.load_model(target_date)
        except Exception as e:
            model, scaler_dict = None, None
            logger.warning(f"无法加载局部模型，跳过第1类模型预测: {e}")

        # 初始化结果数组
        final_lst_2d = np.full(spatial_shape, np.nan, dtype=np.float32)
        predicted_lst_2d = np.full(spatial_shape, np.nan, dtype=np.float32)  # 用于评估（现在用于保存模型预测，无论第1/2类）
        lst_global_2d = np.full(spatial_shape, np.nan, dtype=np.float32)
        qc_2d = np.full(spatial_shape, 255, dtype=np.uint8)

        # 步骤3: 处理第1类（保留原始LST），同时用模型对第1类/可预测的原始像元做预测以便评估
        type1_indices = pixel_classes['type1_indices']
        if len(type1_indices) > 0:
            # 保留原始LST为最终输出
            final_lst_2d.flat[type1_indices] = observed_lst[type1_indices]
            qc_2d.flat[type1_indices] = 0

            # 若能加载模型，则对第1类像元进行预测（不覆盖 final_lst_2d，仅写入 predicted_lst_2d）
            if model is not None and scaler_dict is not None:
                # 构建第1类特征字典（按原始 features 顺序）
                type1_features = {key: values[type1_indices] for key, values in features.items()}
                try:
                    type1_preds, valid_type1_indices = self._predict_pixels_with_nan_removal(
                        model, scaler_dict, type1_features, type1_indices
                    )
                    if len(valid_type1_indices) > 0:
                        # 写入 predicted_lst_2d（用于评估）；不要修改 qc_2d（仍为0表示原始值来源）
                        predicted_lst_2d.flat[valid_type1_indices] = type1_preds
                        #logger.info(f"  对第1类原始像元进行了模型预测用于评估: 原始数 {len(type1_indices)}, 成功预测 {len(valid_type1_indices)}")
                    else:
                        logger.info("  第1类模型预测无有效结果（可能全部含NaN）")
                except Exception as e:
                    logger.warning(f"  第1类模型预测失败: {e}")
            else:
                logger.info("  未加载模型，跳过第1类模型预测（仅保留原始值用于输出）")

            logger.info(f"  已保留 {len(type1_indices)} 个原始LST值")
            #logger.info(f"  原始LST范围: [{observed_lst[type1_indices].min():.2f}, "
                       #f"{observed_lst[type1_indices].max():.2f}] K")
        
        # 步骤4: 处理第2类（模型预测）
        type2_indices = pixel_classes['type2_indices']
        
        if len(type2_indices) > 0:
            model, scaler_dict = self.load_model(target_date)
            
            # 提取第2类像元的特征
            type2_features = {key: values[type2_indices] for key, values in features.items()}
            
            try:
                # ✅ 关键修改：在预测前剔除含NaN的样本
                type2_predictions, valid_type2_indices = self._predict_pixels_with_nan_removal(
                    model, scaler_dict, type2_features, type2_indices
                )
                
                # 只填充成功预测的像元
                if len(valid_type2_indices) > 0:
                    final_lst_2d.flat[valid_type2_indices] = type2_predictions
                    predicted_lst_2d.flat[valid_type2_indices] = type2_predictions
                    
                    logger.info(f"  原始第2类像元数: {len(type2_indices)}")
                    logger.info(f"  剔除含NaN样本后: {len(valid_type2_indices)}")
                    #logger.info(f"  预测成功率: {len(valid_type2_indices)/len(type2_indices)*100:.1f}%")
                    
                    '''if len(type2_predictions) > 0:
                       logger.info(f"  预测LST范围: [{type2_predictions.min():.2f}, {type2_predictions.max():.2f}] K")
                        logger.info(f"  预测LST均值: {type2_predictions.mean():.2f} K")
                else:
                    logger.warning("  ⚠️ 所有第2类像元都含有NaN，无法预测")'''
                
            except Exception as e:
                logger.error(f"  模型预测失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 步骤5: 处理第3类（永久nodata）
        type3_indices = pixel_classes['type3_indices']
        if len(type3_indices) > 0:
            logger.info(f"  已设置 {len(type3_indices)} 个像元为nodata")
        
        # 步骤6: 处理第4类（空间插值）
        type4_indices = pixel_classes['type4_indices']
        
        if len(type4_indices) > 0:
            # 使用步骤3和4的结果进行空间插值
            base_for_interp = final_lst_2d.copy()
            
            # 创建插值掩码：只有第1类和第2类的像元可以作为插值源
            valid_for_interp = np.zeros(spatial_shape, dtype=bool)
            valid_for_interp.flat[type1_indices] = True
            valid_for_interp.flat[type2_indices] = True
            valid_for_interp = valid_for_interp & ~np.isnan(base_for_interp)
            
            logger.info(f"  需要插值: {len(type4_indices)} 个像元")
            
            if np.sum(valid_for_interp) > 0:
                interpolated_2d = self._spatial_interpolate(
                    base_for_interp, 
                    type4_indices, 
                    valid_for_interp,
                    spatial_shape
                )
                
                # 将插值结果填充到最终数组
                final_lst_2d.flat[type4_indices] = interpolated_2d.flat[type4_indices]
                
                # 统计插值成功率
                success_count = np.sum(~np.isnan(final_lst_2d.flat[type4_indices]))
                logger.info(f"  插值成功: {success_count}/{len(type4_indices)} 个像元")
            else:
                logger.warning("  ⚠️ 无有效插值源，第4类像元将保持为nodata")
        
        # 评估（仅对第1类像元：有原始观测的点）

        
        if len(type1_indices) > 0:
            obs_true = observed_lst[type1_indices]
            obs_pred = predicted_lst_2d.flat[type1_indices]
            
            # 过滤掉预测失败的点
            valid_eval_mask = ~np.isnan(obs_pred)
            if np.sum(valid_eval_mask) > 0:
                obs_true_valid = obs_true[valid_eval_mask]
                obs_pred_valid = obs_pred[valid_eval_mask]
                
                rmse = np.sqrt(mean_squared_error(obs_true_valid, obs_pred_valid))
                mae = mean_absolute_error(obs_true_valid, obs_pred_valid)
                r2 = r2_score(obs_true_valid, obs_pred_valid)
                bias = np.mean(obs_pred_valid - obs_true_valid)
                
                logger.info(f"  评估点数: {len(obs_true_valid)}")
                logger.info(f"  RMSE: {rmse:.4f} K")
                logger.info(f"  MAE: {mae:.4f} K")
                logger.info(f"  R²: {r2:.4f}")
                logger.info(f"  Bias: {bias:.4f} K")
                
                if abs(bias) > 3:
                    logger.warning(f"  ⚠️ 系统性偏差较大: {bias:.2f}K")
            else:
                logger.warning("  所有观测点预测失败，无法评估")
        else:
            logger.warning("  无观测点，无法评估")
        
        # 最终统计
        logger.info("【插值完成统计】")
        final_valid = np.sum(~np.isnan(final_lst_2d))
        final_nodata = np.sum(np.isnan(final_lst_2d))
        total = final_lst_2d.size
        
        logger.info(f"  总像元数: {total}")
        logger.info(f"  有效像元: {final_valid} ({final_valid/total*100:.2f}%)")
        logger.info(f"    - 原始LST: {len(type1_indices)}")
        logger.info(f"    - 模型预测: {len(type2_indices)}")
        logger.info(f"    - 空间插值: {len(type4_indices)}")
        logger.info(f"  nodata像元: {final_nodata} ({final_nodata/total*100:.2f}%)")
        
        return self._prepare_result(
            final_lst_2d,
            observed_lst_2d,
            predicted_lst_2d,
            spatial_shape,
            hourly_data['datetime'],
            pixel_classes
        )
    
    def _classify_pixels(self, features, observed_lst, spatial_shape):
        """
        像元分类
        
        分类标准（改为基于 CLCD 判断可否插补）：
        - 第1类：LST存在（有观测值）
        - 第2类：LST缺失 + CLCD存在 + 其他关键特征完整  -> 模型预测
        - 第3类：LST缺失 + CLCD缺失 -> 永久nodata（无法插补）
        - 第4类：LST缺失 + CLCD存在 + 其他特征有缺失 -> 空间插值
        """
        total_pixels = len(observed_lst)
        
        # 基础掩码
        has_lst = ~np.isnan(observed_lst)
        
        # CLCD掩码（用 cloud/clear 标识当前像元是否有 CLCD 信息）
        clcd_key = 'clcd'
        if clcd_key in features:
            has_clcd = ~np.isnan(features[clcd_key])
        else:
            # 如果没有 clcd 信息，则认为所有像元都缺失 CLCD（保守处理）
            has_clcd = np.zeros(total_pixels, dtype=bool)
        
        # 检查关键特征完整性（ERA5核心变量）
        critical_features = ['t2m', 'ssrd', 'strd', 'rh', 'd2m', 'vpd']
        
        # 对每个像元检查特征完整性
        features_complete = np.ones(total_pixels, dtype=bool)
        
        for feat in critical_features:
            if feat in features:
                features_complete &= ~np.isnan(features[feat])
        
        # 第1类：LST存在（保留原始值）
        type1_mask = has_lst
        type1_indices = np.where(type1_mask)[0]
        
        # 第2类：LST缺失 + CLCD存在 + 特征完整（模型预测）
        type2_mask = ~has_lst & has_clcd & features_complete
        type2_indices = np.where(type2_mask)[0]
        
        # 第3类：LST缺失 + CLCD缺失（永久nodata）
        type3_mask = ~has_lst & ~has_clcd
        type3_indices = np.where(type3_mask)[0]
        
        # 第4类：LST缺失 + CLCD存在 + 特征不完整（空间插值）
        type4_mask = ~has_lst & has_clcd & ~features_complete
        type4_indices = np.where(type4_mask)[0]
        
        # 验证分类完整性
        total_classified = (len(type1_indices) + len(type2_indices) + 
                          len(type3_indices) + len(type4_indices))
        
        if total_classified != total_pixels:
            logger.warning(f"⚠️ 像元分类不完整: {total_classified}/{total_pixels}")
        
        return {
            'type1_indices': type1_indices,
            'type1_count': len(type1_indices),
            'type2_indices': type2_indices,
            'type2_count': len(type2_indices),
            'type3_indices': type3_indices,
            'type3_count': len(type3_indices),
            'type4_indices': type4_indices,
            'type4_count': len(type4_indices)
        }
    
    def _predict_pixels_with_nan_removal(self, model, scaler_dict, features_dict, original_indices):
        """
        对特定像元进行模型预测（✅ 关键修改：剔除含NaN的样本）
        
        Args:
            model: 训练好的模型
            scaler_dict: 标准化器字典
            features_dict: 特征字典
            original_indices: 原始像元索引（用于映射回去）
            
        Returns:
            predictions: 预测结果
            valid_indices: 有效样本对应的原始索引
        """
        num_pixels = len(next(iter(features_dict.values())))
        logger.info(f"  准备预测 {num_pixels} 个像元...")
        
        # ===== 特征分类定义 =====
        features_to_standardize = ['t2m', 'ssrd', 'strd', 'rh', 'd2m', 'vpd']  # 仅ERA5
        features_no_standardize = ['_1_km_16_days_NDVI', 'dem', 'slope', 'aspect']
        
        num_feature_names = features_to_standardize + features_no_standardize
        time_feature_names = [ 'day', 'hour', 'doy']
        loc_feature_names = ['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos']
        cat_feature_names = ['clcd', 'sm_surface_wetness']
        
        # 特征重命名
        if 'dem_1km' in features_dict and 'dem' not in features_dict:
            features_dict['dem'] = features_dict['dem_1km']
        if 'slope_1km' in features_dict and 'slope' not in features_dict:
            features_dict['slope'] = features_dict['slope_1km']
        if 'aspect_1km' in features_dict and 'aspect' not in features_dict:
            features_dict['aspect'] = features_dict['aspect_1km']
        
        # ===== ✅ 关键修改：直接剔除含NaN的样本 =====
        
        # 检查所有需要的特征
        all_required_features = (num_feature_names + time_feature_names + 
                                loc_feature_names + cat_feature_names)
        
        # 创建有效样本掩码
        valid_mask = np.ones(num_pixels, dtype=bool)
        nan_stats = {}
        
        for key in all_required_features:
            if key not in features_dict:
                logger.warning(f"    特征 {key} 缺失，将被视为无效")
                valid_mask[:] = False
                continue
            
            values = features_dict[key]
            has_nan = np.isnan(values)
            nan_count = np.sum(has_nan)
            
            if nan_count > 0:
                nan_ratio = nan_count / len(values)
                nan_stats[key] = (nan_count, nan_ratio)
                valid_mask &= ~has_nan
        
        # 输出NaN统计
        '''if nan_stats:
            logger.info(f"    发现含NaN的特征:")
            for key, (count, ratio) in nan_stats.items():
                logger.info(f"      {key}: {count}个NaN ({ratio*100:.1f}%)")'''
        
        # 剔除无效样本
        num_valid = np.sum(valid_mask)
        num_removed = num_pixels - num_valid
        '''
        logger.info(f"    原始样本数: {num_pixels}")
        logger.info(f"    剔除样本数: {num_removed} ({num_removed/num_pixels*100:.1f}%)")
        logger.info(f"    有效样本数: {num_valid} ({num_valid/num_pixels*100:.1f}%)")'''
        
        if num_valid == 0:
            logger.error("    ❌ 所有样本都含有NaN，无法预测")
            return np.array([]), np.array([])
        
        # 过滤特征
        filtered_features = {}
        for key, values in features_dict.items():
            filtered_features[key] = values[valid_mask]
        
        # 保存有效样本的原始索引
        valid_original_indices = original_indices[valid_mask]
        
        # ===== 数值特征标准化（仅对有效样本） =====
        
        for key in num_feature_names:
            if key not in features_to_standardize:
                logger.info(f"    {key}: 跳过标准化")
                continue
            
            if key not in scaler_dict:
                logger.warning(f"    {key}: ⚠️ 缺少scaler")
                continue
            
            scaler = scaler_dict[key]
            values = filtered_features[key].reshape(-1, 1)
            filtered_features[key] = scaler.transform(values).flatten()
            
            after_vals = filtered_features[key]
            #logger.info(f"    {key}: 已标准化,转换后: mean={after_vals.mean():.3f}, std={after_vals.std():.3f}")
            
        # 🔍 在标准化后添加诊断
        for key in ['t2m', 'ssrd', 'strd', 'rh', 'd2m', 'vpd']:
            if key in filtered_features and key in scaler_dict:
                values = filtered_features[key]
                #logger.info(f"  {key}:，标准化后范围: [{values.min():.3f}, {values.max():.3f}]， 标准化后均值: {values.mean():.3f}，标准化后标准差: {values.std():.3f}")
              
                #如果标准化后的值范围过小，警告
                if values.std() < 0.1:
                    logger.warning(f"     {key} 标准化后标准差过小！可能所有样本值接近")
                
        
        # ===== 时间特征编码 =====
        time_encoded = []
        for key, period in [('day', 31), ('hour', 24), ('doy', 365)]:
            values = filtered_features[key]
            time_encoded.append(np.sin(2 * np.pi * values / period).astype(np.float32))
            time_encoded.append(np.cos(2 * np.pi * values / period).astype(np.float32))
        time_features = np.stack(time_encoded, axis=-1)
        
        # ===== 分类特征映射 =====        
        cat_features = {}
        
        if 'cat_features' not in scaler_dict:
            logger.error(f"    ❌ scaler_dict缺少cat_features")
            raise ValueError("无法进行分类特征映射")
        
        cat_config = scaler_dict['cat_features']
        
        for cat_name in cat_feature_names:
            if cat_name not in filtered_features:
                logger.warning(f"    {cat_name}: 缺失，跳过")
                continue
            
            if cat_name not in cat_config:
                logger.error(f"  {cat_name}配置缺失")
                raise ValueError(f"{cat_name}配置不存在")
            
            cat_info = cat_config[cat_name]
            val_to_idx = cat_info.get('val_to_idx', {})
            num_classes = cat_info.get('num_classes')
            
            # 映射（分类特征不含NaN，因为已经被过滤）
            raw_values = filtered_features[cat_name].astype(np.int64)
            mapped_values = np.array([val_to_idx.get(val, 0) for val in raw_values], 
                                    dtype=np.int64)
            
            cat_features[cat_name] = mapped_values
            
            #logger.info(f"    {cat_name}: 映射完成")
            #logger.info(f"       映射检查通过")
        
        # ===== 构建模型输入 =====       
        # 确保特征顺序严格与训练时一致（非常重要）
        num_feature_order = features_to_standardize + features_no_standardize
        time_feature_order = [ 'day', 'hour', 'doy']
        loc_feature_order = ['lon_sin', 'lon_cos', 'lat_sin', 'lat_cos']  # 与训练时顺序保持一致
        cat_feature_order = ['sm_surface_wetness', 'clcd']  # 与训练时保持一致
        
        # 验证所有需要的特征存在
        missing = [k for k in (num_feature_order + time_feature_order + loc_feature_order + cat_feature_order) if k not in filtered_features]
        if missing:
            logger.warning(f"缺失这些特征，预测可能有问题: {missing}")
        
        num_features = np.stack([filtered_features[k] for k in num_feature_order if k in filtered_features], axis=-1)
        loc_features = np.stack([filtered_features[k] for k in loc_feature_order if k in filtered_features], axis=-1)
        time_features = time_features.astype(np.float32)
        
        # 分类特征按固定顺序构建二维数组（如果存在）
        cat_arrays = []
        for k in cat_feature_order:
            if k in cat_features:
                cat_arrays.append(cat_features[k])
            else:
                # 填充默认类别0（已过滤掉含NaN样本，仍作保护）
                cat_arrays.append(np.zeros(num_features.shape[0], dtype=np.int64))
        # 将 cat_features 保持为 dict 便于后续按键批量取
        cat_for_model = {cat_feature_order[i]: cat_arrays[i] for i in range(len(cat_feature_order))}
        
        logger.info(f"    数值特征: {num_features.shape}")
        logger.info(f"    时间特征: {time_features.shape}")
        logger.info(f"    地理特征: {loc_features.shape}")
        logger.info(f"    分类特征: { {k: v.shape for k, v in cat_for_model.items()} }")
        
        # ===== 批量预测（增加诊断） =====
        batch_size = 1024
        predictions = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, num_valid, batch_size), desc="    预测"):
                end = min(i + batch_size, num_valid)
                
                b_num = torch.tensor(num_features[i:end], dtype=torch.float32).to(self.device)
                b_time = torch.tensor(time_features[i:end], dtype=torch.float32).to(self.device)
                b_loc = torch.tensor(loc_features[i:end], dtype=torch.float32).to(self.device)
                b_cat = {k: torch.tensor(v[i:end], dtype=torch.long).to(self.device) for k, v in cat_for_model.items()}
                
                batch_data = {'num': b_num, 'time': b_time, 'loc': b_loc, 'cat': b_cat}
                outputs = model(batch_data).squeeze(-1).cpu().numpy()
                
                # 诊断每批次的输出分布
                #logger.info(f"      批次 {i}-{end}: outputs raw -> min={outputs.min():.6f}, max={outputs.max():.6f}, mean={outputs.mean():.6f}, std={outputs.std():.6f}")
                
                predictions.append(outputs)
        
        if len(predictions) == 0:
            logger.error("    ❌ 无预测结果（可能所有样本含NaN）")
            return np.array([]), np.array([])
        
        predictions = np.concatenate(predictions)
        
        # ===== 反标准化诊断 =====
        if 'target' not in scaler_dict:
            logger.error("    ❌ scaler_dict 中缺少 'target'，无法反标准化")
        else:
            target_scaler = scaler_dict['target']
            #logger.info(f"    target scaler mean: {getattr(target_scaler, 'mean_', None)}, scale: {getattr(target_scaler, 'scale_', None)}")
            predictions_denorm = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            #logger.info(f"    反标准化后: min={predictions_denorm.min():.4f}, max={predictions_denorm.max():.4f}, mean={predictions_denorm.mean():.4f}, std={predictions_denorm.std():.4f}")
        
        return predictions_denorm, valid_original_indices
    
    def _spatial_interpolate(self, data_2d, target_indices, valid_mask_2d, spatial_shape, 
                             max_radius=10, min_neighbors=3, power=2, max_neighbors=32):
        """
        向量化 IDW 插值（基于 scipy.spatial.cKDTree，加速大量点插值）
        Args:
            data_2d: 当前2D LST数据（包含步骤1和2的结果）
            target_indices: 需要插值的像元索引（1D）
            valid_mask_2d: 可用作插值源的像元掩码（2D）
            spatial_shape: (rows, cols)
            max_radius: 最大搜索半径（像素）
            min_neighbors: 最少邻居数（fallback 使用最近邻）
            power: IDW 幂参数
            max_neighbors: 查询邻居上限（性能/精度折中）
        Returns:
            interpolated: 填充后2D数组
        """
        interpolated = data_2d.copy()
        nrows, ncols = spatial_shape

        # 提取有效源点坐标和值
        src_rows, src_cols = np.where(valid_mask_2d & ~np.isnan(data_2d))
        src_vals = data_2d[valid_mask_2d & ~np.isnan(data_2d)]
        n_src = src_vals.size
        if n_src == 0:
            logger.warning("  ⚠ 无有效插值源，全部目标保持为 NaN")
            for idx in np.unravel_index(target_indices, spatial_shape):
                pass
            for i, j in np.unravel_index(target_indices, spatial_shape):
                interpolated[i, j] = np.nan
            return interpolated

        src_coords = np.column_stack((src_rows, src_cols))
        tree = cKDTree(src_coords)

        # 构建目标坐标
        tgt_rows, tgt_cols = np.unravel_index(target_indices, spatial_shape)
        tgt_coords = np.column_stack((tgt_rows, tgt_cols))
        n_tgt = tgt_coords.shape[0]

        k = min(max_neighbors, max(min_neighbors, n_src))
        # 批量查询 k 个最近邻（距离为像素单位）
        dists, idxs = tree.query(tgt_coords, k=k, workers=-1)

        # 确保二维形状
        if k == 1:
            dists = dists.reshape(-1, 1)
            idxs = idxs.reshape(-1, 1)

        # 将超出 max_radius 的距离置为 inf（不参与加权）
        within_mask = (dists <= max_radius)
        # 计算权重矩阵（避免除零）
        eps = 1e-8
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0 / np.power(dists + eps, power)
        weights[~np.isfinite(weights)] = 0.0  # 将 inf/nan 权重置0
        weights[~within_mask] = 0.0

        # 行和归一化
        wsum = weights.sum(axis=1)
        result_vals = np.full(n_tgt, np.nan, dtype=np.float32)

        # 对于有合法邻居（wsum>0），直接计算加权平均
        good_rows = wsum > 0
        if np.any(good_rows):
            idxs_good = idxs[good_rows]  # (m, k)
            vals_matrix = src_vals[idxs_good]  # (m, k)
            w_good = weights[good_rows]
            numer = np.sum(w_good * vals_matrix, axis=1)
            denom = np.sum(w_good, axis=1)
            result_vals[good_rows] = numer / denom

        # 对于没有半径内邻居的点（wsum==0），使用最近的 min_neighbors 个点（不考虑 radius）
        need_fallback = ~good_rows
        if np.any(need_fallback):
            # 对这些点选择前 m = min(min_neighbors, k) 列
            m = min(min_neighbors, k)
            idxs_fb = idxs[need_fallback, :m]  # (q, m)
            dists_fb = dists[need_fallback, :m]
            # 如果存在 inf（源点数 < m），要处理
            valid_mask_fb = np.isfinite(dists_fb)
            if not np.any(valid_mask_fb):
                # 无可用邻居，保持 NaN
                logger.debug("  部分目标点无可用最近邻，保持为 NaN")
            else:
                # 计算权重，忽略无效位置
                with np.errstate(divide='ignore', invalid='ignore'):
                    w_fb = 1.0 / np.power(dists_fb + eps, power)
                w_fb[~valid_mask_fb] = 0.0
                vals_fb = src_vals[idxs_fb]
                numer_fb = np.sum(w_fb * vals_fb, axis=1)
                denom_fb = np.sum(w_fb, axis=1)
                valid_fb = denom_fb > 0
                # 将计算结果写回
                result_vals[need_fallback.nonzero()[0][valid_fb]] = numer_fb[valid_fb] / denom_fb[valid_fb]
                # 如果仍有 denom_fb == 0，则保持 NaN

        # 将结果写回插值网格
        for idx_pos, val in enumerate(result_vals):
            if np.isnan(val):
                interpolated[tgt_rows[idx_pos], tgt_cols[idx_pos]] = np.nan
            else:
                interpolated[tgt_rows[idx_pos], tgt_cols[idx_pos]] = val

        success = np.sum(~np.isnan(result_vals))
        logger.info(f"  IDW 向量化插值: 成功 {success}/{n_tgt} ({success/n_tgt*100:.1f}%)")
        return interpolated
 
    def _load_hourly_data(self, target_date, target_hour):
        """加载指定小时的数据"""
        target_datetime = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=target_hour)
        
        paths = self.data_loader.local_paths
        shape = self.data_loader.local_shape
        lon_lat = self.data_loader.lon_lat_local
        static_data = self.data_loader.static_data_local
        
        total_pixels = np.prod(shape)
        
        # 加载MODIS LST
        start_time = target_datetime - timedelta(minutes=30)
        end_time = target_datetime + timedelta(minutes=30)
        modis_files = self.data_loader.get_file_list(paths['modis'], start_time, end_time, 'modis')
        
        lst_data = np.full(total_pixels, np.nan, dtype=np.float32)
        
        if modis_files:
            target_file = self.data_loader.find_closest_file(modis_files, target_datetime, timedelta(hours=1))
            if target_file:
                logger.info(f"  加载MODIS: {os.path.basename(target_file)}")
                lst_data = self.data_loader._load_dynamic_data(
                    target_file,
                    self.data_loader.var_names['modis'],
                    shape,
                    self.data_loader.nodata_values.get('modis'),
                    data_type='modis'
                )
        
        # 准备特征
        features = {}
        
        # 静态特征
        for var_name, data in static_data.items():
            features[var_name] = data
        
        # 地理特征
        features['lon'] = lon_lat[:, 0]
        features['lat'] = lon_lat[:, 1]
        
        lon_rad = np.radians(features['lon'])
        lat_rad = np.radians(features['lat'])
        features['lon_sin'] = np.sin(lon_rad)
        features['lon_cos'] = np.cos(lon_rad)
        features['lat_sin'] = np.sin(lat_rad)
        features['lat_cos'] = np.cos(lat_rad)
        
        # 辅助数据
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 辅助数据加载诊断")
        logger.info(f"目标时间: {target_datetime}")
        logger.info(f"{'='*80}")
        
        aux_start = target_datetime - timedelta(hours=12)
        aux_end = target_datetime + timedelta(hours=12)
        
        max_deltas = {
            'era5_1km': timedelta(hours=6),
            'smap_1km': timedelta(hours=12),
            'ndvi_1km': timedelta(days=16)
        }
        
        aux_data_info = {}  # 记录每个辅助数据的信息
        
        for var_type in ['era5_1km', 'smap_1km', 'ndvi_1km']:
            if var_type not in paths:
                continue
            
            aux_files = self.data_loader.get_file_list(paths[var_type], aux_start, aux_end, var_type)
            
            logger.info(f"\n{var_type}:")
            logger.info(f"  搜索窗口: {aux_start} 至 {aux_end}")
            logger.info(f"  找到文件数: {len(aux_files)}")
            
            if aux_files:
                closest = self.data_loader.find_closest_file(aux_files, target_datetime, max_deltas[var_type])
                if closest:
                    # 🔍 解析文件时间
                    file_time = self.data_loader.parse_time_from_filename(
                        os.path.basename(closest), var_type
                    )
                    time_diff = abs((file_time - target_datetime).total_seconds() / 3600)
                    
                    logger.info(f"  ✓ 匹配文件: {os.path.basename(closest)}")
                    
                    var_names = self.data_loader.var_names.get(var_type.replace('_1km', ''), [])
                    if isinstance(var_names, str):
                        var_names = [var_names]
                    
                    for var_name in var_names:
                        data = self.data_loader._load_dynamic_data(
                            closest, var_name, shape,
                            self.data_loader.nodata_values.get(var_type.replace('_1km', '')),
                            data_type=var_type
                        )
                        if data is not None:
                            features[var_name] = data
                            
                            # 🔍 统计特征值
                            valid_data = data[~np.isnan(data)]
                            '''logger.info(f"    {var_name}:")
                            logger.info(f"      有效像元: {len(valid_data)}/{len(data)}")
                            logger.info(f"      原始范围: [{valid_data.min():.3f}, {valid_data.max():.3f}]")
                            logger.info(f"      原始均值: {valid_data.mean():.3f}")
                            logger.info(f"      原始标准差: {valid_data.std():.3f}")
                            '''
                            # 记录信息供后续对比
                            aux_data_info[var_name] = {
                                'file_time': file_time,
                                'time_diff_hours': time_diff,
                                'valid_ratio': len(valid_data) / len(data),
                                'mean': valid_data.mean(),
                                'std': valid_data.std()
                            }
                else:
                    logger.warning(f"  ✗ 未找到符合时间窗口的文件")
            else:
                logger.warning(f"  ✗ 未找到任何文件")
        
        # 时间特征
        #features['month'] = np.full(total_pixels, target_datetime.month)
        features['day'] = np.full(total_pixels, target_datetime.day)
        features['hour'] = np.full(total_pixels, target_datetime.hour)
        features['doy'] = np.full(total_pixels, target_datetime.timetuple().tm_yday)
        
        logger.info(f" 辅助数据加载完成")

        
        return {
            'features': features,
            'target': lst_data,
            'spatial_shape': shape,
            'datetime': target_datetime,
            'aux_data_info': aux_data_info  # 返回辅助数据信息
        }

    
    def _prepare_result(self, final_lst_2d, observed_lst_2d, predicted_lst_2d, 
                       spatial_shape, datetime_obj, pixel_classes):
        """准备结果字典"""
        return {
            'lst': final_lst_2d,                    # 最终插值结果
            'lst_original': observed_lst_2d,        # 原始观测
            'lst_predicted': predicted_lst_2d,      # 模型预测（用于评估）
            'spatial_shape': spatial_shape,
            'datetime': datetime_obj,
            'pixel_classes': pixel_classes,
            'stats': {
                'total_pixels': int(final_lst_2d.size),
                'type1_count': pixel_classes['type1_count'],
                'type2_count': pixel_classes['type2_count'],
                'type3_count': pixel_classes['type3_count'],
                'type4_count': pixel_classes['type4_count'],
                'final_valid': int(np.sum(~np.isnan(final_lst_2d))),
                'final_nodata': int(np.sum(np.isnan(final_lst_2d)))
            }
        }
    
    def save_netcdf(self, result, output_path):
        """保存为NetCDF文件（修复_FillValue冲突）"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        lon_range = self.config['data_params']['lon_range']
        lat_range = self.config['data_params']['lat_range']
        spatial_shape = result['spatial_shape']
        
        lons = np.linspace(lon_range[0], lon_range[1], spatial_shape[1])
        lats = np.linspace(lat_range[1], lat_range[0], spatial_shape[0])
        
        lst_data = result['lst'].astype(np.float32)
        
        # NaN替换为_FillValue
        fill_value = np.float32(-9999.0)
        lst_data_filled = np.where(np.isnan(lst_data), fill_value, lst_data)
        
        # 🔧 关键修复：_FillValue只放在encoding中，不放在attrs中
        ds = xr.Dataset(
            {
                'LST': (['lat', 'lon'], 
                       lst_data_filled,
                       {
                           'long_name': 'Land Surface Temperature',
                           'units': 'K',
                           'grid_mapping': 'spatial_ref'
                           # 移除 '_FillValue': fill_value（不放在attrs中）
                       }),
                'spatial_ref': ([], 
                               0,
                               {
                                   'spatial_ref': 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
                                   'GeoTransform': f'{lon_range[0]} {(lon_range[1]-lon_range[0])/(spatial_shape[1]-1)} 0 {lat_range[1]} 0 {(lat_range[0]-lat_range[1])/(spatial_shape[0]-1)}'
                               })
            },
            coords={
                'lat': ('lat', lats, {'units': 'degrees_north', 'long_name': 'latitude'}),
                'lon': ('lon', lons, {'units': 'degrees_east', 'long_name': 'longitude'})
            },
            attrs={
                'Conventions': 'CF-1.6',
                'title': 'Interpolated MODIS LST',
                'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                'interpolation_method': '4-step hierarchical interpolation'
            }
        )
        
        #  _FillValue只在encoding中设置
        encoding = {
            'LST': {
                'dtype': 'float32',
                '_FillValue': fill_value,  # 只在这里设置
                'zlib': True,
                'complevel': 4
            },
            'lat': {'dtype': 'float64'},
            'lon': {'dtype': 'float64'}
        }
        
        ds.to_netcdf(output_path, encoding=encoding, format='NETCDF4_CLASSIC')
        logger.info(f"✓ 已保存NetCDF: {output_path}")
        
        # 验证保存结果
        try:
            ds_check = xr.open_dataset(output_path)
            lst_check = ds_check['LST'].values
            valid_lst = lst_check[lst_check != fill_value]
            
            logger.info(f"  NetCDF验证:")
            logger.info(f"    总像元: {lst_check.size}")
            logger.info(f"    有效像元: {len(valid_lst)} ({len(valid_lst)/lst_check.size*100:.2f}%)")
            logger.info(f"    nodata像元: {np.sum(lst_check == fill_value)} ({np.sum(lst_check == fill_value)/lst_check.size*100:.2f}%)")
            
            if len(valid_lst) > 0:
                logger.info(f"    LST范围: [{valid_lst.min():.2f}, {valid_lst.max():.2f}] K")
                logger.info(f"    LST均值: {valid_lst.mean():.2f} K")
            
            ds_check.close()
            logger.info(f"✓ NetCDF验证成功")
            
        except Exception as e:
            logger.error(f"✗ NetCDF验证失败: {e}")
    
    def save_geotiff(self, result, output_dir, prefix=None):
        """
        将结果以 GeoTIFF 保存到 output_dir。
        生成文件: <prefix>_LST.tif, <prefix>_QC.tif (若存在), <prefix>_LST_global.tif (若存在)
        自动检测是否需要上下翻转以保证输出为北在上（north-up）。
        """
        os.makedirs(output_dir, exist_ok=True)

        spatial_shape = result['spatial_shape']
        lon_range = self.config['data_params']['lon_range']
        lat_range = self.config['data_params']['lat_range']

        nrows, ncols = spatial_shape
        lon_min, lon_max = float(min(lon_range)), float(max(lon_range))
        lat_min, lat_max = float(min(lat_range)), float(max(lat_range))

        # 分辨率（经纬方向增量，正值）
        x_res = (lon_max - lon_min) / (ncols - 1) if ncols > 1 else 0.0
        y_res = (lat_max - lat_min) / (nrows - 1) if nrows > 1 else 0.0

        # 构造用于判断 netCDF 中 lat 顺序的数组（与 save_netcdf 中一致的构造方式）
        # save_netcdf 使用 lats = np.linspace(lat_range[1], lat_range[0], nrows)
        lats_netcdf = np.linspace(lat_range[1], lat_range[0], nrows)
        # 若 lats_netcdf 是从大到小（第一行为北），则 data 的第一行已是北侧，无需 flip。
        netcdf_first_is_north = lats_netcdf[0] > lats_netcdf[-1]

        # 使用北上角（max lat）作为 origin（from_origin expects west, north, xsize, ysize）
        transform = from_origin(lon_min, lat_max, x_res, y_res)

        base = prefix or result['datetime'].strftime("lst_%Y%m%d_%H")
        fill_value = np.float32(-9999.0)

        def _write_tif(arr, path, dtype='float32', nodata=fill_value, flip_for_north=True):
            # 确保数组为 (nrows, ncols)
            a = np.array(arr, copy=False)
            if a.shape != (nrows, ncols):
                logger.warning(f"写 GeoTIFF: 数组形状 {a.shape} 与期望 { (nrows, ncols) } 不匹配，尝试 reshape/裁剪")
                a = a.reshape(nrows, ncols)[:nrows, :ncols]

            # 根据 netCDF lat 顺序判断是否需要 flip。
            # 如果 netcdf_first_is_north == True，则 netCDF 行 0 已为北侧，不要 flip（flip_for_north=False）
            # 如果 netcdf_first_is_north == False，则 netCDF 行 0 为南侧，需要 flip 使 TIFF 第一行为北
            if flip_for_north:
                a_out = np.flipud(a) if not netcdf_first_is_north else a
            else:
                a_out = np.flipud(a)

            profile = {
                'driver': 'GTiff',
                'height': nrows,
                'width': ncols,
                'count': 1,
                'dtype': dtype,
                'crs': 'EPSG:4326',
                'transform': transform,
                'nodata': nodata,
                'compress': 'lzw'
            }
            with rasterio.Env():
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(a_out.astype(dtype), 1)
            logger.info(f"  写出 TIFF: {os.path.basename(path)} (shape={a.shape}, flipped={'yes' if (not netcdf_first_is_north) else 'no'})")

        saved = []
        try:
            # LST
            lst = result['lst'].astype(np.float32)
            lst_filled = np.where(np.isnan(lst), fill_value, lst)
            lst_path = os.path.join(output_dir, f"{base}_LST.tif")
            _write_tif(lst_filled, lst_path, dtype='float32', nodata=fill_value)
            saved.append(lst_path)
            logger.info(f"✓ 已保存 GeoTIFF: {lst_path}")

            # LST_global (若存在)
            if 'lst_global' in result and result.get('lst_global') is not None:
                global_arr = result['lst_global'].astype(np.float32)
                global_filled = np.where(np.isnan(global_arr), fill_value, global_arr)
                gpath = os.path.join(output_dir, f"{base}_LST_global.tif")
                _write_tif(global_filled, gpath, dtype='float32', nodata=fill_value)
                saved.append(gpath)
                logger.info(f"✓ 已保存 GeoTIFF (global): {gpath}")

            # QC (uint8)
            if 'qc' in result and result.get('qc') is not None:
                qc = result['qc'].astype(np.uint8)
                # nodata for QC use 255
                qc_path = os.path.join(output_dir, f"{base}_QC.tif")
                _write_tif(qc, qc_path, dtype='uint8', nodata=255)
                saved.append(qc_path)
                logger.info(f"✓ 已保存 GeoTIFF (QC): {qc_path}")

        except Exception as e:
            logger.error(f"✗ 写出 GeoTIFF 失败: {e}")
            return []

        # 诊断信息，帮助确认方向
        #logger.info(f"GeoTIFF transform: origin_lon={lon_min}, origin_lat={lat_max}, x_res={x_res}, y_res={y_res}")
        #logger.info(f"NetCDF lat 顺序诊断: first_lat={lats_netcdf[0]:.6f}, last_lat={lats_netcdf[-1]:.6f}, first_is_north={netcdf_first_is_north}")
        return saved
    
    def evaluate_and_plot(self, result, output_dir):
        """评估并可视化结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        lst_final = result['lst']
        lst_obs = result['lst_original']
        lst_pred = result['lst_predicted']
        pixel_classes = result['pixel_classes']
        
        # 评估（仅对有观测值的像元）
        type1_indices = pixel_classes['type1_indices']
        
        metrics = None
        if len(type1_indices) > 0:
            obs_true = lst_obs.flat[type1_indices]
            obs_pred = lst_pred.flat[type1_indices]
            
            valid_eval = ~np.isnan(obs_pred)
            if np.sum(valid_eval) > 0:
                obs_true = obs_true[valid_eval]
                obs_pred = obs_pred[valid_eval]
                
                metrics = {
                    'rmse': float(np.sqrt(mean_squared_error(obs_true, obs_pred))),
                    'mae': float(mean_absolute_error(obs_true, obs_pred)),
                    'r2': float(r2_score(obs_true, obs_pred)),
                    'bias': float(np.mean(obs_pred - obs_true)),
                    'n_points': int(len(obs_true))
                }
        
        # 绘图
        self._plot_results(result, metrics, output_dir)
        
        return metrics
    
    def _plot_results(self, result, metrics, output_dir):
        """绘制详细结果图"""
        fig = plt.figure(figsize=(24, 16))
        
        dt_str = result['datetime'].strftime('%Y%m%d_%H')
        vmin, vmax = 220, 340
        
        # 创建2x3网格
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. 原始MODIS LST
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(result['lst_original'], cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax1.set_title('原始MODIS LST', fontsize=14, fontweight='bold')
        ax1.set_xlabel('列')
        ax1.set_ylabel('行')
        plt.colorbar(im1, ax=ax1, label='K')
        
        original_valid = np.sum(~np.isnan(result['lst_original']))
        ax1.text(0.02, 0.98, f'有效像元: {original_valid}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. 模型预测LST（仅第2类）
        ax2 = fig.add_subplot(gs[0, 1])
        
        # 创建只显示第2类像元的预测图
        type2_display = np.full_like(result['lst'], np.nan)
        type2_indices = result['pixel_classes']['type2_indices']
        type2_display.flat[type2_indices] = result['lst_predicted'].flat[type2_indices]
        
        im2 = ax2.imshow(type2_display, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax2.set_title('模型预测LST（仅第2类像元）', fontsize=14, fontweight='bold')
        ax2.set_xlabel('列')
        ax2.set_ylabel('行')
        plt.colorbar(im2, ax=ax2, label='K')
        
        ax2.text(0.02, 0.98, f'预测像元: {len(type2_indices)}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 3. 最终插值结果
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(result['lst'], cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax3.set_title('最终插值LST', fontsize=14, fontweight='bold')
        ax3.set_xlabel('列')
        ax3.set_ylabel('行')
        plt.colorbar(im3, ax=ax3, label='K')
        
        final_valid = np.sum(~np.isnan(result['lst']))
        ax3.text(0.02, 0.98, f'有效像元: {final_valid}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 4. 像元分类图
        ax4 = fig.add_subplot(gs[1, 0])
        
        # 创建分类图（不同颜色代表不同类别）
        classification_map = np.zeros_like(result['lst'])
        classification_map.flat[result['pixel_classes']['type1_indices']] = 1  # 原始LST
        classification_map.flat[result['pixel_classes']['type2_indices']] = 2  # 模型预测
        classification_map.flat[result['pixel_classes']['type3_indices']] = 0  # nodata
        classification_map.flat[result['pixel_classes']['type4_indices']] = 3  # 空间插值
        
        # 将nodata设为NaN以使用特殊颜色
        classification_map = np.where(classification_map == 0, np.nan, classification_map)
        
        im4 = ax4.imshow(classification_map, cmap='tab10', vmin=0, vmax=4)
        ax4.set_title('像元分类图', fontsize=14, fontweight='bold')
        ax4.set_xlabel('列')
        ax4.set_ylabel('行')
        
        # 自定义colorbar
        cbar4 = plt.colorbar(im4, ax=ax4)
        cbar4.set_ticks([1, 2, 3])
        cbar4.set_ticklabels(['第1类\n(原始)', '第2类\n(预测)', '第4类\n(插值)'])
        
        stats_text = (f"第1类: {result['pixel_classes']['type1_count']}\n"
                     f"第2类: {result['pixel_classes']['type2_count']}\n"
                     f"第3类: {result['pixel_classes']['type3_count']}\n"
                     f"第4类: {result['pixel_classes']['type4_count']}")
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # 5. 散点图：预测 vs 观测
        ax5 = fig.add_subplot(gs[1, 1])
        
        if metrics and metrics['n_points'] > 0:
            type1_indices = result['pixel_classes']['type1_indices']
            obs = result['lst_original'].flat[type1_indices]
            pred = result['lst_predicted'].flat[type1_indices]
            
            valid = ~np.isnan(obs) & ~np.isnan(pred)
            obs = obs[valid]
            pred = pred[valid]
                        
            ax5.scatter(obs, pred, alpha=0.5, s=2, c='blue')
            ax5.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2, label='1:1')
            ax5.set_xlabel('观测LST (K)', fontsize=12)
            ax5.set_ylabel('预测LST (K)', fontsize=12)
            ax5.set_title('预测 vs 观测', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            text = (f"RMSE: {metrics['rmse']:.4f} K\n"
                   f"MAE: {metrics['mae']:.4f} K\n"
                   f"$R^2$: {metrics['r2']:.4f}\n"
                   f"Bias: {metrics['bias']:.4f} K\n"
                   f"N: {metrics['n_points']}")
            ax5.text(0.05, 0.95, text, transform=ax5.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        else:
            ax5.text(0.5, 0.5, '无观测点评估', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=14)
        
        # 6. 残差分布
        ax6 = fig.add_subplot(gs[1, 2])
        
        if metrics and metrics['n_points'] > 0:
            type1_indices = result['pixel_classes']['type1_indices']
            obs = result['lst_original'].flat[type1_indices]
            pred = result['lst_predicted'].flat[type1_indices]
            
            valid = ~np.isnan(obs) & ~np.isnan(pred)
            residuals = pred[valid] - obs[valid]
            
            ax6.hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
            ax6.axvline(0, color='r', linestyle='--', linewidth=2, label='零线')
            ax6.axvline(np.mean(residuals), color='orange', linestyle='--', 
                       linewidth=2, label=f'均值={np.mean(residuals):.2f}K')
            ax6.set_xlabel('残差 (预测 - 观测, K)', fontsize=12)
            ax6.set_ylabel('频数', fontsize=12)
            ax6.set_title('残差分布', fontsize=14, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, '无残差数据', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=14)
        
        # 保存图片
        plot_path = os.path.join(output_dir, f'lst_interpolation_{dt_str}.png')
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 结果图已保存: {plot_path}")
    
def main():
    """主函数"""
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    local_finetune_dir = r"G:\CNN_SpatialDownscaling\output\local_finetune"
    
    try:
        interpolator = LSTInterpolator(config_path, local_finetune_dir)
        
        # 测试参数
        test_date = datetime(2018, 1, 1).date()

        for test_hour in range(24):
            dt_str = test_date.strftime("%Y%m%d") + f"_{test_hour:02d}"
        
            logger.info(f"开始测试插值: {test_date} {test_hour:02d}:00")
            
            # 执行插值
            result = interpolator.interpolate_hourly(test_date, test_hour)
            
            # 保存结果
            output_dir = os.path.join(os.path.dirname(config_path), '../output/interpolation_v8')
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存NetCDF
            nc_path = os.path.join(output_dir, f'lst_{test_date.strftime("%Y%m%d")}_{test_hour:02d}.nc')
            interpolator.save_netcdf(result, nc_path)
            
            # 保存GeoTIFF
            gtiff_paths = interpolator.save_geotiff(result, output_dir, prefix=dt_str)
            
            # 评估并绘图
            metrics = interpolator.evaluate_and_plot(result, output_dir)
            
            # 输出最终总结
            logger.info("【插值任务完成】")
            logger.info(f"NetCDF文件: {nc_path}")
            logger.info(f"GeoTIFF文件: {gtiff_paths}")
            
            if metrics:
                logger.info(f"\n模型预测精度:")
                logger.info(f"  RMSE: {metrics['rmse']:.4f} K")
                logger.info(f"  MAE: {metrics['mae']:.4f} K")
                logger.info(f"  R²: {metrics['r2']:.4f}")
                logger.info(f"  Bias: {metrics['bias']:.4f} K")
                logger.info(f"  评估点数: {metrics['n_points']}")
        
        
    except Exception as e:
        logger.error(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()