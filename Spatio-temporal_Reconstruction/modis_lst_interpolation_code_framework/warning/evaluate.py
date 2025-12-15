
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import logging
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config):
        """初始化评估器
        
        Args:
            config (dict): 配置文件内容
        """
        self.config = config
        logger.info("评估器初始化完成")

    def evaluate(self, pred, true, mask):
        """计算评估指标
        
        Args:
            pred (ndarray): 预测值
            true (ndarray): 真实值
            mask (ndarray): 有效数据掩码
        
        Returns:
            dict: 评估指标（R2, RMSE, MAE, SpatialCorr）
        """
        pred = pred[mask]
        true = true[mask]
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        spatial_corr, _ = pearsonr(true.flatten(), pred.flatten())
        metrics = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'SpatialCorr': spatial_corr}
        logger.info(f"评估结果: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, SpatialCorr={spatial_corr:.4f}")
        return metrics

    def validate(self, target_date):
        """验证插补结果
        
        Args:
            target_date (datetime): 目标日期
        
        Returns:
            dict: 评估指标
        """
        ds = xr.open_dataset(f"{self.config['output_dir']}/lst_interpolated_{target_date.strftime('%Y%m%d')}.nc")
        pred = ds['lst'].values
        true = ds['lst'].where(ds['mask']).values
        metrics = self.evaluate(pred, true, ds['mask'].values)
        
        # 若 MODIS 数据不足，使用上采样 CLDAS 验证
        if np.sum(ds['mask']) < 0.1 * np.prod(self.config['data_params']['local_shape']):
            logger.warning("MODIS 数据不足，使用上采样 CLDAS 验证")
            cldas_ds = xr.open_dataset(
                f"{self.config['global_data_paths']['cldas']}/CLDAS_{target_date.strftime('%Y%m%d_%H')}_00.nc")
            cldas_true = cldas_ds[self.config['var_names']['cldas']].values
            cldas_pred = ds['lst_global_upsampled'].values
            cldas_metrics = self.evaluate(cldas_pred, cldas_true, 
                                        ~np.isclose(cldas_true, self.config['nodata_values']['cldas']))
            metrics.update({f"CLDAS_{k}": v for k, v in cldas_metrics.items()})
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        with open(f"{self.config['output_dir']}/metrics_{target_date.strftime('%Y%m%d')}.json", 'w') as f:
            json.dump(metrics, f)
        logger.info(f"验证完成，日期: {target_date}, 保存评估结果")
        return metrics

if __name__ == "__main__":
    # 单独调试：使用 2018-01-15 实际数据进行评估
    logger.info("开始调试 Evaluator 模块，使用实际数据")
    import json
    config = json.load(open("config.json"))
    evaluator = Evaluator(config)
    target_date = datetime(2018, 1, 15)
    
    # 假设插补结果已生成（需先运行 interpolate.py 调试）
    try:
        metrics = evaluator.validate(target_date)
        logger.info(f"验证结果: {metrics}")
    except FileNotFoundError:
        logger.warning("插补结果文件未找到，请先运行 interpolate.py 调试生成 NetCDF 文件")