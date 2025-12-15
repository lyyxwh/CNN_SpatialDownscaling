import os
import xarray as xr
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, config):
        """初始化可视化器
        
        Args:
            config (dict): 配置文件内容
        """
        self.config = config
        logger.info("可视化器初始化完成")

    def visualize(self, target_date):
        """生成预测 vs 真实 LST 的可视化
        
        Args:
            target_date (datetime): 目标日期
        """
        ds = xr.open_dataset(f"{self.config['output_dir']}/lst_interpolated_{target_date.strftime('%Y%m%d')}.nc")
        pred = ds['lst'].values
        true = ds['lst'].where(ds['mask']).values

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Predicted LST {target_date.strftime('%Y%m%d')}")
        plt.imshow(pred, cmap='jet')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title(f"True LST {target_date.strftime('%Y%m%d')}")
        plt.imshow(true, cmap='jet')
        plt.colorbar()
        os.makedirs(self.config['output_dir'], exist_ok=True)
        plt.savefig(f"{self.config['output_dir']}/lst_pred_{target_date.strftime('%Y%m%d')}.png")
        plt.close()
        logger.info(f"可视化完成，日期: {target_date}, 保存图像: lst_pred_{target_date.strftime('%Y%m%d')}.png")

if __name__ == "__main__":
    # 单独调试：使用 2018-01-15 实际数据进行可视化
    logger.info("开始调试 Visualizer 模块，使用实际数据")
    import json
    config = json.load(open("config.json"))
    visualizer = Visualizer(config)
    target_date = datetime(2018, 1, 15)
    
    # 假设插补结果已生成（需先运行 interpolate.py 调试）
    try:
        visualizer.visualize(target_date)
        logger.info("可视化调试完成")
    except FileNotFoundError:
        logger.warning("插补结果文件未找到，请先运行 interpolate.py 调试生成 NetCDF 文件")