import os
import logging
from datetime import datetime
import json

# 导入所有模块
from train_global import GlobalPretrainer
from global_finetune import GlobalFineTuner
from train_local_finetune import LocalFineTuner
from lst_interpolation import PredictInterpolator

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main_pipeline_debug(target_date_str="2018-01-15"):
    """
    主调试函数，用于端到端运行整个LST插补流程。
    
    Args:
        target_date_str (str): 目标调试日期，格式 'YYYY-MM-DD'
    """
    logger.info("--- 开始 LST 时空插补主调试流程 ---")
    
    # 1. 路径配置
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    output_base_dir = r"G:\CNN_SpatialDownscaling\output"
    
    # 子目录
    pretrain_dir = os.path.join(output_base_dir, "global_pretrain")
    global_finetune_dir = os.path.join(output_base_dir, "global_finetune")
    local_finetune_dir = os.path.join(output_base_dir, "local_finetune")
    interpolated_dir = os.path.join(output_base_dir, "interpolated_results")

    # 确保所有目录存在
    os.makedirs(pretrain_dir, exist_ok=True)
    os.makedirs(global_finetune_dir, exist_ok=True)
    os.makedirs(local_finetune_dir, exist_ok=True)
    os.makedirs(interpolated_dir, exist_ok=True)

    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        
        # 2. 全局预训练 (仅调试模式)
        logger.info(f"--- 步骤1: 预训练 {target_date.strftime('%Y-%m')} 月的全局模型 (仅调试模式) ---")
        pretrain_model_path = os.path.join(pretrain_dir, f"global_pretrain_{target_date.strftime('%Y%m')}.pt")
        if not os.path.exists(pretrain_model_path):
            pretrainer = GlobalPretrainer(config_path, pretrain_dir)
            pretrainer.train_model(target_date, is_debug=True)
        else:
            logger.info("预训练模型已存在，跳过预训练步骤。")

        # 3. 全局微调
        logger.info(f"--- 步骤2: 全局微调 {target_date_str} 日的模型 ---")
        global_finetuner = GlobalFineTuner(config_path, pretrain_dir, global_finetune_dir)
        global_finetune_path = global_finetuner.fine_tune(target_date, is_debug=True)
        
        if not global_finetune_path:
            logger.error("全局微调失败，流程中止。")
            return

        # 4. 局部微调
        logger.info(f"--- 步骤3: 局部微调 {target_date_str} 日的模型 ---")
        local_finetuner = LocalFineTuner(config_path, global_finetune_dir, local_finetune_dir)
        local_finetune_path = local_finetuner.fine_tune(target_date, is_debug=True)
        
        if not local_finetune_path:
            logger.error("局部微调失败，流程中止。")
            return
            
        # 5. 预测插补
        logger.info(f"--- 步骤4: 插补 {target_date_str} 日的LST数据 ---")
        interpolator = PredictInterpolator(config_path, global_finetune_dir, local_finetune_dir, interpolated_dir)
        output_nc_path = interpolator.interpolate(target_date, is_debug=True)
        
        if output_nc_path:
            logger.info("--- 主调试流程成功完成！ ---")
            logger.info(f"最终结果文件保存在: {output_nc_path}")
        else:
            logger.error("插补预测失败。")

    except Exception as e:
        logger.error(f"主调试流程发生致命错误: {e}", exc_info=True)