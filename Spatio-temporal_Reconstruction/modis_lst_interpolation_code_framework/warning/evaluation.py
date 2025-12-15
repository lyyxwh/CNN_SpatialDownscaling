import os
import json
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LSTInterpolationEvaluator:
    """LST插补结果综合评估器"""
    
    def __init__(self, interpolated_dir, evaluation_output_dir, config_path=None):
        """
        初始化评估器
        
        Args:
            interpolated_dir (str): 插补结果NetCDF文件目录
            evaluation_output_dir (str): 评估结果输出目录
            config_path (str): 配置文件路径（可选）
        """
        self.interpolated_dir = interpolated_dir
        self.evaluation_output_dir = evaluation_output_dir
        self.config_path = config_path
        
        # 创建输出目录结构
        self.metrics_dir = os.path.join(evaluation_output_dir, 'metrics')
        self.plots_dir = os.path.join(evaluation_output_dir, 'plots')
        self.spatial_analysis_dir = os.path.join(evaluation_output_dir, 'spatial_analysis')
        self.temporal_analysis_dir = os.path.join(evaluation_output_dir, 'temporal_analysis')
        
        for dir_path in [self.metrics_dir, self.plots_dir, self.spatial_analysis_dir, self.temporal_analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = None
        
        # 初始化统计数据存储
        self.daily_metrics = []
        self.hourly_metrics = []
        self.spatial_metrics = []
        
        logger.info(f"评估器初始化完成，输出目录: {evaluation_output_dir}")

    def get_interpolation_files(self, start_date=None, end_date=None):
        """
        获取指定时间范围内的插补文件列表
        
        Args:
            start_date (str): 开始日期 'YYYY-MM-DD'
            end_date (str): 结束日期 'YYYY-MM-DD'
            
        Returns:
            list: 文件路径列表
        """
        files = []
        for file_path in Path(self.interpolated_dir).glob("lst_interpolated_*.nc"):
            # 从文件名解析日期和小时
            filename = file_path.name
            try:
                # 假设文件名格式为 lst_interpolated_YYYYMMDD_HHH.nc
                date_hour = filename.replace('lst_interpolated_', '').replace('.nc', '')
                if '_H' in date_hour:
                    date_str, hour_str = date_hour.split('_H')
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    # 检查日期范围
                    if start_date and file_date < datetime.strptime(start_date, '%Y-%m-%d'):
                        continue
                    if end_date and file_date > datetime.strptime(end_date, '%Y-%m-%d'):
                        continue
                        
                    files.append(str(file_path))
            except:
                logger.warning(f"无法解析文件名: {filename}")
                continue
        
        files.sort()
        logger.info(f"找到 {len(files)} 个插补结果文件")
        return files

    def evaluate_single_file(self, file_path):
        """
        评估单个NetCDF文件
        
        Args:
            file_path (str): NetCDF文件路径
            
        Returns:
            dict: 评估指标字典
        """
        try:
            with xr.open_dataset(file_path) as ds:
                # 提取数据
                lst_interpolated = ds['lst_interpolated'].values
                original_modis = ds['original_modis'].values
                mask = ds['mask'].values.astype(bool)
                
                # 获取时间信息
                filename = os.path.basename(file_path)
                date_hour = filename.replace('lst_interpolated_', '').replace('.nc', '')
                date_str, hour_str = date_hour.split('_H')
                eval_date = datetime.strptime(date_str, '%Y%m%d')
                eval_hour = int(hour_str)
                
                # 计算基本统计
                total_pixels = lst_interpolated.size
                valid_modis_pixels = np.sum(mask)
                interpolated_pixels = np.sum(~np.isnan(lst_interpolated))
                coverage_improvement = (interpolated_pixels - valid_modis_pixels) / total_pixels * 100
                
                metrics = {
                    'file_path': file_path,
                    'date': eval_date,
                    'hour': eval_hour,
                    'total_pixels': total_pixels,
                    'valid_modis_pixels': valid_modis_pixels,
                    'interpolated_pixels': interpolated_pixels,
                    'modis_coverage': valid_modis_pixels / total_pixels * 100,
                    'final_coverage': interpolated_pixels / total_pixels * 100,
                    'coverage_improvement': coverage_improvement
                }
                
                # 对于有有效MODIS数据的区域，计算预测精度
                if valid_modis_pixels > 0:
                    # 如果有预测数据，计算与原始MODIS的对比
                    if 'lst_local_predictions' in ds:
                        local_predictions = ds['lst_local_predictions'].values
                        
                        # 只在有效MODIS数据的位置计算误差
                        valid_mask = mask & ~np.isnan(local_predictions)
                        if np.sum(valid_mask) > 0:
                            modis_valid = original_modis[valid_mask]
                            pred_valid = local_predictions[valid_mask]
                            
                            metrics.update({
                                'rmse': np.sqrt(mean_squared_error(modis_valid, pred_valid)),
                                'mae': mean_absolute_error(modis_valid, pred_valid),
                                'r2': r2_score(modis_valid, pred_valid),
                                'bias': np.mean(pred_valid - modis_valid),
                                'std_error': np.std(pred_valid - modis_valid),
                                'correlation': np.corrcoef(modis_valid, pred_valid)[0, 1]
                            })
                
                # 空间统计
                if not np.all(np.isnan(lst_interpolated)):
                    valid_interpolated = lst_interpolated[~np.isnan(lst_interpolated)]
                    metrics.update({
                        'lst_mean': np.mean(valid_interpolated),
                        'lst_std': np.std(valid_interpolated),
                        'lst_min': np.min(valid_interpolated),
                        'lst_max': np.max(valid_interpolated),
                        'lst_median': np.median(valid_interpolated)
                    })
                
                return metrics
                
        except Exception as e:
            logger.error(f"评估文件 {file_path} 时出错: {str(e)}")
            return None

    def batch_evaluate_files(self, file_list):
        """
        批量评估文件列表
        
        Args:
            file_list (list): 文件路径列表
            
        Returns:
            pd.DataFrame: 评估结果数据框
        """
        logger.info(f"开始批量评估 {len(file_list)} 个文件...")
        
        results = []
        for i, file_path in enumerate(file_list):
            if i % 100 == 0:
                logger.info(f"已处理 {i}/{len(file_list)} 个文件")
            
            metrics = self.evaluate_single_file(file_path)
            if metrics:
                results.append(metrics)
        
        if results:
            df = pd.DataFrame(results)
            logger.info(f"成功评估 {len(df)} 个文件")
            return df
        else:
            logger.error("没有成功评估的文件")
            return pd.DataFrame()

    def analyze_temporal_patterns(self, metrics_df):
        """分析时间模式"""
        logger.info("分析时间模式...")
        
        # 按小时统计
        hourly_stats = metrics_df.groupby('hour').agg({
            'modis_coverage': ['mean', 'std'],
            'final_coverage': ['mean', 'std'],
            'coverage_improvement': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'r2': ['mean', 'std']
        }).round(3)
        
        # 按月统计
        metrics_df['month'] = metrics_df['date'].dt.month
        monthly_stats = metrics_df.groupby('month').agg({
            'modis_coverage': ['mean', 'std'],
            'final_coverage': ['mean', 'std'],
            'coverage_improvement': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'r2': ['mean', 'std']
        }).round(3)
        
        # 保存统计结果
        hourly_stats.to_csv(os.path.join(self.temporal_analysis_dir, 'hourly_statistics.csv'))
        monthly_stats.to_csv(os.path.join(self.temporal_analysis_dir, 'monthly_statistics.csv'))
        
        # 绘制时间模式图
        self.plot_temporal_patterns(metrics_df)
        
        return hourly_stats, monthly_stats

    def plot_temporal_patterns(self, metrics_df):
        """绘制时间模式图表"""
        # 1. 小时变化模式
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 覆盖率变化
        hourly_coverage = metrics_df.groupby('hour')['final_coverage'].mean()
        axes[0, 0].plot(hourly_coverage.index, hourly_coverage.values, 'b-', linewidth=2)
        axes[0, 0].set_title('日内覆盖率变化')
        axes[0, 0].set_xlabel('小时')
        axes[0, 0].set_ylabel('覆盖率 (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE变化
        if 'rmse' in metrics_df.columns:
            hourly_rmse = metrics_df.groupby('hour')['rmse'].mean()
            axes[0, 1].plot(hourly_rmse.index, hourly_rmse.values, 'r-', linewidth=2)
            axes[0, 1].set_title('日内RMSE变化')
            axes[0, 1].set_xlabel('小时')
            axes[0, 1].set_ylabel('RMSE (K)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # R²变化
        if 'r2' in metrics_df.columns:
            hourly_r2 = metrics_df.groupby('hour')['r2'].mean()
            axes[1, 0].plot(hourly_r2.index, hourly_r2.values, 'g-', linewidth=2)
            axes[1, 0].set_title('日内R²变化')
            axes[1, 0].set_xlabel('小时')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 覆盖率改善
        hourly_improvement = metrics_df.groupby('hour')['coverage_improvement'].mean()
        axes[1, 1].plot(hourly_improvement.index, hourly_improvement.values, 'm-', linewidth=2)
        axes[1, 1].set_title('日内覆盖率改善')
        axes[1, 1].set_xlabel('小时')
        axes[1, 1].set_ylabel('覆盖率改善 (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.temporal_analysis_dir, 'hourly_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 月度变化模式
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        monthly_coverage = metrics_df.groupby('month')['final_coverage'].mean()
        axes[0, 0].bar(monthly_coverage.index, monthly_coverage.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('月度覆盖率变化')
        axes[0, 0].set_xlabel('月份')
        axes[0, 0].set_ylabel('覆盖率 (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        if 'rmse' in metrics_df.columns:
            monthly_rmse = metrics_df.groupby('month')['rmse'].mean()
            axes[0, 1].bar(monthly_rmse.index, monthly_rmse.values, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('月度RMSE变化')
            axes[0, 1].set_xlabel('月份')
            axes[0, 1].set_ylabel('RMSE (K)')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'r2' in metrics_df.columns:
            monthly_r2 = metrics_df.groupby('month')['r2'].mean()
            axes[1, 0].bar(monthly_r2.index, monthly_r2.values, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('月度R²变化')
            axes[1, 0].set_xlabel('月份')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].grid(True, alpha=0.3)
        
        monthly_improvement = metrics_df.groupby('month')['coverage_improvement'].mean()
        axes[1, 1].bar(monthly_improvement.index, monthly_improvement.values, color='plum', alpha=0.7)
        axes[1, 1].set_title('月度覆盖率改善')
        axes[1, 1].set_xlabel('月份')
        axes[1, 1].set_ylabel('覆盖率改善 (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.temporal_analysis_dir, 'monthly_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_spatial_patterns(self, file_list, sample_size=50):
        """分析空间模式（采样分析以节省内存）"""
        logger.info(f"分析空间模式（采样 {sample_size} 个文件）...")
        
        # 随机采样文件
        import random
        sampled_files = random.sample(file_list, min(sample_size, len(file_list)))
        
        spatial_stats = []
        
        for file_path in sampled_files:
            try:
                with xr.open_dataset(file_path) as ds:
                    lst_interpolated = ds['lst_interpolated'].values
                    original_modis = ds['original_modis'].values
                    mask = ds['mask'].values.astype(bool)
                    
                    # 计算空间自相关（简化版）
                    if not np.all(np.isnan(lst_interpolated)):
                        # 计算梯度作为空间变化的指标
                        grad_y, grad_x = np.gradient(lst_interpolated)
                        spatial_gradient = np.sqrt(grad_y**2 + grad_x**2)
                        
                        spatial_stats.append({
                            'file_path': file_path,
                            'spatial_gradient_mean': np.nanmean(spatial_gradient),
                            'spatial_gradient_std': np.nanstd(spatial_gradient),
                            'lst_spatial_std': np.nanstd(lst_interpolated)
                        })
            except Exception as e:
                logger.warning(f"空间分析失败: {file_path} - {str(e)}")
                continue
        
        if spatial_stats:
            spatial_df = pd.DataFrame(spatial_stats)
            spatial_df.to_csv(os.path.join(self.spatial_analysis_dir, 'spatial_statistics.csv'), index=False)
            return spatial_df
        else:
            return pd.DataFrame()

    def create_validation_plots(self, metrics_df):
        """创建验证图表"""
        logger.info("创建验证图表...")
        
        # 1. 整体精度散点图
        if 'rmse' in metrics_df.columns and 'r2' in metrics_df.columns:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # RMSE分布
            axes[0].hist(metrics_df['rmse'].dropna(), bins=30, alpha=0.7, color='skyblue')
            axes[0].set_title('RMSE分布')
            axes[0].set_xlabel('RMSE (K)')
            axes[0].set_ylabel('频次')
            axes[0].axvline(metrics_df['rmse'].mean(), color='red', linestyle='--', 
                          label=f'均值: {metrics_df["rmse"].mean():.2f}K')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # R²分布
            axes[1].hist(metrics_df['r2'].dropna(), bins=30, alpha=0.7, color='lightgreen')
            axes[1].set_title('R²分布')
            axes[1].set_xlabel('R²')
            axes[1].set_ylabel('频次')
            axes[1].axvline(metrics_df['r2'].mean(), color='red', linestyle='--',
                          label=f'均值: {metrics_df["r2"].mean():.3f}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 覆盖率改善分布
            axes[2].hist(metrics_df['coverage_improvement'].dropna(), bins=30, alpha=0.7, color='orange')
            axes[2].set_title('覆盖率改善分布')
            axes[2].set_xlabel('覆盖率改善 (%)')
            axes[2].set_ylabel('频次')
            axes[2].axvline(metrics_df['coverage_improvement'].mean(), color='red', linestyle='--',
                          label=f'均值: {metrics_df["coverage_improvement"].mean():.1f}%')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'accuracy_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def generate_summary_report(self, metrics_df, hourly_stats, monthly_stats):
        """生成总结报告"""
        logger.info("生成总结报告...")
        
        # 计算总体统计
        total_files = len(metrics_df)
        avg_coverage_improvement = metrics_df['coverage_improvement'].mean()
        avg_final_coverage = metrics_df['final_coverage'].mean()
        avg_modis_coverage = metrics_df['modis_coverage'].mean()
        
        summary = {
            'evaluation_date': datetime.now().isoformat(),
            'total_files_evaluated': total_files,
            'time_range': {
                'start_date': metrics_df['date'].min().isoformat() if not metrics_df.empty else None,
                'end_date': metrics_df['date'].max().isoformat() if not metrics_df.empty else None
            },
            'overall_performance': {
                'average_modis_coverage_percent': round(avg_modis_coverage, 2),
                'average_final_coverage_percent': round(avg_final_coverage, 2),
                'average_coverage_improvement_percent': round(avg_coverage_improvement, 2)
            }
        }
        
        # 添加精度指标（如果有）
        if 'rmse' in metrics_df.columns:
            valid_accuracy = metrics_df.dropna(subset=['rmse', 'mae', 'r2'])
            if not valid_accuracy.empty:
                summary['accuracy_metrics'] = {
                    'rmse_mean': round(valid_accuracy['rmse'].mean(), 3),
                    'rmse_std': round(valid_accuracy['rmse'].std(), 3),
                    'mae_mean': round(valid_accuracy['mae'].mean(), 3),
                    'mae_std': round(valid_accuracy['mae'].std(), 3),
                    'r2_mean': round(valid_accuracy['r2'].mean(), 4),
                    'r2_std': round(valid_accuracy['r2'].std(), 4),
                    'valid_samples': len(valid_accuracy)
                }
        
        # 最佳和最差表现
        if 'rmse' in metrics_df.columns:
            best_rmse_idx = metrics_df['rmse'].idxmin()
            worst_rmse_idx = metrics_df['rmse'].idxmax()
            
            summary['performance_extremes'] = {
                'best_accuracy': {
                    'file': metrics_df.loc[best_rmse_idx, 'file_path'],
                    'date': metrics_df.loc[best_rmse_idx, 'date'].isoformat(),
                    'hour': int(metrics_df.loc[best_rmse_idx, 'hour']),
                    'rmse': round(metrics_df.loc[best_rmse_idx, 'rmse'], 3),
                    'r2': round(metrics_df.loc[best_rmse_idx, 'r2'], 4)
                },
                'worst_accuracy': {
                    'file': metrics_df.loc[worst_rmse_idx, 'file_path'],
                    'date': metrics_df.loc[worst_rmse_idx, 'date'].isoformat(),
                    'hour': int(metrics_df.loc[worst_rmse_idx, 'hour']),
                    'rmse': round(metrics_df.loc[worst_rmse_idx, 'rmse'], 3),
                    'r2': round(metrics_df.loc[worst_rmse_idx, 'r2'], 4)
                }
            }
        
        # 保存报告
        report_file = os.path.join(self.evaluation_output_dir, 'evaluation_summary.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存详细指标
        metrics_file = os.path.join(self.metrics_dir, 'detailed_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        
        logger.info(f"总结报告已保存: {report_file}")
        logger.info(f"详细指标已保存: {metrics_file}")
        
        return summary

    def run_comprehensive_evaluation(self, start_date=None, end_date=None):
        """运行综合评估"""
        logger.info("开始综合评估...")
        
        # 获取文件列表
        file_list = self.get_interpolation_files(start_date, end_date)
        if not file_list:
            logger.error("未找到插补结果文件")
            return None
        
        # 批量评估
        metrics_df = self.batch_evaluate_files(file_list)
        if metrics_df.empty:
            logger.error("评估失败，无有效结果")
            return None
        
        # 时间模式分析
        hourly_stats, monthly_stats = self.analyze_temporal_patterns(metrics_df)
        
        # 空间模式分析
        spatial_df = self.analyze_spatial_patterns(file_list)
        
        # 创建验证图表
        self.create_validation_plots(metrics_df)
        
        # 生成总结报告
        summary = self.generate_summary_report(metrics_df, hourly_stats, monthly_stats)
        
        logger.info("综合评估完成!")
        logger.info(f"总体覆盖率改善: {summary['overall_performance']['average_coverage_improvement_percent']:.1f}%")
        if 'accuracy_metrics' in summary:
            logger.info(f"平均RMSE: {summary['accuracy_metrics']['rmse_mean']:.3f}K")
            logger.info(f"平均R²: {summary['accuracy_metrics']['r2_mean']:.4f}")
        
        return summary

def main():
    """主函数 - 评估示例"""
    # 配置路径 - 根据您的实际路径修改
    interpolated_dir = r"G:\CNN_SpatialDownscaling\output\batch_interpolation\interpolated_results"
    evaluation_output_dir = r"G:\CNN_SpatialDownscaling\output\evaluation_results" 
    config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
    
    # 创建评估器
    evaluator = LSTInterpolationEvaluator(
        interpolated_dir=interpolated_dir,
        evaluation_output_dir=evaluation_output_dir,
        config_path=config_path
    )
    
    # 运行综合评估
    # 选项1: 评估全年
    summary = evaluator.run_comprehensive_evaluation()
    
    # 选项2: 评估特定时间段
    # summary = evaluator.run_comprehensive_evaluation(start_date='2018-01-01', end_date='2018-01-31')
    
    if summary:
        print("评估完成！主要结果:")
        print(f"- 处理文件数: {summary['total_files_evaluated']}")
        print(f"- 平均覆盖率改善: {summary['overall_performance']['average_coverage_improvement_percent']:.1f}%")
        print(f"- 最终平均覆盖率: {summary['overall_performance']['average_final_coverage_percent']:.1f}%")

if __name__ == "__main__":
    main()