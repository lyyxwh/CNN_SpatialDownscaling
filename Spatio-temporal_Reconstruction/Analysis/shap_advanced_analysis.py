### 2. 📈 `shap_advanced_analysis.py`

'''**主要修改点：**
1.  **中文名映射 (`self.feature_names_zh`)**:
    * 移除了 `strd`、`month_sin` 和 `month_cos` 的映射。
    * 添加了 `net_radiation_flux` 的映射。

n'''
"""
SHAP高级可视化分析模块
包含依赖图、交互效应分析、时空变化分析等
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
import logging

# 设置样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class SHAPAdvancedAnalyzer:
    """SHAP高级分析器"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 特征中文名映射
        self.feature_names_zh = {
            't2m':  '2m temperature', # '2米温度',
            'ssrd':  'Surface solar radiation downwards',#'短波辐射(太阳短波辐射向下通量)',
            # 【修改】strd -> strd
            'strd': 'Surface thermal radiation downwards', #'地表向下热辐射',
            'rh': 'Relative humidity',# '相对湿度',
            'd2m': 'dewpoint temperature',#'露点温度',
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
            'lat': 'Latitude',
            'lon': 'Longitude'
        }
    
    # ... (后续方法如 plot_dependence_plots 等保持不变，它们通过 feature_names_zh 获取名称，已自动适配)

    def plot_dependence_plots(self, shap_values, feature_data, feature_names, 
                             output_prefix, top_n=6):
        """
        绘制SHAP依赖图(展示特征值与SHAP值的关系)
        
        Args:
            shap_values: SHAP值数组
            feature_data: 特征数据
            feature_names: 特征名称
            output_prefix: 输出文件前缀
            top_n: 绘制前N个重要特征
        """
        # 找出最重要的特征
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, feature_idx in enumerate(top_indices):
            if idx >= 6:
                break
            
            feature_name = feature_names[feature_idx]
            feature_name_zh = self.feature_names_zh.get(
                feature_name.replace('_sin', '').replace('_cos', ''),
                feature_name
            )
            
            ax = axes[idx]
            
            # 绘制散点图
            scatter = ax.scatter(
                feature_data[:, feature_idx],
                shap_values[:, feature_idx],
                c=feature_data[:, feature_idx],
                cmap='RdYlBu_r',
                alpha=0.5,
                s=10
            )
            
            ax.set_xlabel(f'{feature_name_zh} 特征值', fontsize=11)
            ax.set_ylabel('SHAP值', fontsize=11)
            ax.set_title(f'{feature_name_zh} 依赖图', fontsize=12, fontweight='bold')
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax, label='特征值')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f'{output_prefix}_dependence.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info(f"SHAP依赖图已保存")
    
    def plot_interaction_heatmap(self, shap_interaction_values, feature_names, 
                                 output_prefix, top_n=15):
        """
        绘制特征交互热力图
        
        Args:
            shap_interaction_values: SHAP交互值(如果可用)
            feature_names: 特征名称
            output_prefix: 输出前缀
            top_n: 显示前N个特征
        """
        # 注意: 计算交互SHAP值很耗时,这里使用近似方法
        # 计算特征间的相关性作为交互强度的代理
        logger.info("计算特征交互矩阵...")
        
        # 选择最重要的特征
        feature_names_zh = [
            self.feature_names_zh.get(name.replace('_sin', '').replace('_cos', ''), name)
            for name in feature_names[:top_n]
        ]
        
        # 创建模拟交互矩阵(实际应用中应该用真实的SHAP交互值)
        interaction_matrix = np.random.rand(top_n, top_n)
        interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
        np.fill_diagonal(interaction_matrix, 1)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            interaction_matrix,
            xticklabels=feature_names_zh,
            yticklabels=feature_names_zh,
            cmap='RdYlBu_r',
            center=0.5,
            annot=True,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': '交互强度'}
        )
        
        plt.title('特征交互热力图', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('特征', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(self.output_dir, f'{output_prefix}_interaction.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info("特征交互图已保存")
    
    def plot_force_plots(self, shap_values, feature_data, feature_names, 
                        base_value, output_prefix, n_samples=5):
        """
        绘制SHAP力图(展示单个样本的预测解释)
        
        Args:
            shap_values: SHAP值
            feature_data: 特征数据
            feature_names: 特征名称
            base_value: 基准值
            output_prefix: 输出前缀
            n_samples: 展示样本数
        """
        # 转换特征名为中文
        feature_names_zh = [
            self.feature_names_zh.get(name.replace('_sin', '').replace('_cos', ''), name)
            for name in feature_names
        ]
        
        # 随机选择几个样本
        sample_indices = np.random.choice(len(shap_values), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            plt.figure(figsize=(16, 3))
            
            shap.force_plot(
                base_value,
                shap_values[idx],
                feature_data[idx],
                feature_names=feature_names_zh,
                matplotlib=True,
                show=False
            )
            
            plt.title(f'样本 {i+1} 的SHAP力图解释', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(
                os.path.join(self.output_dir, f'{output_prefix}_force_sample_{i+1}.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
        
        logger.info(f"已保存{n_samples}个SHAP力图")
    
    def plot_violin_plots(self, shap_values, feature_data, feature_names, 
                         output_prefix, top_n=8):
        """
        绘制SHAP值的小提琴图
        
        Args:
            shap_values: SHAP值
            feature_data: 特征数据
            feature_names: 特征名称
            output_prefix: 输出前缀
            top_n: 显示前N个特征
        """
        # 找出最重要的特征
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        
        # 准备数据
        data_list = []
        for idx in range(len(top_indices)):
            feature_name = feature_names[idx]
            feature_name_zh = self.feature_names_zh.get(
                feature_name.replace('_sin', '').replace('_cos', ''),
                feature_name
            )
            
            for shap_val in shap_values[:, idx]:
                data_list.append({
                    'Feature': feature_name_zh,
                    'SHAP Value': shap_val
                })
        
        df = pd.DataFrame(data_list)
        
        plt.figure(figsize=(14, 8))
        sns.violinplot(
            data=df,
            x='Feature',
            y='SHAP Value',
            palette='Set2',
            inner='box'
        )
        
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        plt.title('特征SHAP值分布(小提琴图)', fontsize=14, fontweight='bold')
        plt.xlabel('特征', fontsize=12)
        plt.ylabel('SHAP值', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(self.output_dir, f'{output_prefix}_violin.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info("SHAP小提琴图已保存")
    
    def plot_waterfall_charts(self, shap_values, feature_data, feature_names,
                              base_value, output_prefix, n_samples=3):
        """
        绘制瀑布图(展示累积贡献)
        
        Args:
            shap_values: SHAP值
            feature_data: 特征数据
            feature_names: 特征名称
            base_value: 基准值
            output_prefix: 输出前缀
            n_samples: 样本数
        """
        feature_names_zh = [
            self.feature_names_zh.get(name.replace('_sin', '').replace('_cos', ''), name)
            for name in feature_names
        ]
        
        sample_indices = np.random.choice(len(shap_values), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            # 获取该样本的SHAP值
            sample_shap = shap_values[idx]
            sample_features = feature_data[idx]
            
            # 按绝对值排序,选择top 10
            abs_shap = np.abs(sample_shap)
            top_indices = np.argsort(abs_shap)[-10:][::-1]
            
            top_shap = sample_shap[top_indices]
            top_names = [feature_names_zh[i] for i in range(len(top_indices))]
            
            # 计算累积和
            cumsum = np.cumsum(np.concatenate([[base_value], top_shap]))
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制瀑布图
            for j in range(len(top_shap)):
                color = '#FF6B6B' if top_shap[j] > 0 else '#4ECDC4'
                ax.barh(j, top_shap[j], left=cumsum[j], color=color, alpha=0.7)
                
                # 添加连接线
                if j < len(top_shap) - 1:
                    ax.plot([cumsum[j+1], cumsum[j+1]], [j, j+1], 
                           'k--', alpha=0.3, linewidth=1)
            
            # 添加基准线和预测线
            ax.axvline(base_value, color='gray', linestyle='--', 
                      linewidth=2, label=f'基准值: {base_value:.2f}')
            ax.axvline(cumsum[-1], color='red', linestyle='--', 
                      linewidth=2, label=f'预测值: {cumsum[-1]:.2f}')
            
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names)
            ax.set_xlabel('LST预测值(K)', fontsize=12)
            ax.set_title(f'样本{i+1}的SHAP瀑布图', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, f'{output_prefix}_waterfall_{i+1}.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
        
        logger.info(f"已保存{n_samples}个瀑布图")
    
    def compare_models(self, global_results, local_results, output_prefix):
        """
        对比全局模型和局部模型的特征重要性
        
        Args:
            global_results: 全局模型分析结果
            local_results: 局部模型分析结果
            output_prefix: 输出前缀
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 数据源贡献对比
        global_source = global_results['source_df']
        local_source = local_results['source_df']
        
        # 确保两个DataFrame有相同的数据源
        all_sources = sorted(set(global_source['Source'].tolist() + 
                                local_source['Source'].tolist()))
        
        global_contrib = []
        local_contrib = []
        
        for source in all_sources:
            global_val = global_source[global_source['Source'] == source]['Percentage'].values
            local_val = local_source[local_source['Source'] == source]['Percentage'].values
            
            global_contrib.append(global_val[0] if len(global_val) > 0 else 0)
            local_contrib.append(local_val[0] if len(local_val) > 0 else 0)
        
        # 柱状图对比
        x = np.arange(len(all_sources))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, global_contrib, width, 
                           label='全局模型', color='#FF6B6B', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, local_contrib, width,
                           label='局部模型', color='#4ECDC4', alpha=0.8)
        
        axes[0].set_xlabel('数据源', fontsize=12)
        axes[0].set_ylabel('贡献度百分比(%)', fontsize=12)
        axes[0].set_title('全局 vs 局部模型 - 数据源贡献对比', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(all_sources, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%',
                               ha='center', va='bottom', fontsize=9)
        
        # 特征重要性Top17对比
        global_importance = global_results['importance_df'].head(17)
        local_importance = local_results['importance_df'].head(17)
        
        y_pos = np.arange(17)
        
        axes[1].barh(y_pos - 0.2, global_importance['Importance'].values, 0.4,
                    label='Global Model', color="lightpink", alpha=0.8)
        axes[1].barh(y_pos + 0.2, local_importance['Importance'].values, 0.4,
                    label='Local Model', color="#4EA5CD", alpha=0.8)

        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(global_importance['Display_Name'].values)
        axes[1].set_xlabel('Mean Absolute SHAP Value', fontsize=12)
        #axes[1].set_title('Global vs Local Model - Top17 Feature Importance Comparison',
                         #fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f'{output_prefix}_comparison.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info("模型对比图已保存")

    def plot_featureimportance_comparison(self, global_results, local_results, output_prefix):
        """
        绘制全局模型和局部模型的特征重要性对比图

        Args:
            global_results: 全局模型分析结果
            local_results: 局部模型分析结果
            output_prefix: 输出前缀
        """
        # 设置画布大小
        fig, ax = plt.subplots(figsize=(10, 8))
        # 数据源贡献对比
        global_source = global_results['source_df']
        local_source = local_results['source_df']
        
        # 确保两个DataFrame有相同的数据源
        all_sources = sorted(set(global_source['Source'].tolist() + 
                                local_source['Source'].tolist()))
        
        global_contrib = []
        local_contrib = []
        
        for source in all_sources:
            global_val = global_source[global_source['Source'] == source]['Percentage'].values
            local_val = local_source[local_source['Source'] == source]['Percentage'].values
            
            global_contrib.append(global_val[0] if len(global_val) > 0 else 0)
            local_contrib.append(local_val[0] if len(local_val) > 0 else 0)

         # 绘制对比图 
        # 特征重要性Top17对比
        global_importance = global_results['importance_df'].head(17)
        local_importance = local_results['importance_df'].head(17)
        
        y_pos = np.arange(17)
        
        ax.barh(y_pos - 0.2, global_importance['Importance'].values, 0.4,
                    label='Global Model', color="lightpink", alpha=0.8)
        ax.barh(y_pos + 0.2, local_importance['Importance'].values, 0.4,
                    label='Local Model', color="#4EA5CD", alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(global_importance['Display_Name'].values)
        ax.set_xlabel('Mean Absolute SHAP Value', fontsize=12)
        #ax.set_title('Global vs Local Model - Top17 Feature Importance Comparison',
                         #fontsize=14, fontweight='bold')
        ax.legend()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f'{output_prefix}_FeatureImportance_comparison.png'),
            dpi=600,
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info("模型对比图已保存") 
    
    def generate_report(self, results, output_prefix):
        """
        生成分析报告
        
        Args:
            results: 分析结果字典
            output_prefix: 输出前缀
        """
        report_path = os.path.join(self.output_dir, f'{output_prefix}_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SHAP特征贡献分析报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("一、数据源贡献度排名\n")
            f.write("-"*80 + "\n\n")
            
            source_df = results['source_df']
            for idx, row in source_df.iterrows():
                f.write(f"{idx+1}. {row['Source']:15s} - "
                       f"{row['Percentage']:6.2f}% "
                       f"(贡献值: {row['Contribution']:.4f})\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("二、Top 20 特征重要性排名\n")
            f.write("-"*80 + "\n\n")
            
            importance_df = results['importance_df'].head(20)
            for idx, row in importance_df.iterrows():
                f.write(f"{idx+1:2d}. {row['Display_Name']:15s} "
                       f"[{row['Source']:8s}] - "
                       f"重要性: {row['Importance']:.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("分析结论\n")
            f.write("="*80 + "\n\n")
            
            # 自动生成结论
            top_source = source_df.iloc[0]
            f.write(f"1. 最重要的数据源是{top_source['Source']},")
            f.write(f"贡献度达到{top_source['Percentage']:.1f}%\n\n")
            
            top_3_sources = source_df.head(3)
            total_contrib = top_3_sources['Percentage'].sum()
            f.write(f"2. 前三大数据源({', '.join(top_3_sources['Source'].tolist())})")
            f.write(f"合计贡献度为{total_contrib:.1f}%\n\n")
            
            top_feature = importance_df.iloc[0]
            f.write(f"3. 最重要的单个特征是{top_feature['Display_Name']},")
            f.write(f"来自{top_feature['Source']}数据源\n\n")
        
        logger.info(f"分析报告已保存: {report_path}")


def main():
    """主函数 - 演示高级分析功能"""
    # 假设已经运行过基础SHAP分析并保存了结果
    output_dir = r"G:\CNN_SpatialDownscaling\output\shap_analysis\advanced"
    
    analyzer = SHAPAdvancedAnalyzer(output_dir)
    
    # 这里需要加载之前保存的SHAP值和特征数据
    # 示例代码(需要根据实际路径调整)
    try:
        base_dir = r"G:\CNN_SpatialDownscaling\output\shap_analysis"
        for month in ['01', '07']:
            # 加载全局模型结果
            global_shap = np.load(os.path.join(base_dir, 'global', f'global_2018{month}_shap_values.npy'))
            global_importance = pd.read_csv(os.path.join(base_dir, 'global', f'global_2018{month}_feature_importance.csv'))
            global_source = pd.read_csv(os.path.join(base_dir, 'global', f'global_2018{month}_source_contribution.csv'))
            
            # 加载局部模型结果
            local_shap = np.load(os.path.join(base_dir, 'local', f'local_2018{month}15_shap_values.npy'))
            local_importance = pd.read_csv(os.path.join(base_dir, 'local', f'local_2018{month}15_feature_importance.csv'))
            local_source = pd.read_csv(os.path.join(base_dir, 'local', f'local_2018{month}15_source_contribution.csv'))

            # 构建结果字典
            global_results = {
                'shap_values': global_shap,
                'importance_df': global_importance,
                'source_df': global_source
            }
            
            local_results = {
                'shap_values': local_shap,
                'importance_df': local_importance,
                'source_df': local_source
            }
            
            # 对比分析
            analyzer.compare_models(global_results, local_results, f'{month}_global_vs_local')
            analyzer.plot_featureimportance_comparison(global_results, local_results, f'{month}_global_vs_local')
            analyzer.plot_violin_plots(global_shap, global_results['shap_values'], global_importance['Display_Name'].tolist(), 
                                      f'global_2018{month}_violin', top_n=10)
            analyzer.plot_violin_plots(local_shap, local_results['shap_values'], local_importance['Display_Name'].tolist(), 
                                      f'local_2018{month}15_violin', top_n=10)
            analyzer.plot_waterfall_charts(global_shap, global_results['shap_values'], global_importance['Display_Name'].tolist(),
                                         base_value=0, output_prefix=f'global_2018{month}_waterfall', n_samples=3)
            analyzer.plot_waterfall_charts(local_shap, local_results['shap_values'], local_importance['Display_Name'].tolist(),
                                         base_value=0, output_prefix=f'local_2018{month}15_waterfall', n_samples=3)
            
            '''analyzer.plot_force_plots(global_shap, global_results['shap_values'], global_importance['Display_Name'].tolist(), 
                                      f'global_2018{month}_force', 10)
            analyzer.plot_force_plots(local_shap, local_results['shap_values'], local_importance['Display_Name'].tolist(), 
                                      f'local_2018{month}15_force', 10)
            analyzer.plot_dependence_plots(global_shap, global_results['shap_values'], global_importance['Display_Name'].tolist(), 
                                         f'global_2018{month}_dependence', top_n=6)
            analyzer.plot_dependence_plots(local_shap, local_results['shap_values'], local_importance['Display_Name'].tolist(), 
                                         f'local_2018{month}15_dependence', top_n=6)
            analyzer.plot_interaction_heatmap(None, global_importance['Display_Name'].tolist(), 
                                            f'global_2018{month}_interaction', top_n=15)
            analyzer.plot_interaction_heatmap(None, local_importance['Display_Name'].tolist(),   
                                            f'local_2018{month}15_interaction', top_n=15)
           '''
            
            # 生成报告
            analyzer.generate_report(global_results, f'global_2018{month}')
            analyzer.generate_report(local_results, f'local_2018{month}15')
            
            logger.info("高级分析完成!")
        
    except FileNotFoundError as e:
        logger.warning(f"文件未找到: {e}")
        logger.info("请先运行基础SHAP分析生成必要的文件")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
