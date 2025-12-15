import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import torch
from datetime import datetime

# 设置日志
logger = logging.getLogger(__name__)

def set_plot_style(font_sans=['SimHei','Arial', 'DejaVu Sans']):
    """
    设置符合学术出版要求的全局绘图风格
    """
    plt.rcParams['font.sans-serif'] = font_sans          # 设置字体优先列表（按顺序回退，支持中/英文字体混合显示）
    plt.rcParams['axes.unicode_minus'] = False          # 允许坐标轴负号正常显示（避免中文环境下被替换为方块）
    plt.rcParams['xtick.direction'] = 'out'              # x 轴刻度线朝外绘制（符合学术绘图习惯）
    plt.rcParams['ytick.direction'] = 'out'              # y 轴刻度线朝外绘制
    plt.rcParams['xtick.top'] = False                    # 在图表顶部不显示 x 轴刻度（提高可读性）
    plt.rcParams['ytick.right'] = False                  # 在图表右侧不显示 y 轴刻度
    plt.rcParams['xtick.major.size'] = 6                # 主刻度线长度（像素）
    plt.rcParams['ytick.major.size'] = 6                # 主刻度线长度（像素）
    plt.rcParams['xtick.minor.size'] = 3                # 次刻度线长度（像素）
    plt.rcParams['ytick.minor.size'] = 3                # 次刻度线长度（像素）
    plt.rcParams['font.size'] = 12                      # 全局默认字体大小（用于标签、注释等）
    plt.rcParams['axes.linewidth'] = 1                # 坐标轴线宽（提高打印和展示清晰度）
    plt.rcParams['legend.frameon'] = False              # 图例不绘制边框（更简洁、美观）


# 初始化风格
set_plot_style()

def _clean_data(y_true, y_pred):
    """内部工具：数据清洗与扁平化"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    return y_true[mask], y_pred[mask]

def _calc_metrics(y_true, y_pred):
    """内部工具：计算常用指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return rmse, mae, r2, bias

def plot_density_scatter(y_true, y_pred, output_path, 
                         title='Model Validation',
                         xlabel='Reference LST (K)',
                         ylabel='Predicted LST (K)',
                         figsize=(8, 8),
                         dpi=300,
                         sample_threshold=50000):
    """
    绘制符合学术标准的密度散点图（优化版）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        output_path: 保存路径
        sample_threshold: 触发降采样的阈值，防止KDE计算过慢
    """
    # 1. 数据清洗
    targets, predictions = _clean_data(y_true, y_pred)
    n_samples = len(targets)
    
    if n_samples == 0:
        logger.warning(f"没有有效数据用于绘制散点图: {output_path}")
        return

    # 2. 计算指标
    rmse, mae, r2, bias = _calc_metrics(targets, predictions)
    
    # 3. 计算密度 (优化版)
    xy = np.vstack([targets, predictions])
    
    # 如果数据量过大，随机采样部分点来估算密度函数（显著提升速度）
    if n_samples > sample_threshold:
        indices = np.random.choice(n_samples, sample_threshold, replace=False)
        kde_model = gaussian_kde(xy[:, indices])
        # 仅对这部分点计算密度用于排序，或者对所有点计算(耗时)
        # 这里为了视觉效果，对所有点计算密度值，但基于采样后的KDE模型
        z = kde_model(xy)
    else:
        z = gaussian_kde(xy)(xy)
    
    # 按密度排序，确保高密度点显示在上方
    idx = z.argsort()
    targets_sorted = targets[idx]
    predictions_sorted = predictions[idx]
    z_sorted = z[idx]
    
    # 4. 绘图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 自定义颜色映射
    colors = ['#00008B', '#0000FF', '#00FFFF', '#00FF00', 
              '#FFFF00', '#FF8C00', '#FF0000', '#8B0000']
    cmap = LinearSegmentedColormap.from_list('density', colors, N=256)
    
    scatter = ax.scatter(targets_sorted, predictions_sorted, c=z_sorted, s=2, 
                         cmap=cmap, alpha=0.8, edgecolors='none', rasterized=True)
    
    # 1:1 参考线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    # 留一点边距
    margin = (max_val - min_val) * 0.05
    min_ax = min_val - margin
    max_ax = max_val + margin
    
    ax.plot([min_ax, max_ax], [min_ax, max_ax], 'k--', linewidth=1.5, label='1:1 Line')
    
    # 5. 统计文本框
    stats_text = (f'N = {n_samples}\n'
                  f'Bias = {bias:.2f}\n'
                  f'RMSE = {rmse:.2f}\n'
                  f'MAE = {mae:.2f}\n'
                  f'$R^2$ = {r2:.4f}')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, family='monospace')
    
    # 6. 装饰
    ax.set_xlabel(xlabel, fontsize=14, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14, weight='bold')
    ax.set_title(title, fontsize=16, weight='bold', pad=15)
    
    ax.set_xlim(min_ax, max_ax)
    ax.set_ylim(min_ax, max_ax)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density', rotation=270, labelpad=15)
    cbar.set_ticks([])  # 隐藏具体密度数值
    
    plt.tight_layout()
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"密度散点图已保存: {output_path} (RMSE={rmse:.2f}, R2={r2:.4f})")
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'bias': bias}

def plot_training_history(history, output_path, title_prefix="Training History"):
    """
    通用训练历史曲线绘制
    自动识别 history 字典中的键来决定布局：
    - Global模式: train_loss, val_loss, val_r2, val_rmse
    - Local模式: train_loss, train_mse, train_kl, val_rmse, etc.
    """
    
    # 判断是否为包含KL散度的局部微调历史
    has_kl = 'train_kl' in history and any(k > 0 for k in history['train_kl'])
    
    if has_kl:
        # 布局：2x2 (Loss总览, Loss分解, R2, RMSE)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 总Loss
        ax = axes[0, 0]
        ax.plot(history.get('train_loss', []), label='Train Total Loss', linewidth=2)
        ax.plot(history.get('val_loss', []), label='Val Loss', linewidth=2)
        ax.set_title('Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Loss分解
        ax = axes[0, 1]
        ax.plot(history.get('train_mse', []), label='MSE Loss', linewidth=2, color='tab:blue')
        ax.plot(history.get('train_kl', []), label='KL Div (Distillation)', linewidth=2, color='tab:orange')
        ax.set_title('Training Loss Decomposition')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. R2
        ax = axes[1, 0]
        if 'val_r2' in history:
            ax.plot(history['val_r2'], label='Val $R^2$', color='tab:green', linewidth=2)
        if 'train_r2' in history:
            ax.plot(history['train_r2'], label='Train $R^2$', color='tab:green', linestyle='--', alpha=0.6)
        ax.set_title('$R^2$ Score')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. RMSE
        ax = axes[1, 1]
        if 'val_rmse' in history:
            ax.plot(history['val_rmse'], label='Val RMSE', color='tab:red', linewidth=2)
        if 'train_rmse' in history:
            ax.plot(history['train_rmse'], label='Train RMSE', color='tab:red', linestyle='--', alpha=0.6)
        ax.set_title('RMSE (K)')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    else:
        # 布局：1x3 (适用于全局预训练) -> Loss, RMSE(或者MSE), R2
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Loss (Standardized or Raw)
        ax = axes[0]
        epochs = range(1, len(history.get('train_losses', [])) + 1)
        # 兼容不同的键名
        t_loss = history.get('train_losses', history.get('train_loss', []))
        v_loss = history.get('val_losses', history.get('val_loss', []))
        
        ax.plot(epochs, t_loss, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, v_loss, 'r-', label='Val Loss', linewidth=2)
        ax.set_title(f'{title_prefix} - Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Metric 1 (MSE/RMSE)
        ax = axes[1]
        # 优先找 RMSE，其次找 True MSE (K)
        if 'val_rmse' in history or 'val_true_mses' in history:
            if 'val_rmse' in history:
                ax.plot(epochs, history['val_rmse'], 'g-', label='Val RMSE (K)', linewidth=2)
                ylabel = 'RMSE'
            else:
                ax.plot(epochs, history['val_true_mses'], 'g-', label='Val MSE (K)', linewidth=2)
                ylabel = 'MSE'
            ax.set_title(f'{title_prefix} - {ylabel}')
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Metric 2 (R2)
        ax = axes[2]
        r2_key = 'val_r2_scores' if 'val_r2_scores' in history else 'val_r2'
        if r2_key in history:
            ax.plot(epochs, history[r2_key], 'm-', label='Val $R^2$', linewidth=2)
            ax.set_title(f'{title_prefix} - $R^2$ Score')
            ax.set_xlabel('Epoch')
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"训练曲线已保存: {output_path}")

def plot_hyperparameter_search(results_dict, output_path, title='Hyperparameter Search Results'):
    """
    可视化超参数搜索结果
    Args:
        results_dict: { 'param_str': metric_value }
    """
    if not results_dict:
        return

    # 排序
    sorted_results = sorted(results_dict.items(), key=lambda item: item[1])
    labels = [item[0] for item in sorted_results]
    values = [item[1] for item in sorted_results]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, values, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Validation Loss / MSE', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱状图上标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"超参数搜索结果图已保存: {output_path}")

def plot_dual_comparison(y_true, pred1, pred2, output_path, 
                         label1='Model 1', label2='Model 2',
                         figsize=(16, 7)):
    """
    绘制双模型对比图 (如 Noah-MP vs DeepLearning)
    """
    targets, p1 = _clean_data(y_true, pred1)
    _, p2 = _clean_data(y_true, pred2)
    
    # 确保长度一致 (基于y_true的mask可能略有不同，取交集最稳妥，这里简化处理假设y_true一致)
    # 实际应用中最好先统一对齐数据
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 公共范围
    all_data = np.concatenate([targets, p1, p2])
    min_val, max_val = all_data.min(), all_data.max()
    
    # 绘图函数复用内部逻辑
    def _draw_subplot(ax, true, pred, label, sub_label):
        rmse, mae, _, bias = _calc_metrics(true, pred)
        
        # 密度计算
        xy = np.vstack([true, pred])
        if len(true) > 50000:
            idx_sample = np.random.choice(len(true), 50000, replace=False)
            z = gaussian_kde(xy[:, idx_sample])(xy)
        else:
            z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        
        colors = ['#00008B', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000', '#8B0000']
        cmap = LinearSegmentedColormap.from_list('d', colors)
        
        sc = ax.scatter(true[idx], pred[idx], c=z[idx], s=2, cmap=cmap, edgecolors='none', rasterized=True)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
        
        # 文本
        stats = f'Bias={bias:.2f}\nRMSE={rmse:.2f}\nMAE={mae:.2f}'
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), va='top', fontfamily='monospace')
        
        ax.set_xlabel('Reference LST (K)', fontsize=12, weight='bold')
        ax.set_ylabel(f'{label} (K)', fontsize=12, weight='bold')
        ax.set_title(sub_label, loc='left', fontsize=14, weight='bold')
        
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        return sc

    sc1 = _draw_subplot(ax1, targets, p1, label1, '(a)')
    sc2 = _draw_subplot(ax2, targets, p2, label2, '(b)')
    
    # 公共Colorbar
    cbar = fig.colorbar(sc2, ax=[ax1, ax2], fraction=0.03, pad=0.02)
    cbar.set_label('Density', rotation=270, labelpad=15)
    cbar.set_ticks([])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"双模型对比图已保存: {output_path}")

def plot_hexbin_scatter(y_true, y_pred, output_path,
                        title='Model Validation (Hexbin)',
                        xlabel='Reference LST (K)',
                        ylabel='Predicted LST (K)',
                        figsize=(8, 8), gridsize=100):
    """
    Hexbin图：适合超大规模数据的密度可视化，速度极快。
    """
    # 清洗数据
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    targets = y_true[mask]
    predictions = y_pred[mask]
    
    # 计算指标
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 核心：使用 hexbin 代替 scatter
    # bins='log' 可以让低密度区域也能看清，避免高密度区域掩盖一切
    hb = ax.hexbin(targets, predictions, gridsize=gridsize, cmap='jet', bins='log', mincnt=1)
    
    # 1:1 线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    # 统计信息
    stats_text = f'N={len(targets)}\nRMSE={rmse:.2f}\n$R^2$={r2:.4f}\nMAE={mean_absolute_error(targets, predictions):.2f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', fontsize=12)
    
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('$log_{10}^(Count)$', rotation=270, labelpad=15)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Hexbin图已保存: {output_path}")


def plot_model_scatter(model, data_loader, scaler_dict, file_prefix, output_dir,                                              
                        xlabel='Reference LST (K)',
                        ylabel='Predicted LST (K)',
                       device=None, dpi=600, figsize=(8, 8)):
    """绘制并保存预测-实际散点图（从模型和data_loader批量推断）

    Args:
        model: PyTorch 模型，接受 batch dict 返回张量
        data_loader: DataLoader，产出包含 'num_features','time_features','loc_features','cat','target' 的 batch
        scaler_dict: 包含 'target' 的 scaler（需实现 inverse_transform）
        file_prefix: 文件名前缀（如 'YYYYMMDD'）
        output_dir: 输出目录
        device: 可选，PyTorch 设备字符串或 torch.device
    Returns:
        dict: {'rmse','mae','r2'}
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch_data = {
                'num': batch['num_features'].to(device),
                'time': batch['time_features'].to(device),
                'loc': batch['loc_features'].to(device),
                'cat': {k: v.to(device) for k, v in batch['cat'].items()}
            }
            targets = batch['target'].to(device)
            outputs = model(batch_data).squeeze(-1)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if len(all_preds) == 0:
        logger.warning(f"没有预测数据用于绘图: {file_prefix}")
        return None

    preds_np = np.concatenate(all_preds).ravel()
    targets_np = np.concatenate(all_targets).ravel()

    # 反标准化
    try:
        preds_denorm = scaler_dict['target'].inverse_transform(preds_np.reshape(-1, 1)).flatten()
        targets_denorm = scaler_dict['target'].inverse_transform(targets_np.reshape(-1, 1)).flatten()
    except Exception as e:
        logger.error(f"反标准化失败: {e}")
        preds_denorm = preds_np
        targets_denorm = targets_np

    # 绘图
    plt.figure(figsize=figsize)
    plt.scatter(targets_denorm, preds_denorm, alpha=0.5, s=1)

    # 绘制对角线
    min_val = min(np.nanmin(targets_denorm), np.nanmin(preds_denorm))
    max_val = max(np.nanmax(targets_denorm), np.nanmax(preds_denorm))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')

    # 计算并标注度量指标
    rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
    r2 = r2_score(targets_denorm, preds_denorm)
    mae = mean_absolute_error(targets_denorm, preds_denorm)

    plt.text(0.05, 0.95, f'RMSE: {rmse:.2f} K', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.90, f'$R^2$: {r2:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.85, f'MAE: {mae:.2f} K', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # 设置图表标题和标签
    try:
        dt_title = datetime.strptime(file_prefix, "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        dt_title = str(file_prefix)
    plt.title(f'Local Fine-tuning: {dt_title}', fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图像
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    plt.savefig(output_dir, dpi=dpi, bbox_inches='tight')
    logger.info(f"预测-实际散点图已保存到: {output_dir}")
    plt.close()

    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def loal_plot_scatter( model, data_loader, scaler_dict, file_prefix, output_dir):
        """绘制并保存预测-实际散点图"""
        device = torch.device(device)
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch_data = {
                    'num': batch['num_features'].to(device),
                    'time': batch['time_features'].to(device),
                    'loc': batch['loc_features'].to(device),
                    'cat': {k: v.to(device) for k, v in batch['cat'].items()}
                }
                targets = batch['target'].to(device)
                outputs = model(batch_data).squeeze(-1)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        preds_np = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)
        
        # 反标准化
        preds_denorm = scaler_dict['target'].inverse_transform(preds_np.reshape(-1, 1)).flatten()
        targets_denorm = scaler_dict['target'].inverse_transform(targets_np.reshape(-1, 1)).flatten()

        # 绘图
        plt.figure(figsize=(8, 8))
        plt.scatter(targets_denorm, preds_denorm, alpha=0.5, s=1)
        
        # 绘制对角线
        min_val = min(targets_denorm.min(), preds_denorm.min())
        max_val = max(targets_denorm.max(), preds_denorm.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
        
        # 计算并标注度量指标
        rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
        r2 = r2_score(targets_denorm, preds_denorm)
        mae = mean_absolute_error(targets_denorm, preds_denorm)

        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f} K', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.90, f'$R^2$: {r2:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.85, f'MAE: {mae:.2f} K', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # 设置图表标题和标签
        plt.title(f'Local Fine-tuning: {datetime.strptime(file_prefix, "%Y%m%d").strftime("%Y-%m-%d")}', fontsize=14)
        plt.xlabel('Original LST (K)', fontsize=12)
        plt.ylabel('Predicted LST (K)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        scatter_path = os.path.join(output_dir, f"local_finetune_{file_prefix}_scatter.png")
        plt.savefig(scatter_path, dpi=600)
        logger.info(f"预测-实际散点图已保存到: {scatter_path}")
        plt.close()

def plot_scatter(targets, outputs, output_dir, file_prefix, dpi=600):
        """绘制并保存预测-实际散点图"""
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, outputs, alpha=0.5, s=1)
        
        # 绘制对角线
        min_val = min(targets.min(), outputs.min())
        max_val = max(targets.max(), outputs.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 计算并标注度量指标
        rmse = np.sqrt(mean_squared_error(targets, outputs))
        r2 = r2_score(targets, outputs)
        mae = mean_absolute_error(targets, outputs)
        
        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f} K', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.90, f'$R^2$: {r2:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.85, f'MAE: {mae:.2f} K', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # 设置图表标题和标签
        plt.title(datetime.strptime(file_prefix, '%Y%m').strftime('%Y-%m'))
        plt.xlabel('Original LST (K)')
        plt.ylabel('Predicted LST (K)')
        plt.grid(True)
        
        # 保存图像
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        plt.savefig(output_dir, dpi=dpi, bbox_inches='tight')
        
        logger.info(f"预测-实际散点图已保存到: {output_dir}")
        plt.close()
    
