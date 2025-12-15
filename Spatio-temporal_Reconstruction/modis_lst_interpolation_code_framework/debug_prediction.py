"""
调试脚本：诊断预测值异常的根本原因
"""
import torch
import numpy as np
from datetime import datetime
import json

# 配置
config_path = r"G:\CNN_SpatialDownscaling\scripts\Spatio-temporal_Reconstruction\modis_lst_interpolation_code_framework\config.json"
local_finetune_dir = r"G:\CNN_SpatialDownscaling\output\local_finetune"

target_date = datetime(2018, 1, 15)
date_str = target_date.strftime('%Y%m%d')
model_path = f"{local_finetune_dir}/local_finetune_{date_str}.pt"

print("="*80)
print("诊断预测值异常问题")
print("="*80)

# 1. 加载模型和scaler
print("\n1. 加载局部微调模型...")
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
scaler_dict = checkpoint['scaler_dict']

print(f"✓ 模型加载成功")
print(f"\nScaler信息:")

# 2. 检查target scaler
target_scaler = scaler_dict['target']
print(f"\nTarget Scaler:")
print(f"  Mean: {target_scaler.mean_[0]:.4f} K")
print(f"  Std:  {target_scaler.scale_[0]:.4f} K")

# 3. 模拟预测过程
print(f"\n2. 模拟标准化/反标准化过程...")
print("="*80)

# 模拟真实LST值（观测数据的范围）
real_lst_values = np.array([260.30, 270.00, 280.00, 290.00, 296.26])
print(f"\n真实LST值 (观测范围): {real_lst_values}")

# 标准化
real_lst_normalized = target_scaler.transform(real_lst_values.reshape(-1, 1)).flatten()
print(f"标准化后: {real_lst_normalized}")
print(f"  均值: {real_lst_normalized.mean():.4f}")
print(f"  标准差: {real_lst_normalized.std():.4f}")

# 反标准化
real_lst_denormalized = target_scaler.inverse_transform(real_lst_normalized.reshape(-1, 1)).flatten()
print(f"反标准化后: {real_lst_denormalized}")
print(f"  是否一致: {np.allclose(real_lst_values, real_lst_denormalized)}")

# 4. 模拟预测值（模型输出的标准化值）
print(f"\n3. 模拟模型预测的标准化输出...")
print("="*80)

# 假设模型预测的标准化输出在 [-2, 2] 范围内（正常情况）
predicted_normalized_normal = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
print(f"\n正常预测（标准化）: {predicted_normalized_normal}")

# 反标准化
predicted_denorm_normal = target_scaler.inverse_transform(predicted_normalized_normal.reshape(-1, 1)).flatten()
print(f"反标准化后: {predicted_denorm_normal}")
print(f"  范围: [{predicted_denorm_normal.min():.2f}, {predicted_denorm_normal.max():.2f}] K")
print(f"  标准差: {predicted_denorm_normal.std():.2f} K")

# 5. 检查是否是模型输出过于集中
print(f"\n4. 检查异常预测情况...")
print("="*80)

# 你观察到的异常预测
abnormal_denorm = np.array([266.92, 268.00, 268.16, 269.00, 270.72])
print(f"\n异常预测（反标准化后）: {abnormal_denorm}")
print(f"  范围: [{abnormal_denorm.min():.2f}, {abnormal_denorm.max():.2f}] K")
print(f"  均值: {abnormal_denorm.mean():.2f} K")
print(f"  标准差: {abnormal_denorm.std():.2f} K ⚠️ 过小！")

# 反推标准化前的值
abnormal_normalized = target_scaler.transform(abnormal_denorm.reshape(-1, 1)).flatten()
print(f"\n反推标准化前的值: {abnormal_normalized}")
print(f"  范围: [{abnormal_normalized.min():.4f}, {abnormal_normalized.max():.4f}]")
print(f"  标准差: {abnormal_normalized.std():.4f} ⚠️ 如果这个值很小，说明模型输出过于集中！")

# 6. 诊断结论
print(f"\n5. 诊断结论:")
print("="*80)

if abnormal_normalized.std() < 0.1:
    print("❌ 问题诊断：模型预测的标准化输出过于集中！")
    print("\n可能原因：")
    print("  1. 模型训练不充分（过早收敛到局部最优）")
    print("  2. 学习率过小导致模型参数更新不足")
    print("  3. 冻结层过多，模型表达能力不足")
    print("  4. 训练数据与测试数据分布差异过大")
    print("\n建议解决方案：")
    print("  方案1: 调整局部微调的超参数")
    print("    - 减少冻结层数（freeze_layers: 2 -> 0）")
    print("    - 增加学习率（learning_rate: 0.00005 -> 0.0001）")
    print("    - 增加训练轮数（epochs: 100 -> 200）")
    print("  方案2: 检查训练数据质量")
    print("    - 确认局部数据的LST分布是否正常")
    print("    - 检查数据标准化是否正确")
    print("  方案3: 验证全局模型性能")
    print("    - 直接用全局模型预测，看结果是否正常")
else:
    print("✓ 标准化/反标准化过程正常")
    print("  问题可能出在其他地方")

# 7. 特征scaler检查
print(f"\n6. 检查特征scaler...")
print("="*80)

critical_features = ['t2m', 'ssrd', 'strd', 'rh', 'd2m', 'vpd']
for feat in critical_features:
    if feat in scaler_dict:
        scaler = scaler_dict[feat]
        print(f"{feat}:")
        print(f"  Mean: {scaler.mean_[0]:.4f}")
        print(f"  Std:  {scaler.scale_[0]:.4f}")

print("\n" + "="*80)
print("诊断完成！请根据以上信息进行针对性修复。")
print("="*80)
