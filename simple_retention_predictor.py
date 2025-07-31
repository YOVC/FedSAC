"""
简化版精度保留率预测器
输入：当前模型的保留比例
输出：子模型应该保留全局模型精度的百分之多少
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def load_and_fit_retention_model(csv_file_path='docs/accs.csv'):
    """
    加载数据并训练精度保留率预测模型
    
    Args:
        csv_file_path: CSV文件路径
    
    Returns:
        训练好的模型、多项式特征转换器和全局模型准确率统计
    """
    # 读取数据
    data = pd.read_csv(csv_file_path, header=None)
    
    # 第一行是保留比例
    retention_ratios = data.iloc[0].values
    
    # 后续行是准确率数据
    accuracy_data = data.iloc[1:].values
    
    # 找到保留比例为1.0的索引（全局模型）
    global_ratio_idx = np.where(retention_ratios == 1.0)[0][0]
    
    # 提取每轮的全局模型准确率
    global_model_accuracies = accuracy_data[:, global_ratio_idx]
    
    # 计算每个保留比例下的精度保留率
    retention_rates_data = []
    for i, row in enumerate(accuracy_data):
        global_acc = global_model_accuracies[i]
        # 计算每个保留比例下的精度保留率
        retention_rates = (row / global_acc) * 100
        retention_rates_data.append(retention_rates)
    
    retention_rates_data = np.array(retention_rates_data)
    
    # 将所有精度保留率数据展平
    accuracy_retention_rates = retention_rates_data.flatten()
    
    # 重复保留比例以匹配精度保留率数据的长度
    retention_ratios_expanded = np.tile(retention_ratios, len(accuracy_data))
    
    # 使用5次多项式拟合（根据之前的结果，这是最佳模型）
    poly_features = PolynomialFeatures(degree=5)
    X_poly = poly_features.fit_transform(retention_ratios_expanded.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_poly, accuracy_retention_rates)
    
    # 计算全局模型准确率统计
    global_acc_stats = {
        'mean': np.mean(global_model_accuracies),
        'std': np.std(global_model_accuracies),
        'min': np.min(global_model_accuracies),
        'max': np.max(global_model_accuracies)
    }
    
    print(f"精度保留率预测模型训练完成！")
    print(f"数据点数: {len(accuracy_retention_rates)}")
    print(f"保留比例范围: {retention_ratios.min():.2f} - {retention_ratios.max():.2f}")
    print(f"精度保留率范围: {accuracy_retention_rates.min():.2f}% - {accuracy_retention_rates.max():.2f}%")
    print(f"全局模型准确率统计: 均值={global_acc_stats['mean']:.4f}, 标准差={global_acc_stats['std']:.4f}")
    
    return model, poly_features, global_acc_stats

def predict_retention_rate(retention_ratio, model, poly_features):
    """
    预测给定保留比例的精度保留率
    
    Args:
        retention_ratio: 保留比例 (可以是单个值或数组)
        model: 训练好的模型
        poly_features: 多项式特征转换器
    
    Returns:
        预测的精度保留率（百分比）
    """
    # 确保输入是numpy数组
    retention_ratio = np.array(retention_ratio).reshape(-1, 1)
    
    # 多项式特征转换
    X_poly = poly_features.transform(retention_ratio)
    
    # 预测
    prediction = model.predict(X_poly)
    
    # 确保预测值在合理范围内（0-100%）
    prediction = np.clip(prediction, 0, 100)
    
    return prediction

def predict_absolute_accuracy(retention_ratio, global_accuracy, model, poly_features):
    """
    预测给定保留比例和全局模型准确率下的绝对准确率
    
    Args:
        retention_ratio: 保留比例
        global_accuracy: 全局模型准确率
        model: 训练好的模型
        poly_features: 多项式特征转换器
    
    Returns:
        预测的绝对准确率
    """
    retention_rate = predict_retention_rate(retention_ratio, model, poly_features)
    absolute_accuracy = (retention_rate / 100) * global_accuracy
    return absolute_accuracy

def get_retention_level_description(retention_rate):
    """
    根据精度保留率返回描述性说明
    
    Args:
        retention_rate: 精度保留率（百分比）
    
    Returns:
        描述性说明
    """
    if retention_rate >= 90:
        return "极高精度保留"
    elif retention_rate >= 80:
        return "高精度保留"
    elif retention_rate >= 60:
        return "中等精度保留"
    elif retention_rate >= 40:
        return "低精度保留"
    else:
        return "极低精度保留"

def main():
    """
    主函数 - 使用示例
    """
    print("=" * 70)
    print("联邦学习子模型精度保留率预测器")
    print("=" * 70)
    
    # 训练模型
    model, poly_features, global_acc_stats = load_and_fit_retention_model()
    
    print("\n" + "=" * 70)
    print("预测示例:")
    print("=" * 70)
    
    # 示例预测
    test_ratios = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"{'保留比例':<10} {'精度保留率':<12} {'保留水平':<15} {'说明':<20}")
    print("-" * 70)
    
    for ratio in test_ratios:
        retention_rate = predict_retention_rate(ratio, model, poly_features)[0]
        level = get_retention_level_description(retention_rate)
        
        if ratio == 1.0:
            explanation = "全局模型基准"
        else:
            explanation = f"保留{retention_rate:.1f}%的全局精度"
        
        print(f"{ratio:<10.2f} {retention_rate:<12.2f}% {level:<15} {explanation:<20}")
    
    print("\n" + "=" * 70)
    print("实际应用场景分析:")
    print("=" * 70)
    
    # 使用平均全局准确率进行分析
    avg_global_acc = global_acc_stats['mean']
    print(f"基于平均全局模型准确率: {avg_global_acc:.4f} ({avg_global_acc*100:.2f}%)")
    print(f"{'保留比例':<10} {'精度保留率':<12} {'预测准确率':<12} {'准确率损失':<12} {'应用建议':<15}")
    print("-" * 75)
    
    scenarios = [
        (0.3, "资源极度受限"),
        (0.5, "资源受限"),
        (0.7, "平衡性能与资源"),
        (0.9, "追求高性能")
    ]
    
    for ratio, scenario in scenarios:
        retention_rate = predict_retention_rate(ratio, model, poly_features)[0]
        pred_acc = predict_absolute_accuracy(ratio, avg_global_acc, model, poly_features)[0]
        loss = avg_global_acc - pred_acc
        
        print(f"{ratio:<10.2f} {retention_rate:<12.2f}% {pred_acc:<12.4f} {loss:<12.4f} {scenario:<15}")
    
    print("\n" + "=" * 70)
    print("交互式预测:")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\n请输入保留比例 (0.25-1.0) 或全局准确率,保留比例 (输入'q'退出): ")
            
            if user_input.lower() == 'q':
                print("退出程序。")
                break
            
            # 检查是否包含逗号（表示输入了全局准确率和保留比例）
            if ',' in user_input:
                parts = user_input.split(',')
                global_acc = float(parts[0].strip())
                ratio = float(parts[1].strip())
                
                retention_rate = predict_retention_rate(ratio, model, poly_features)[0]
                pred_acc = predict_absolute_accuracy(ratio, global_acc, model, poly_features)[0]
                loss = global_acc - pred_acc
                level = get_retention_level_description(retention_rate)
                
                print(f"\n结果分析:")
                print(f"全局模型准确率: {global_acc:.4f} ({global_acc*100:.2f}%)")
                print(f"保留比例: {ratio:.3f}")
                print(f"精度保留率: {retention_rate:.2f}% ({level})")
                print(f"预测子模型准确率: {pred_acc:.4f} ({pred_acc*100:.2f}%)")
                print(f"准确率损失: {loss:.4f} ({loss*100:.2f}个百分点)")
                
            else:
                ratio = float(user_input)
                
                if ratio < 0.25 or ratio > 1.0:
                    print("警告: 保留比例超出训练数据范围 (0.25-1.0)，预测可能不准确。")
                
                retention_rate = predict_retention_rate(ratio, model, poly_features)[0]
                level = get_retention_level_description(retention_rate)
                
                print(f"\n保留比例 {ratio:.3f} 的预测结果:")
                print(f"精度保留率: {retention_rate:.2f}% ({level})")
                print(f"说明: 子模型将保留全局模型 {retention_rate:.1f}% 的精度")
                
                # 基于平均全局准确率给出绝对准确率预测
                pred_acc = predict_absolute_accuracy(ratio, avg_global_acc, model, poly_features)[0]
                print(f"基于平均全局准确率({avg_global_acc:.4f})的预测准确率: {pred_acc:.4f}")
            
        except ValueError:
            print("请输入有效的数字！格式: 保留比例 或 全局准确率,保留比例")
        except KeyboardInterrupt:
            print("\n程序被中断。")
            break
    
    return model, poly_features, global_acc_stats

if __name__ == "__main__":
    model, poly_features, global_acc_stats = main()