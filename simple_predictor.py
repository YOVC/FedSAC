"""
简化版准确率预测器使用示例
基于保留比例预测子模型准确率
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def load_and_fit_model(csv_file_path='docs/accs.csv'):
    """
    加载数据并训练最佳模型
    
    Args:
        csv_file_path: CSV文件路径
    
    Returns:
        训练好的模型和多项式特征转换器
    """
    # 读取数据
    data = pd.read_csv(csv_file_path, header=None)
    
    # 第一行是保留比例
    retention_ratios = data.iloc[0].values
    
    # 后续行是准确率数据
    accuracy_data = data.iloc[1:].values
    
    # 将所有准确率数据展平
    accuracies = accuracy_data.flatten()
    
    # 重复保留比例以匹配准确率数据的长度
    retention_ratios_expanded = np.tile(retention_ratios, len(accuracy_data))
    
    # 使用5次多项式拟合（根据之前的结果，这是最佳模型）
    poly_features = PolynomialFeatures(degree=5)
    X_poly = poly_features.fit_transform(retention_ratios_expanded.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_poly, accuracies)
    
    print(f"模型训练完成！")
    print(f"数据点数: {len(accuracies)}")
    print(f"保留比例范围: {retention_ratios.min():.2f} - {retention_ratios.max():.2f}")
    print(f"准确率范围: {accuracies.min():.4f} - {accuracies.max():.4f}")
    
    return model, poly_features

def predict_accuracy(retention_ratio, model, poly_features):
    """
    预测给定保留比例的准确率
    
    Args:
        retention_ratio: 保留比例 (可以是单个值或数组)
        model: 训练好的模型
        poly_features: 多项式特征转换器
    
    Returns:
        预测的准确率
    """
    # 确保输入是numpy数组
    retention_ratio = np.array(retention_ratio).reshape(-1, 1)
    
    # 多项式特征转换
    X_poly = poly_features.transform(retention_ratio)
    
    # 预测
    prediction = model.predict(X_poly)
    
    return prediction

def main():
    """
    主函数 - 使用示例
    """
    print("=" * 60)
    print("联邦学习子模型准确率预测器")
    print("=" * 60)
    
    # 训练模型
    model, poly_features = load_and_fit_model()
    
    print("\n" + "=" * 60)
    print("预测示例:")
    print("=" * 60)
    
    # 示例预测
    test_ratios = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"{'保留比例':<10} {'预测准确率':<12} {'预测百分比':<12}")
    print("-" * 40)
    
    for ratio in test_ratios:
        pred_acc = predict_accuracy(ratio, model, poly_features)[0]
        print(f"{ratio:<10.2f} {pred_acc:<12.4f} {pred_acc*100:<12.2f}%")
    
    print("\n" + "=" * 60)
    print("交互式预测:")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n请输入保留比例 (0.25-1.0，输入'q'退出): ")
            
            if user_input.lower() == 'q':
                print("退出程序。")
                break
            
            ratio = float(user_input)
            
            if ratio < 0.25 or ratio > 1.0:
                print("警告: 保留比例超出训练数据范围 (0.25-1.0)，预测可能不准确。")
            
            pred_acc = predict_accuracy(ratio, model, poly_features)[0]
            print(f"保留比例 {ratio:.3f} 的预测准确率: {pred_acc:.4f} ({pred_acc*100:.2f}%)")
            
        except ValueError:
            print("请输入有效的数字！")
        except KeyboardInterrupt:
            print("\n程序被中断。")
            break
    
    return model, poly_features

if __name__ == "__main__":
    model, poly_features = main()