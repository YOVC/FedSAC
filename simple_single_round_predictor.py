#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版单轮次精度保留率预测器
快速拟合指定轮次的数据并进行预测
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_round_data(csv_file, round_num):
    """
    加载指定轮次的数据
    
    Args:
        csv_file: CSV文件路径
        round_num: 轮次编号 (1-15)
    
    Returns:
        tuple: (保留比例, 子模型准确率, 全局模型准确率, 精度保留率)
    """
    try:
        # 读取CSV文件
        data = pd.read_csv(csv_file, header=None)
        
        # 第一行是保留比例
        retention_ratios = data.iloc[0].values
        
        # 检查轮次是否有效
        total_rounds = len(data) - 1
        if round_num < 1 or round_num > total_rounds:
            raise ValueError(f"轮次 {round_num} 无效，可用轮次: 1-{total_rounds}")
        
        # 获取指定轮次的准确率数据 (第round_num+1行，因为第1行是保留比例)
        sub_accuracies = data.iloc[round_num].values
        global_accuracy = sub_accuracies[-1]  # 保留比例1.0对应的准确率
        
        # 计算精度保留率
        retention_rates = (sub_accuracies / global_accuracy) * 100
        
        return retention_ratios, sub_accuracies, global_accuracy, retention_rates
        
    except Exception as e:
        raise Exception(f"数据加载失败: {e}")

def fit_retention_model(retention_ratios, retention_rates, degree=5):
    """
    拟合精度保留率模型
    
    Args:
        retention_ratios: 保留比例
        retention_rates: 精度保留率
        degree: 多项式次数
    
    Returns:
        tuple: (模型, 多项式特征转换器, R², MSE)
    """
    X = retention_ratios.reshape(-1, 1)
    y = retention_rates
    
    # 使用多项式拟合
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # 计算性能指标
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    return model, poly_features, r2, mse

def predict_retention_rate(retention_ratios, model, poly_features):
    """
    预测精度保留率
    
    Args:
        retention_ratios: 保留比例，可以是单个值或数组
        model: 训练好的模型
        poly_features: 多项式特征转换器
    
    Returns:
        预测的精度保留率
    """
    # 确保输入是数组
    ratios = np.array(retention_ratios).reshape(-1, 1)
    
    # 多项式特征转换
    X_poly = poly_features.transform(ratios)
    
    # 预测
    predictions = model.predict(X_poly)
    
    # 确保预测值在合理范围内
    predictions = np.clip(predictions, 0, 100)
    
    return predictions.flatten() if len(predictions) > 1 else predictions[0]

def predict_absolute_accuracy(retention_ratios, model, poly_features, global_accuracy):
    """
    预测绝对准确率
    
    Args:
        retention_ratios: 保留比例
        model: 训练好的模型
        poly_features: 多项式特征转换器
        global_accuracy: 全局模型准确率
    
    Returns:
        预测的绝对准确率
    """
    retention_rates = predict_retention_rate(retention_ratios, model, poly_features)
    if isinstance(retention_rates, (int, float)):
        retention_rates = [retention_rates]
    
    absolute_accuracies = []
    for rate in retention_rates:
        abs_acc = (rate / 100) * global_accuracy
        absolute_accuracies.append(abs_acc)
    
    return np.array(absolute_accuracies).flatten() if len(absolute_accuracies) > 1 else absolute_accuracies[0]

def get_retention_level_description(retention_rate):
    """
    获取精度保留率的描述性说明
    
    Args:
        retention_rate: 精度保留率
    
    Returns:
        描述性说明
    """
    if retention_rate >= 80:
        return "优秀 - 接近原始性能"
    elif retention_rate >= 60:
        return "良好 - 性能损失较小"
    elif retention_rate >= 40:
        return "一般 - 中等性能损失"
    elif retention_rate >= 20:
        return "较差 - 显著性能损失"
    else:
        return "很差 - 严重性能损失"

def load_and_fit_single_round_model(csv_file, round_num, degree=5):
    """
    便捷函数：加载指定轮次数据并拟合模型
    
    Args:
        csv_file: CSV文件路径
        round_num: 轮次编号
        degree: 多项式次数
    
    Returns:
        tuple: (模型, 多项式特征转换器, 全局准确率, 轮次统计信息)
    """
    # 加载数据
    retention_ratios, sub_accuracies, global_accuracy, retention_rates = load_round_data(csv_file, round_num)
    
    # 拟合模型
    model, poly_features, r2, mse = fit_retention_model(retention_ratios, retention_rates, degree)
    
    # 统计信息
    stats = {
        'round_number': round_num,
        'global_accuracy': global_accuracy,
        'retention_rate_range': (retention_rates.min(), retention_rates.max()),
        'accuracy_range': (sub_accuracies.min(), sub_accuracies.max()),
        'r2_score': r2,
        'mse': mse,
        'data_points': len(retention_ratios)
    }
    
    return model, poly_features, global_accuracy, stats

def compare_rounds(csv_file, round_list=None):
    """
    比较不同轮次的模型性能
    
    Args:
        csv_file: CSV文件路径
        round_list: 要比较的轮次列表，默认比较所有轮次
    
    Returns:
        比较结果字典
    """
    # 读取数据确定总轮次数
    data = pd.read_csv(csv_file, header=None)
    total_rounds = len(data) - 1
    
    if round_list is None:
        round_list = list(range(1, total_rounds + 1))
    
    comparison_results = {}
    
    for round_num in round_list:
        try:
            model, poly_features, global_accuracy, stats = load_and_fit_single_round_model(csv_file, round_num)
            comparison_results[round_num] = stats
        except Exception as e:
            print(f"轮次 {round_num} 处理失败: {e}")
    
    return comparison_results

def main():
    """主函数 - 演示使用"""
    print("简化版单轮次精度保留率预测器")
    print("=" * 50)
    
    try:
        csv_file = 'docs/accs.csv'
        
        # 读取数据确定总轮次数
        data = pd.read_csv(csv_file, header=None)
        total_rounds = len(data) - 1
        
        print(f"数据文件: {csv_file}")
        print(f"可用轮次: 1-{total_rounds}")
        
        # 让用户选择轮次
        while True:
            try:
                round_input = input(f"\n请输入要分析的轮次 (1-{total_rounds}): ")
                round_num = int(round_input)
                if 1 <= round_num <= total_rounds:
                    break
                else:
                    print(f"请输入 1-{total_rounds} 之间的数字")
            except ValueError:
                print("请输入有效的数字")
        
        # 加载数据并拟合模型
        print(f"\n正在加载轮次 {round_num} 的数据...")
        model, poly_features, global_accuracy, stats = load_and_fit_single_round_model(csv_file, round_num)
        
        # 显示统计信息
        print(f"\n轮次 {round_num} 统计信息:")
        print(f"全局模型准确率: {stats['global_accuracy']:.4f}")
        print(f"精度保留率范围: {stats['retention_rate_range'][0]:.2f}% - {stats['retention_rate_range'][1]:.2f}%")
        print(f"准确率范围: {stats['accuracy_range'][0]:.4f} - {stats['accuracy_range'][1]:.4f}")
        print(f"模型性能: R² = {stats['r2_score']:.4f}, MSE = {stats['mse']:.6f}")
        print(f"数据点数: {stats['data_points']}")
        
        # 预测示例
        print(f"\n轮次 {round_num} 预测示例:")
        test_ratios = [0.25, 0.3, 0.5, 0.7, 0.9, 1.0]
        retention_rates = predict_retention_rate(test_ratios, model, poly_features)
        abs_accuracies = predict_absolute_accuracy(test_ratios, model, poly_features, global_accuracy)
        
        print(f"{'保留比例':<8} {'精度保留率':<12} {'预测准确率':<12} {'性能等级'}")
        print("-" * 60)
        for ratio, retention, accuracy in zip(test_ratios, retention_rates, abs_accuracies):
            level = get_retention_level_description(retention)
            print(f"{ratio:<8.2f} {retention:<12.2f}% {accuracy:<12.4f} {level}")
        
        # 应用场景分析
        print(f"\n基于轮次 {round_num} 的应用建议:")
        avg_global_acc = global_accuracy
        
        scenarios = [
            (0.3, "边缘设备"),
            (0.5, "移动设备"),
            (0.7, "普通PC"),
            (0.9, "高性能服务器")
        ]
        
        for ratio, device_type in scenarios:
            retention_rate = predict_retention_rate(ratio, model, poly_features)
            pred_accuracy = predict_absolute_accuracy(ratio, model, poly_features, global_accuracy)
            accuracy_loss = global_accuracy - pred_accuracy
            
            print(f"{device_type} (保留比例{ratio}): 精度保留率{retention_rate:.1f}%, "
                  f"预测准确率{pred_accuracy:.4f}, 损失{accuracy_loss:.4f}")
        
        # 交互式预测
        print(f"\n交互式预测 (轮次 {round_num}):")
        while True:
            try:
                ratio_input = input("请输入保留比例 (0.25-1.0，输入 'q' 退出): ")
                if ratio_input.lower() == 'q':
                    break
                
                ratio = float(ratio_input)
                if 0.25 <= ratio <= 1.0:
                    retention_rate = predict_retention_rate(ratio, model, poly_features)
                    abs_accuracy = predict_absolute_accuracy(ratio, model, poly_features, global_accuracy)
                    level = get_retention_level_description(retention_rate)
                    
                    print(f"保留比例: {ratio:.3f}")
                    print(f"精度保留率: {retention_rate:.2f}%")
                    print(f"预测准确率: {abs_accuracy:.4f} ({abs_accuracy*100:.2f}%)")
                    print(f"准确率损失: {global_accuracy - abs_accuracy:.4f}")
                    print(f"性能等级: {level}")
                else:
                    print("保留比例应在 0.25-1.0 之间")
            except ValueError:
                print("请输入有效的数字")
            except EOFError:
                break
        
    except FileNotFoundError:
        print("错误: 找不到 docs/accs.csv 文件")
    except Exception as e:
        print(f"运行错误: {e}")

if __name__ == "__main__":
    main()