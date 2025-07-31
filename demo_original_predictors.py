#!/usr/bin/env python3
"""
原版本预测器使用演示
使用完整版的预测器进行精度预测和分析
"""

import numpy as np
import matplotlib.pyplot as plt
from accuracy_retention_predictor import AccuracyRetentionPredictor
from single_round_predictor import SingleRoundRetentionPredictor
from round_comparator import RoundComparator

def demo_full_retention_predictor():
    """演示完整版精度保留率预测器的使用"""
    print("=" * 60)
    print("完整版精度保留率预测器演示")
    print("=" * 60)
    
    # 创建预测器实例（数据在初始化时自动加载）
    predictor = AccuracyRetentionPredictor('docs/accs.csv')
    
    # 拟合所有模型
    best_model_name = predictor.fit_all_models()
    
    # 显示模型性能总结
    print("\n模型性能总结:")
    summary_df = predictor.get_model_summary()
    print(summary_df.to_string(index=False))
    
    # 预测示例
    print("\n预测示例:")
    test_ratios = [0.25, 0.3, 0.5, 0.7, 0.9]
    
    for ratio in test_ratios:
        retention_rate = predictor.predict_retention_rate(ratio)[0]
        abs_accuracy = predictor.predict_absolute_accuracy(ratio, global_accuracy=0.65)[0]
        
        print(f"保留比例 {ratio:.2f}:")
        print(f"  - 精度保留率: {retention_rate:.2f}%")
        print(f"  - 预测准确率: {abs_accuracy:.4f} (假设全局准确率65%)")
        print(f"  - 准确率损失: {0.65 - abs_accuracy:.4f}")
    
    # 绘制结果
    predictor.plot_results('accuracy_retention_results.png')
    print(f"\n结果图表已保存为: accuracy_retention_results.png")
    
    return predictor

def demo_single_round_predictor():
    """演示完整版单轮次预测器的使用"""
    print("\n" + "=" * 60)
    print("完整版单轮次预测器演示")
    print("=" * 60)
    
    # 分析多个轮次
    rounds_to_analyze = [1, 5, 10, 15]
    
    for round_num in rounds_to_analyze:
        print(f"\n--- 轮次 {round_num} 分析 ---")
        
        # 创建单轮次预测器
        predictor = SingleRoundRetentionPredictor('docs/accs.csv')
        
        # 加载指定轮次数据
        predictor.load_round_data(round_num)
        print(f"轮次 {round_num} 数据加载完成")
        print(f"- 全局模型准确率: {predictor.global_accuracy:.4f}")
        print(f"- 数据点数: {len(predictor.retention_ratios)}")
        
        # 拟合模型
        predictor.fit_models()
        
        # 获取最佳模型
        best_model = predictor.get_best_model()
        print(f"- 最佳模型: {best_model}")
        print(f"- R² = {predictor.model_performance[best_model]['r2']:.4f}")
        print(f"- MSE = {predictor.model_performance[best_model]['mse']:.6f}")
        
        # 预测示例
        test_ratios = [0.3, 0.5, 0.7, 0.9]
        print(f"\n轮次 {round_num} 预测结果:")
        
        for ratio in test_ratios:
            retention_rate = predictor.predict_retention_rate(ratio)
            abs_accuracy = predictor.predict_absolute_accuracy(ratio)
            
            print(f"  保留比例 {ratio:.1f}: 精度保留率 {retention_rate:.2f}%, "
                  f"预测准确率 {abs_accuracy:.4f}")

def demo_round_comparison():
    """演示轮次比较分析工具的使用"""
    print("\n" + "=" * 60)
    print("轮次比较分析工具演示")
    print("=" * 60)
    
    # 创建比较器
    comparator = RoundComparator('docs/accs.csv')
    
    # 加载指定轮次
    rounds_to_compare = [1, 5, 10, 15]
    comparator.load_all_rounds(rounds_to_compare)
    
    print(f"已加载轮次: {rounds_to_compare}")
    
    # 比较模型性能
    print("\n模型性能比较:")
    performance_df = comparator.compare_model_performance()
    print(performance_df.to_string(index=False))
    
    # 在不同保留比例下比较预测结果
    print("\n不同保留比例下的预测比较:")
    test_ratios = [0.3, 0.5, 0.7, 0.9]
    
    for ratio in test_ratios:
        print(f"\n保留比例 {ratio}:")
        prediction_df = comparator.compare_predictions_at_ratio(ratio)
        print(prediction_df.to_string(index=False))
        
        # 找到最佳轮次
        best_round = comparator.find_best_round_for_ratio(ratio)
        print(f"最佳轮次: {best_round['round']} (精度保留率: {best_round['retention_rate']:.2f}%)")
    
    # 分析稳定性
    print("\n稳定性分析:")
    stability_stats = comparator.analyze_round_stability()
    
    # 可视化比较结果
    comparator.visualize_comparison()
    print(f"\n比较图表已保存为: round_comparison_results.png")
    
    return comparator

def demo_federated_learning_application():
    """演示在联邦学习中的实际应用"""
    print("\n" + "=" * 60)
    print("联邦学习实际应用演示")
    print("=" * 60)
    
    # 使用完整版预测器
    predictor = AccuracyRetentionPredictor('docs/accs.csv')
    predictor.load_and_preprocess_data()
    predictor.fit_all_models()
    
    # 模拟联邦学习场景
    clients = [
        {'id': 1, 'name': '高性能客户端', 'resource_level': 0.9, 'data_size': 1000},
        {'id': 2, 'name': '中等性能客户端', 'resource_level': 0.7, 'data_size': 800},
        {'id': 3, 'name': '低性能客户端', 'resource_level': 0.5, 'data_size': 600},
        {'id': 4, 'name': '资源受限客户端', 'resource_level': 0.3, 'data_size': 400}
    ]
    
    print("客户端性能分析:")
    print("-" * 80)
    print(f"{'客户端':<15} {'保留比例':<10} {'精度保留率':<12} {'预测准确率':<12} {'性能等级':<10}")
    print("-" * 80)
    
    global_accuracy = 0.65  # 假设全局模型准确率为65%
    total_weighted_accuracy = 0
    total_data_size = sum(client['data_size'] for client in clients)
    
    for client in clients:
        retention_rate = predictor.predict_retention_rate(client['resource_level'])
        predicted_accuracy = predictor.predict_absolute_accuracy(client['resource_level'], global_accuracy)
        
        # 性能等级评估
        if retention_rate >= 80:
            performance_level = "优秀"
        elif retention_rate >= 60:
            performance_level = "良好"
        elif retention_rate >= 40:
            performance_level = "中等"
        else:
            performance_level = "较差"
        
        print(f"{client['name']:<15} {client['resource_level']:<10.1f} "
              f"{retention_rate:<12.2f}% {predicted_accuracy:<12.4f} {performance_level:<10}")
        
        # 计算加权平均准确率
        weight = client['data_size'] / total_data_size
        total_weighted_accuracy += predicted_accuracy * weight
    
    print("-" * 80)
    print(f"联邦学习加权平均准确率: {total_weighted_accuracy:.4f}")
    print(f"相对于全局模型的性能保留: {(total_weighted_accuracy/global_accuracy)*100:.2f}%")
    
    # 公平性分析
    accuracies = [predictor.predict_absolute_accuracy(client['resource_level'], global_accuracy) 
                  for client in clients]
    fairness_gap = max(accuracies) - min(accuracies)
    print(f"公平性差距: {fairness_gap:.4f}")
    
    if fairness_gap < 0.1:
        fairness_level = "公平"
    elif fairness_gap < 0.2:
        fairness_level = "较公平"
    else:
        fairness_level = "不公平"
    
    print(f"公平性评估: {fairness_level}")

def main():
    """主函数"""
    print("原版本预测器完整演示")
    print("使用完整功能的预测器进行精度分析")
    
    try:
        # 1. 完整版精度保留率预测器演示
        full_predictor = demo_full_retention_predictor()
        
        # 2. 完整版单轮次预测器演示
        demo_single_round_predictor()
        
        # 3. 轮次比较分析演示
        comparator = demo_round_comparison()
        
        # 4. 联邦学习实际应用演示
        demo_federated_learning_application()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("生成的文件:")
        print("- accuracy_retention_results.png (精度保留率分析图)")
        print("- round_comparison_results.png (轮次比较图)")
        print("\n所有原版本预测器都已成功运行并展示了完整功能。")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()