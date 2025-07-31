#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮次比较分析工具
比较不同轮次的模型性能和预测结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simple_single_round_predictor import (
    load_and_fit_single_round_model,
    predict_retention_rate,
    predict_absolute_accuracy
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class RoundComparator:
    """轮次比较分析器"""
    
    def __init__(self, csv_file='docs/accs.csv'):
        """
        初始化比较器
        
        Args:
            csv_file: CSV数据文件路径
        """
        self.csv_file = csv_file
        self.round_models = {}
        self.comparison_data = {}
        
        # 读取数据确定总轮次数
        data = pd.read_csv(csv_file, header=None)
        self.total_rounds = len(data) - 1
        
    def load_all_rounds(self, round_list=None):
        """
        加载所有轮次的模型
        
        Args:
            round_list: 要加载的轮次列表，默认加载所有轮次
        """
        if round_list is None:
            round_list = list(range(1, self.total_rounds + 1))
        
        print(f"正在加载 {len(round_list)} 个轮次的数据...")
        
        for round_num in round_list:
            try:
                model, poly_features, global_accuracy, stats = load_and_fit_single_round_model(
                    self.csv_file, round_num
                )
                
                self.round_models[round_num] = {
                    'model': model,
                    'poly_features': poly_features,
                    'global_accuracy': global_accuracy,
                    'stats': stats
                }
                
                print(f"轮次 {round_num}: R² = {stats['r2_score']:.4f}, "
                      f"全局准确率 = {global_accuracy:.4f}")
                
            except Exception as e:
                print(f"轮次 {round_num} 加载失败: {e}")
        
        print(f"成功加载 {len(self.round_models)} 个轮次的模型")
    
    def compare_model_performance(self):
        """比较不同轮次的模型性能"""
        if not self.round_models:
            raise ValueError("请先调用 load_all_rounds() 加载模型")
        
        rounds = sorted(self.round_models.keys())
        r2_scores = []
        mse_scores = []
        global_accuracies = []
        
        for round_num in rounds:
            stats = self.round_models[round_num]['stats']
            r2_scores.append(stats['r2_score'])
            mse_scores.append(stats['mse'])
            global_accuracies.append(stats['global_accuracy'])
        
        # 创建性能比较表
        comparison_df = pd.DataFrame({
            '轮次': rounds,
            'R²分数': r2_scores,
            'MSE': mse_scores,
            '全局准确率': global_accuracies
        })
        
        print("\n轮次性能比较:")
        print("=" * 60)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # 找出最佳和最差轮次
        best_r2_round = rounds[np.argmax(r2_scores)]
        worst_r2_round = rounds[np.argmin(r2_scores)]
        best_acc_round = rounds[np.argmax(global_accuracies)]
        worst_acc_round = rounds[np.argmin(global_accuracies)]
        
        print(f"\n性能统计:")
        print(f"最佳拟合轮次 (R²): 轮次 {best_r2_round} (R² = {max(r2_scores):.4f})")
        print(f"最差拟合轮次 (R²): 轮次 {worst_r2_round} (R² = {min(r2_scores):.4f})")
        print(f"最高全局准确率: 轮次 {best_acc_round} ({max(global_accuracies):.4f})")
        print(f"最低全局准确率: 轮次 {worst_acc_round} ({min(global_accuracies):.4f})")
        
        return comparison_df
    
    def compare_predictions(self, test_ratios=[0.3, 0.5, 0.7, 0.9]):
        """
        比较不同轮次在相同保留比例下的预测结果
        
        Args:
            test_ratios: 测试的保留比例列表
        """
        if not self.round_models:
            raise ValueError("请先调用 load_all_rounds() 加载模型")
        
        rounds = sorted(self.round_models.keys())
        
        print(f"\n不同轮次的预测结果比较:")
        print("=" * 80)
        
        for ratio in test_ratios:
            print(f"\n保留比例 {ratio}:")
            print(f"{'轮次':<6} {'精度保留率':<12} {'预测准确率':<12} {'全局准确率':<12} {'准确率损失'}")
            print("-" * 70)
            
            retention_rates = []
            pred_accuracies = []
            
            for round_num in rounds:
                model_info = self.round_models[round_num]
                model = model_info['model']
                poly_features = model_info['poly_features']
                global_acc = model_info['global_accuracy']
                
                retention_rate = predict_retention_rate(ratio, model, poly_features)
                pred_accuracy = predict_absolute_accuracy(ratio, model, poly_features, global_acc)
                accuracy_loss = global_acc - pred_accuracy
                
                retention_rates.append(retention_rate)
                pred_accuracies.append(pred_accuracy)
                
                print(f"{round_num:<6} {retention_rate:<12.2f}% {pred_accuracy:<12.4f} "
                      f"{global_acc:<12.4f} {accuracy_loss:.4f}")
            
            # 统计信息
            print(f"\n保留比例 {ratio} 统计:")
            print(f"精度保留率: 平均 {np.mean(retention_rates):.2f}%, "
                  f"标准差 {np.std(retention_rates):.2f}%, "
                  f"范围 {np.min(retention_rates):.2f}%-{np.max(retention_rates):.2f}%")
            print(f"预测准确率: 平均 {np.mean(pred_accuracies):.4f}, "
                  f"标准差 {np.std(pred_accuracies):.4f}, "
                  f"范围 {np.min(pred_accuracies):.4f}-{np.max(pred_accuracies):.4f}")
    
    def plot_comparison(self, save_path=None):
        """绘制轮次比较图表"""
        if not self.round_models:
            raise ValueError("请先调用 load_all_rounds() 加载模型")
        
        rounds = sorted(self.round_models.keys())
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. R²分数比较
        r2_scores = [self.round_models[r]['stats']['r2_score'] for r in rounds]
        ax1.bar(rounds, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('R² 分数')
        ax1.set_title('不同轮次的模型拟合性能 (R²)')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (round_num, score) in enumerate(zip(rounds, r2_scores)):
            ax1.text(round_num, score + 0.01, f'{score:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        # 2. 全局准确率比较
        global_accs = [self.round_models[r]['global_accuracy'] for r in rounds]
        ax2.plot(rounds, global_accs, 'o-', color='green', linewidth=2, markersize=6)
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('全局准确率')
        ax2.set_title('不同轮次的全局模型准确率')
        ax2.grid(True, alpha=0.3)
        
        # 3. 预测结果比较 (保留比例0.5)
        test_ratio = 0.5
        retention_rates = []
        pred_accuracies = []
        
        for round_num in rounds:
            model_info = self.round_models[round_num]
            model = model_info['model']
            poly_features = model_info['poly_features']
            global_acc = model_info['global_accuracy']
            
            retention_rate = predict_retention_rate(test_ratio, model, poly_features)
            pred_accuracy = predict_absolute_accuracy(test_ratio, model, poly_features, global_acc)
            
            retention_rates.append(retention_rate)
            pred_accuracies.append(pred_accuracy)
        
        ax3.bar(rounds, retention_rates, color='orange', alpha=0.7)
        ax3.set_xlabel('轮次')
        ax3.set_ylabel('精度保留率 (%)')
        ax3.set_title(f'保留比例 {test_ratio} 下的精度保留率')
        ax3.grid(True, alpha=0.3)
        
        # 4. 预测准确率比较
        ax4.plot(rounds, pred_accuracies, 's-', color='red', linewidth=2, markersize=6)
        ax4.set_xlabel('轮次')
        ax4.set_ylabel('预测准确率')
        ax4.set_title(f'保留比例 {test_ratio} 下的预测准确率')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"比较图表已保存到: {save_path}")
        
        plt.show()
    
    def find_best_round_for_ratio(self, target_ratio):
        """
        找到在指定保留比例下表现最好的轮次
        
        Args:
            target_ratio: 目标保留比例
        
        Returns:
            最佳轮次信息
        """
        if not self.round_models:
            raise ValueError("请先调用 load_all_rounds() 加载模型")
        
        rounds = sorted(self.round_models.keys())
        results = []
        
        for round_num in rounds:
            model_info = self.round_models[round_num]
            model = model_info['model']
            poly_features = model_info['poly_features']
            global_acc = model_info['global_accuracy']
            
            retention_rate = predict_retention_rate(target_ratio, model, poly_features)
            pred_accuracy = predict_absolute_accuracy(target_ratio, model, poly_features, global_acc)
            
            results.append({
                'round': round_num,
                'retention_rate': retention_rate,
                'pred_accuracy': pred_accuracy,
                'global_accuracy': global_acc,
                'r2_score': model_info['stats']['r2_score']
            })
        
        # 按预测准确率排序
        results.sort(key=lambda x: x['pred_accuracy'], reverse=True)
        
        print(f"\n保留比例 {target_ratio} 下的轮次排名:")
        print(f"{'排名':<4} {'轮次':<6} {'精度保留率':<12} {'预测准确率':<12} {'全局准确率':<12} {'R²分数'}")
        print("-" * 70)
        
        for i, result in enumerate(results[:5], 1):  # 显示前5名
            print(f"{i:<4} {result['round']:<6} {result['retention_rate']:<12.2f}% "
                  f"{result['pred_accuracy']:<12.4f} {result['global_accuracy']:<12.4f} "
                  f"{result['r2_score']:.4f}")
        
        return results[0]  # 返回最佳轮次
    
    def analyze_round_stability(self):
        """分析轮次间的稳定性"""
        if not self.round_models:
            raise ValueError("请先调用 load_all_rounds() 加载模型")
        
        rounds = sorted(self.round_models.keys())
        
        # 收集各轮次的统计数据
        r2_scores = [self.round_models[r]['stats']['r2_score'] for r in rounds]
        global_accs = [self.round_models[r]['global_accuracy'] for r in rounds]
        
        # 计算稳定性指标
        r2_std = np.std(r2_scores)
        r2_cv = r2_std / np.mean(r2_scores)  # 变异系数
        
        acc_std = np.std(global_accs)
        acc_cv = acc_std / np.mean(global_accs)
        
        print(f"\n轮次稳定性分析:")
        print("=" * 50)
        print(f"R²分数:")
        print(f"  平均值: {np.mean(r2_scores):.4f}")
        print(f"  标准差: {r2_std:.4f}")
        print(f"  变异系数: {r2_cv:.4f}")
        print(f"  范围: {np.min(r2_scores):.4f} - {np.max(r2_scores):.4f}")
        
        print(f"\n全局准确率:")
        print(f"  平均值: {np.mean(global_accs):.4f}")
        print(f"  标准差: {acc_std:.4f}")
        print(f"  变异系数: {acc_cv:.4f}")
        print(f"  范围: {np.min(global_accs):.4f} - {np.max(global_accs):.4f}")
        
        # 稳定性评估
        if r2_cv < 0.05:
            r2_stability = "非常稳定"
        elif r2_cv < 0.1:
            r2_stability = "稳定"
        elif r2_cv < 0.2:
            r2_stability = "中等稳定"
        else:
            r2_stability = "不稳定"
        
        if acc_cv < 0.05:
            acc_stability = "非常稳定"
        elif acc_cv < 0.1:
            acc_stability = "稳定"
        elif acc_cv < 0.2:
            acc_stability = "中等稳定"
        else:
            acc_stability = "不稳定"
        
        print(f"\n稳定性评估:")
        print(f"模型拟合稳定性: {r2_stability}")
        print(f"全局准确率稳定性: {acc_stability}")

def main():
    """主函数 - 演示使用"""
    print("轮次比较分析工具")
    print("=" * 50)
    
    try:
        # 创建比较器
        comparator = RoundComparator('docs/accs.csv')
        
        print(f"数据文件包含 {comparator.total_rounds} 个轮次")
        
        # 选择要比较的轮次
        choice = input("\n选择比较模式:\n1. 比较所有轮次\n2. 比较指定轮次\n请输入选择 (1/2): ")
        
        if choice == '1':
            # 比较所有轮次
            comparator.load_all_rounds()
        elif choice == '2':
            # 比较指定轮次
            round_input = input(f"请输入要比较的轮次 (用逗号分隔，如: 1,5,10): ")
            try:
                round_list = [int(x.strip()) for x in round_input.split(',')]
                round_list = [r for r in round_list if 1 <= r <= comparator.total_rounds]
                if not round_list:
                    print("没有有效的轮次，使用所有轮次")
                    comparator.load_all_rounds()
                else:
                    comparator.load_all_rounds(round_list)
            except ValueError:
                print("输入格式错误，使用所有轮次")
                comparator.load_all_rounds()
        else:
            print("无效选择，使用所有轮次")
            comparator.load_all_rounds()
        
        # 执行比较分析
        print("\n1. 模型性能比较:")
        comparator.compare_model_performance()
        
        print("\n2. 预测结果比较:")
        comparator.compare_predictions()
        
        print("\n3. 稳定性分析:")
        comparator.analyze_round_stability()
        
        # 寻找最佳轮次
        test_ratio = 0.5
        print(f"\n4. 最佳轮次分析 (保留比例 {test_ratio}):")
        best_round = comparator.find_best_round_for_ratio(test_ratio)
        print(f"\n推荐使用轮次 {best_round['round']} 进行预测")
        
        # 绘制比较图表
        comparator.plot_comparison('round_comparison_results.png')
        
    except FileNotFoundError:
        print("错误: 找不到 docs/accs.csv 文件")
    except Exception as e:
        print(f"运行错误: {e}")

if __name__ == "__main__":
    main()