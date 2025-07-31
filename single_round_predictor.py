#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单轮次精度保留率预测器
根据指定轮次的数据拟合函数，预测子模型保留全局模型精度的百分比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SingleRoundRetentionPredictor:
    """单轮次精度保留率预测器"""
    
    def __init__(self, csv_file='docs/accs.csv'):
        """
        初始化预测器
        
        Args:
            csv_file: CSV数据文件路径
        """
        self.csv_file = csv_file
        self.retention_ratios = None
        self.sub_accuracies = None
        self.global_accuracies = None
        self.retention_rates = None
        self.models = {}
        self.round_data = {}
        
    def load_data(self):
        """加载CSV数据"""
        try:
            # 读取CSV文件
            data = pd.read_csv(self.csv_file, header=None)
            
            # 第一行是保留比例
            self.retention_ratios = data.iloc[0].values
            
            # 后续行是不同轮次的准确率数据
            accuracy_data = data.iloc[1:].values
            
            # 存储每轮数据
            for round_idx in range(accuracy_data.shape[0]):
                round_num = round_idx + 1
                sub_accs = accuracy_data[round_idx]
                global_acc = sub_accs[-1]  # 保留比例1.0对应的准确率
                
                # 计算精度保留率
                retention_rates = (sub_accs / global_acc) * 100
                
                self.round_data[round_num] = {
                    'retention_ratios': self.retention_ratios,
                    'sub_accuracies': sub_accs,
                    'global_accuracy': global_acc,
                    'retention_rates': retention_rates
                }
            
            print(f"数据加载成功！")
            print(f"保留比例范围: {self.retention_ratios.min():.2f} - {self.retention_ratios.max():.2f}")
            print(f"总轮次数: {len(self.round_data)}")
            
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def fit_round_models(self, round_num):
        """
        为指定轮次拟合模型
        
        Args:
            round_num: 轮次编号 (1-15)
        """
        if round_num not in self.round_data:
            raise ValueError(f"轮次 {round_num} 不存在，可用轮次: 1-{len(self.round_data)}")
        
        round_data = self.round_data[round_num]
        X = round_data['retention_ratios'].reshape(-1, 1)
        y = round_data['retention_rates']
        
        # 存储当前轮次数据
        self.current_round = round_num
        self.retention_ratios = round_data['retention_ratios']
        self.retention_rates = round_data['retention_rates']
        self.global_accuracy = round_data['global_accuracy']
        self.sub_accuracies = round_data['sub_accuracies']
        
        models = {}
        
        # 1. 多项式拟合 (2-5次)
        for degree in range(2, 6):
            poly_model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            poly_model.fit(X, y)
            
            y_pred = poly_model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            models[f'polynomial_{degree}'] = {
                'model': poly_model,
                'r2': r2,
                'mse': mse,
                'type': 'polynomial'
            }
        
        # 2. 指数拟合 (转换为线性)
        try:
            # y = a * exp(b * x) -> ln(y) = ln(a) + b * x
            y_log = np.log(np.maximum(y, 1e-10))  # 避免log(0)
            exp_model = LinearRegression()
            exp_model.fit(X, y_log)
            
            y_pred_log = exp_model.predict(X)
            y_pred = np.exp(y_pred_log)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            models['exponential'] = {
                'model': exp_model,
                'r2': r2,
                'mse': mse,
                'type': 'exponential'
            }
        except:
            pass
        
        # 3. 幂函数拟合
        try:
            # y = a * x^b -> ln(y) = ln(a) + b * ln(x)
            X_log = np.log(np.maximum(X.flatten(), 1e-10)).reshape(-1, 1)
            y_log = np.log(np.maximum(y, 1e-10))
            power_model = LinearRegression()
            power_model.fit(X_log, y_log)
            
            y_pred_log = power_model.predict(X_log)
            y_pred = np.exp(y_pred_log)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            models['power'] = {
                'model': power_model,
                'r2': r2,
                'mse': mse,
                'type': 'power'
            }
        except:
            pass
        
        self.models = models
        
        # 找到最佳模型
        best_model_name = max(models.keys(), key=lambda k: models[k]['r2'])
        self.best_model_name = best_model_name
        self.best_model = models[best_model_name]
        
        print(f"\n轮次 {round_num} 模型拟合完成:")
        print(f"全局模型准确率: {self.global_accuracy:.4f}")
        print(f"精度保留率范围: {y.min():.2f}% - {y.max():.2f}%")
        print(f"数据点数: {len(X)}")
        
        print(f"\n模型性能对比:")
        for name, model_info in models.items():
            print(f"{name}: R² = {model_info['r2']:.4f}, MSE = {model_info['mse']:.6f}")
        
        print(f"\n最佳模型: {best_model_name} (R² = {self.best_model['r2']:.4f})")
    
    def predict_retention_rate(self, retention_ratios, model_name=None):
        """
        预测精度保留率
        
        Args:
            retention_ratios: 保留比例，可以是单个值或数组
            model_name: 模型名称，默认使用最佳模型
        
        Returns:
            预测的精度保留率
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("请先调用 fit_round_models() 拟合模型")
        
        model_name = model_name or self.best_model_name
        model_info = self.models[model_name]
        model = model_info['model']
        model_type = model_info['type']
        
        # 确保输入是数组
        ratios = np.array(retention_ratios).reshape(-1, 1)
        
        if model_type == 'polynomial':
            predictions = model.predict(ratios)
        elif model_type == 'exponential':
            predictions = np.exp(model.predict(ratios))
        elif model_type == 'power':
            ratios_log = np.log(np.maximum(ratios, 1e-10))
            predictions = np.exp(model.predict(ratios_log))
        else:
            predictions = model.predict(ratios)
        
        # 确保预测值在合理范围内
        predictions = np.clip(predictions, 0, 100)
        
        return predictions.flatten() if len(predictions) > 1 else predictions[0]
    
    def predict_absolute_accuracy(self, retention_ratios, model_name=None):
        """
        预测绝对准确率
        
        Args:
            retention_ratios: 保留比例
            model_name: 模型名称
        
        Returns:
            预测的绝对准确率
        """
        retention_rates = self.predict_retention_rate(retention_ratios, model_name)
        if isinstance(retention_rates, (int, float)):
            retention_rates = [retention_rates]
        
        absolute_accuracies = []
        for rate in retention_rates:
            abs_acc = (rate / 100) * self.global_accuracy
            absolute_accuracies.append(abs_acc)
        
        return np.array(absolute_accuracies).flatten() if len(absolute_accuracies) > 1 else absolute_accuracies[0]
    
    def plot_results(self, save_path=None):
        """绘制拟合结果"""
        if not hasattr(self, 'best_model'):
            raise ValueError("请先调用 fit_round_models() 拟合模型")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 生成预测曲线的x值
        x_smooth = np.linspace(self.retention_ratios.min(), self.retention_ratios.max(), 200)
        
        # 1. 精度保留率拟合
        ax1.scatter(self.retention_ratios, self.retention_rates, alpha=0.6, s=30, label='实际数据')
        y_smooth = self.predict_retention_rate(x_smooth)
        ax1.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'最佳拟合 ({self.best_model_name})')
        ax1.set_xlabel('保留比例')
        ax1.set_ylabel('精度保留率 (%)')
        ax1.set_title(f'轮次 {self.current_round} - 精度保留率拟合 (R² = {self.best_model["r2"]:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 绝对准确率对比
        ax2.scatter(self.retention_ratios, self.sub_accuracies, alpha=0.6, s=30, label='实际准确率')
        pred_accuracies = self.predict_absolute_accuracy(x_smooth)
        ax2.plot(x_smooth, pred_accuracies, 'g-', linewidth=2, label='预测准确率')
        ax2.set_xlabel('保留比例')
        ax2.set_ylabel('准确率')
        ax2.set_title(f'轮次 {self.current_round} - 绝对准确率对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 模型性能对比
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        colors = ['red' if name == self.best_model_name else 'skyblue' for name in model_names]
        
        bars = ax3.bar(range(len(model_names)), r2_scores, color=colors)
        ax3.set_xlabel('模型')
        ax3.set_ylabel('R² 分数')
        ax3.set_title('模型性能对比 (R²)')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 4. 残差分析
        y_pred = self.predict_retention_rate(self.retention_ratios)
        residuals = self.retention_rates - y_pred
        ax4.scatter(y_pred, residuals, alpha=0.6, s=30)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('预测值')
        ax4.set_ylabel('残差')
        ax4.set_title('残差分析')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def get_round_summary(self):
        """获取当前轮次的总结信息"""
        if not hasattr(self, 'best_model'):
            raise ValueError("请先调用 fit_round_models() 拟合模型")
        
        summary = {
            'round_number': self.current_round,
            'global_accuracy': self.global_accuracy,
            'retention_rate_range': (self.retention_rates.min(), self.retention_rates.max()),
            'best_model': self.best_model_name,
            'best_r2': self.best_model['r2'],
            'best_mse': self.best_model['mse'],
            'data_points': len(self.retention_ratios)
        }
        
        return summary

def load_and_fit_single_round_model(csv_file, round_num):
    """
    便捷函数：加载数据并拟合指定轮次的模型
    
    Args:
        csv_file: CSV文件路径
        round_num: 轮次编号
    
    Returns:
        预测器实例
    """
    predictor = SingleRoundRetentionPredictor(csv_file)
    predictor.load_data()
    predictor.fit_round_models(round_num)
    return predictor

def predict_single_round_retention(retention_ratios, predictor):
    """
    便捷函数：预测精度保留率
    
    Args:
        retention_ratios: 保留比例
        predictor: 预测器实例
    
    Returns:
        精度保留率
    """
    return predictor.predict_retention_rate(retention_ratios)

def main():
    """主函数 - 演示使用"""
    print("单轮次精度保留率预测器")
    print("=" * 50)
    
    try:
        # 加载数据
        predictor = SingleRoundRetentionPredictor('docs/accs.csv')
        if not predictor.load_data():
            return
        
        # 显示可用轮次
        print(f"\n可用轮次: 1-{len(predictor.round_data)}")
        
        # 让用户选择轮次
        while True:
            try:
                round_input = input(f"\n请输入要分析的轮次 (1-{len(predictor.round_data)}): ")
                round_num = int(round_input)
                if 1 <= round_num <= len(predictor.round_data):
                    break
                else:
                    print(f"请输入 1-{len(predictor.round_data)} 之间的数字")
            except ValueError:
                print("请输入有效的数字")
        
        # 拟合模型
        predictor.fit_round_models(round_num)
        
        # 显示预测示例
        print(f"\n轮次 {round_num} 预测示例:")
        test_ratios = [0.3, 0.5, 0.7, 0.9]
        retention_rates = predictor.predict_retention_rate(test_ratios)
        abs_accuracies = predictor.predict_absolute_accuracy(test_ratios)
        
        print(f"{'保留比例':<8} {'精度保留率':<12} {'预测准确率':<12} {'准确率损失'}")
        print("-" * 50)
        for ratio, retention, accuracy in zip(test_ratios, retention_rates, abs_accuracies):
            loss = predictor.global_accuracy - accuracy
            print(f"{ratio:<8.1f} {retention:<12.2f}% {accuracy:<12.4f} {loss:.4f}")
        
        # 绘制结果
        predictor.plot_results(f'single_round_{round_num}_results.png')
        
        # 交互式预测
        print(f"\n交互式预测 (轮次 {round_num}):")
        while True:
            try:
                ratio_input = input("请输入保留比例 (0.25-1.0，输入 'q' 退出): ")
                if ratio_input.lower() == 'q':
                    break
                
                ratio = float(ratio_input)
                if 0.25 <= ratio <= 1.0:
                    retention_rate = predictor.predict_retention_rate(ratio)
                    abs_accuracy = predictor.predict_absolute_accuracy(ratio)
                    
                    print(f"保留比例: {ratio:.3f}")
                    print(f"精度保留率: {retention_rate:.2f}%")
                    print(f"预测准确率: {abs_accuracy:.4f} ({abs_accuracy*100:.2f}%)")
                    print(f"准确率损失: {predictor.global_accuracy - abs_accuracy:.4f}")
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