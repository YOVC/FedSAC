"""
联邦学习精度保留率预测器
输入：当前模型的保留比例
输出：子模型应该保留全局模型精度的百分之多少
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class AccuracyRetentionPredictor:
    """
    基于保留比例预测子模型精度保留率的类
    精度保留率 = (子模型准确率 / 全局模型准确率) * 100%
    """
    
    def __init__(self, csv_file_path):
        """
        初始化预测器
        
        Args:
            csv_file_path: CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.retention_ratios = None
        self.accuracy_retention_rates = None
        self.global_model_accuracies = None
        self.models = {}
        self.load_data()
    
    def load_data(self):
        """
        从CSV文件加载数据并计算精度保留率
        """
        # 读取CSV文件
        data = pd.read_csv(self.csv_file_path, header=None)
        
        # 第一行是保留比例
        self.retention_ratios = data.iloc[0].values
        
        # 后续行是准确率数据
        accuracy_data = data.iloc[1:].values
        
        # 找到保留比例为1.0的索引（全局模型）
        global_ratio_idx = np.where(self.retention_ratios == 1.0)[0][0]
        
        # 提取每轮的全局模型准确率
        self.global_model_accuracies = accuracy_data[:, global_ratio_idx]
        
        # 计算每个保留比例下的精度保留率
        retention_rates_data = []
        for i, row in enumerate(accuracy_data):
            global_acc = self.global_model_accuracies[i]
            # 计算每个保留比例下的精度保留率
            retention_rates = (row / global_acc) * 100
            retention_rates_data.append(retention_rates)
        
        retention_rates_data = np.array(retention_rates_data)
        
        # 将所有精度保留率数据展平，用于训练
        self.accuracy_retention_rates = retention_rates_data.flatten()
        
        # 重复保留比例以匹配精度保留率数据的长度
        self.retention_ratios_expanded = np.tile(self.retention_ratios, len(accuracy_data))
        
        print(f"数据加载完成:")
        print(f"保留比例范围: {self.retention_ratios.min():.2f} - {self.retention_ratios.max():.2f}")
        print(f"精度保留率范围: {self.accuracy_retention_rates.min():.2f}% - {self.accuracy_retention_rates.max():.2f}%")
        print(f"全局模型准确率范围: {self.global_model_accuracies.min():.4f} - {self.global_model_accuracies.max():.4f}")
        print(f"总数据点数: {len(self.accuracy_retention_rates)}")
    
    def polynomial_fit(self, degree=3):
        """
        多项式拟合
        
        Args:
            degree: 多项式次数
        """
        # 创建多项式特征
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(self.retention_ratios_expanded.reshape(-1, 1))
        
        # 线性回归
        model = LinearRegression()
        model.fit(X_poly, self.accuracy_retention_rates)
        
        # 预测
        y_pred = model.predict(X_poly)
        
        # 评估
        r2 = r2_score(self.accuracy_retention_rates, y_pred)
        mse = mean_squared_error(self.accuracy_retention_rates, y_pred)
        
        self.models[f'polynomial_{degree}'] = {
            'model': model,
            'poly_features': poly_features,
            'r2': r2,
            'mse': mse,
            'type': 'polynomial'
        }
        
        print(f"多项式拟合 (次数={degree}): R² = {r2:.4f}, MSE = {mse:.6f}")
        
        return model, poly_features
    
    def sigmoid_fit(self):
        """
        Sigmoid函数拟合: f(x) = a / (1 + exp(-b * (x - c))) + d
        """
        def sigmoid_func(x, a, b, c, d):
            return a / (1 + np.exp(-b * (x - c))) + d
        
        try:
            # 初始参数猜测
            popt, _ = curve_fit(sigmoid_func, self.retention_ratios_expanded, self.accuracy_retention_rates,
                              p0=[80.0, 10.0, 0.6, 20.0], maxfev=5000)
            
            # 预测
            y_pred = sigmoid_func(self.retention_ratios_expanded, *popt)
            
            # 评估
            r2 = r2_score(self.accuracy_retention_rates, y_pred)
            mse = mean_squared_error(self.accuracy_retention_rates, y_pred)
            
            self.models['sigmoid'] = {
                'params': popt,
                'function': sigmoid_func,
                'r2': r2,
                'mse': mse,
                'type': 'sigmoid'
            }
            
            print(f"Sigmoid拟合: R² = {r2:.4f}, MSE = {mse:.6f}")
            print(f"参数: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}, d={popt[3]:.4f}")
            
        except Exception as e:
            print(f"Sigmoid拟合失败: {e}")
    
    def power_fit(self):
        """
        幂函数拟合: f(x) = a * x^b + c
        """
        def power_func(x, a, b, c):
            return a * np.power(x, b) + c
        
        try:
            # 初始参数猜测
            popt, _ = curve_fit(power_func, self.retention_ratios_expanded, self.accuracy_retention_rates,
                              p0=[80.0, 2.0, 10.0], maxfev=5000)
            
            # 预测
            y_pred = power_func(self.retention_ratios_expanded, *popt)
            
            # 评估
            r2 = r2_score(self.accuracy_retention_rates, y_pred)
            mse = mean_squared_error(self.accuracy_retention_rates, y_pred)
            
            self.models['power'] = {
                'params': popt,
                'function': power_func,
                'r2': r2,
                'mse': mse,
                'type': 'power'
            }
            
            print(f"幂函数拟合: R² = {r2:.4f}, MSE = {mse:.6f}")
            print(f"参数: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
            
        except Exception as e:
            print(f"幂函数拟合失败: {e}")
    
    def fit_all_models(self):
        """
        拟合所有模型
        """
        print("开始拟合所有模型...")
        print("=" * 50)
        
        # 多项式拟合（不同次数）
        for degree in [2, 3, 4, 5]:
            self.polynomial_fit(degree)
        
        # 非线性拟合
        self.sigmoid_fit()
        self.power_fit()
        
        print("=" * 50)
        
        # 找到最佳模型
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        best_model = self.models[best_model_name]
        
        print(f"最佳模型: {best_model_name}")
        print(f"最佳R²: {best_model['r2']:.4f}")
        print(f"最佳MSE: {best_model['mse']:.6f}")
        
        return best_model_name
    
    def predict_retention_rate(self, retention_ratio, model_name=None):
        """
        预测给定保留比例的精度保留率
        
        Args:
            retention_ratio: 保留比例 (0-1之间的值或数组)
            model_name: 使用的模型名称，如果为None则使用最佳模型
        
        Returns:
            预测的精度保留率（百分比）
        """
        if model_name is None:
            model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model_info = self.models[model_name]
        
        # 确保输入是numpy数组
        retention_ratio = np.array(retention_ratio)
        
        if model_info['type'] == 'polynomial':
            X_poly = model_info['poly_features'].transform(retention_ratio.reshape(-1, 1))
            prediction = model_info['model'].predict(X_poly)
        else:
            prediction = model_info['function'](retention_ratio, *model_info['params'])
        
        # 确保预测值在合理范围内（0-100%）
        prediction = np.clip(prediction, 0, 100)
        
        return prediction
    
    def predict_absolute_accuracy(self, retention_ratio, global_accuracy, model_name=None):
        """
        预测给定保留比例和全局模型准确率下的绝对准确率
        
        Args:
            retention_ratio: 保留比例
            global_accuracy: 全局模型准确率
            model_name: 使用的模型名称
        
        Returns:
            预测的绝对准确率
        """
        retention_rate = self.predict_retention_rate(retention_ratio, model_name)
        absolute_accuracy = (retention_rate / 100) * global_accuracy
        return absolute_accuracy
    
    def plot_results(self, save_path=None):
        """
        绘制拟合结果
        
        Args:
            save_path: 保存图片的路径，如果为None则显示图片
        """
        plt.figure(figsize=(15, 10))
        
        # 创建平滑的x轴用于绘制拟合曲线
        x_smooth = np.linspace(self.retention_ratios.min(), self.retention_ratios.max(), 200)
        
        # 绘制原始数据点
        plt.scatter(self.retention_ratios_expanded, self.accuracy_retention_rates, 
                   alpha=0.3, s=10, color='gray', label='原始数据')
        
        # 计算每个保留比例的平均精度保留率
        unique_ratios = np.unique(self.retention_ratios)
        mean_retention_rates = []
        for ratio in unique_ratios:
            mask = self.retention_ratios_expanded == ratio
            mean_rate = np.mean(self.accuracy_retention_rates[mask])
            mean_retention_rates.append(mean_rate)
        
        plt.scatter(unique_ratios, mean_retention_rates, 
                   color='red', s=30, label='平均精度保留率', zorder=5)
        
        # 绘制不同模型的拟合曲线
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        color_idx = 0
        
        for model_name, model_info in self.models.items():
            if color_idx >= len(colors):
                break
                
            try:
                y_smooth = self.predict_retention_rate(x_smooth, model_name)
                plt.plot(x_smooth, y_smooth, 
                        color=colors[color_idx], linewidth=2,
                        label=f'{model_name} (R²={model_info["r2"]:.3f})')
                color_idx += 1
            except:
                continue
        
        plt.xlabel('保留比例', fontsize=12)
        plt.ylabel('精度保留率 (%)', fontsize=12)
        plt.title('子模型精度保留率与保留比例的关系拟合', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)  # 设置y轴范围为0-105%
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        else:
            plt.show()
    
    def get_model_summary(self):
        """
        获取所有模型的性能总结
        """
        summary = []
        for model_name, model_info in self.models.items():
            summary.append({
                'Model': model_name,
                'R²': model_info['r2'],
                'MSE': model_info['mse'],
                'Type': model_info['type']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('R²', ascending=False)
        
        return summary_df

def main():
    """
    主函数
    """
    # 创建预测器
    predictor = AccuracyRetentionPredictor('docs/accs.csv')
    
    # 拟合所有模型
    best_model = predictor.fit_all_models()
    
    # 显示模型性能总结
    print("\n模型性能总结:")
    print(predictor.get_model_summary())
    
    # 示例预测
    print(f"\n使用最佳模型 ({best_model}) 进行预测:")
    print("=" * 60)
    print(f"{'保留比例':<10} {'精度保留率':<12} {'说明':<30}")
    print("-" * 60)
    
    test_ratios = [0.25, 0.3, 0.5, 0.7, 0.9, 1.0]
    for ratio in test_ratios:
        retention_rate = predictor.predict_retention_rate(ratio)[0]
        if ratio == 1.0:
            description = "全局模型（基准）"
        elif retention_rate >= 80:
            description = "高精度保留"
        elif retention_rate >= 60:
            description = "中等精度保留"
        elif retention_rate >= 40:
            description = "低精度保留"
        else:
            description = "极低精度保留"
            
        print(f"{ratio:<10.2f} {retention_rate:<12.2f}% {description:<30}")
    
    # 实际应用示例
    print(f"\n实际应用示例:")
    print("=" * 60)
    global_acc = 0.65  # 假设全局模型准确率为65%
    print(f"假设全局模型准确率为 {global_acc:.1%}")
    print(f"{'保留比例':<10} {'精度保留率':<12} {'预测准确率':<12} {'绝对损失':<12}")
    print("-" * 50)
    
    for ratio in [0.3, 0.5, 0.7, 0.9]:
        retention_rate = predictor.predict_retention_rate(ratio)[0]
        pred_acc = predictor.predict_absolute_accuracy(ratio, global_acc)[0]
        loss = global_acc - pred_acc
        print(f"{ratio:<10.2f} {retention_rate:<12.2f}% {pred_acc:<12.4f} {loss:<12.4f}")
    
    # 绘制结果
    predictor.plot_results('accuracy_retention_results.png')
    
    return predictor

if __name__ == "__main__":
    predictor = main()