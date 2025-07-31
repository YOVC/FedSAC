import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class AccuracyPredictor:
    """
    基于保留比例预测子模型准确率的类
    """
    
    def __init__(self, csv_file_path):
        """
        初始化预测器
        
        Args:
            csv_file_path: CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.retention_ratios = None
        self.accuracies = None
        self.models = {}
        self.load_data()
    
    def load_data(self):
        """
        从CSV文件加载数据
        """
        # 读取CSV文件
        data = pd.read_csv(self.csv_file_path, header=None)
        
        # 第一行是保留比例
        self.retention_ratios = data.iloc[0].values
        
        # 后续行是准确率数据
        accuracy_data = data.iloc[1:].values
        
        # 将所有准确率数据展平，用于训练
        self.accuracies = accuracy_data.flatten()
        
        # 重复保留比例以匹配准确率数据的长度
        self.retention_ratios_expanded = np.tile(self.retention_ratios, len(accuracy_data))
        
        print(f"数据加载完成:")
        print(f"保留比例范围: {self.retention_ratios.min():.2f} - {self.retention_ratios.max():.2f}")
        print(f"准确率范围: {self.accuracies.min():.4f} - {self.accuracies.max():.4f}")
        print(f"总数据点数: {len(self.accuracies)}")
    
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
        model.fit(X_poly, self.accuracies)
        
        # 预测
        y_pred = model.predict(X_poly)
        
        # 评估
        r2 = r2_score(self.accuracies, y_pred)
        mse = mean_squared_error(self.accuracies, y_pred)
        
        self.models[f'polynomial_{degree}'] = {
            'model': model,
            'poly_features': poly_features,
            'r2': r2,
            'mse': mse,
            'type': 'polynomial'
        }
        
        print(f"多项式拟合 (次数={degree}): R² = {r2:.4f}, MSE = {mse:.6f}")
        
        return model, poly_features
    
    def exponential_fit(self):
        """
        指数函数拟合: f(x) = a * (1 - exp(-b * x)) + c
        """
        def exponential_func(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c
        
        try:
            # 初始参数猜测
            popt, _ = curve_fit(exponential_func, self.retention_ratios_expanded, self.accuracies,
                              p0=[0.6, 3.0, 0.1], maxfev=5000)
            
            # 预测
            y_pred = exponential_func(self.retention_ratios_expanded, *popt)
            
            # 评估
            r2 = r2_score(self.accuracies, y_pred)
            mse = mean_squared_error(self.accuracies, y_pred)
            
            self.models['exponential'] = {
                'params': popt,
                'function': exponential_func,
                'r2': r2,
                'mse': mse,
                'type': 'exponential'
            }
            
            print(f"指数拟合: R² = {r2:.4f}, MSE = {mse:.6f}")
            print(f"参数: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
            
        except Exception as e:
            print(f"指数拟合失败: {e}")
    
    def sigmoid_fit(self):
        """
        Sigmoid函数拟合: f(x) = a / (1 + exp(-b * (x - c))) + d
        """
        def sigmoid_func(x, a, b, c, d):
            return a / (1 + np.exp(-b * (x - c))) + d
        
        try:
            # 初始参数猜测
            popt, _ = curve_fit(sigmoid_func, self.retention_ratios_expanded, self.accuracies,
                              p0=[0.6, 10.0, 0.6, 0.1], maxfev=5000)
            
            # 预测
            y_pred = sigmoid_func(self.retention_ratios_expanded, *popt)
            
            # 评估
            r2 = r2_score(self.accuracies, y_pred)
            mse = mean_squared_error(self.accuracies, y_pred)
            
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
            popt, _ = curve_fit(power_func, self.retention_ratios_expanded, self.accuracies,
                              p0=[0.5, 2.0, 0.0], maxfev=5000)
            
            # 预测
            y_pred = power_func(self.retention_ratios_expanded, *popt)
            
            # 评估
            r2 = r2_score(self.accuracies, y_pred)
            mse = mean_squared_error(self.accuracies, y_pred)
            
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
        self.exponential_fit()
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
    
    def predict(self, retention_ratio, model_name=None):
        """
        预测给定保留比例的准确率
        
        Args:
            retention_ratio: 保留比例 (0-1之间的值或数组)
            model_name: 使用的模型名称，如果为None则使用最佳模型
        
        Returns:
            预测的准确率
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
        
        return prediction
    
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
        plt.scatter(self.retention_ratios_expanded, self.accuracies, 
                   alpha=0.3, s=10, color='gray', label='原始数据')
        
        # 计算每个保留比例的平均准确率
        unique_ratios = np.unique(self.retention_ratios)
        mean_accuracies = []
        for ratio in unique_ratios:
            mask = self.retention_ratios_expanded == ratio
            mean_acc = np.mean(self.accuracies[mask])
            mean_accuracies.append(mean_acc)
        
        plt.scatter(unique_ratios, mean_accuracies, 
                   color='red', s=30, label='平均准确率', zorder=5)
        
        # 绘制不同模型的拟合曲线
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan']
        color_idx = 0
        
        for model_name, model_info in self.models.items():
            if color_idx >= len(colors):
                break
                
            try:
                y_smooth = self.predict(x_smooth, model_name)
                plt.plot(x_smooth, y_smooth, 
                        color=colors[color_idx], linewidth=2,
                        label=f'{model_name} (R²={model_info["r2"]:.3f})')
                color_idx += 1
            except:
                continue
        
        plt.xlabel('保留比例', fontsize=12)
        plt.ylabel('准确率', fontsize=12)
        plt.title('子模型准确率与保留比例的关系拟合', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
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
    predictor = AccuracyPredictor('docs/accs.csv')
    
    # 拟合所有模型
    best_model = predictor.fit_all_models()
    
    # 显示模型性能总结
    print("\n模型性能总结:")
    print(predictor.get_model_summary())
    
    # 示例预测
    print(f"\n使用最佳模型 ({best_model}) 进行预测:")
    test_ratios = [0.3, 0.5, 0.7, 0.9]
    for ratio in test_ratios:
        pred_acc = predictor.predict(ratio)
        print(f"保留比例 {ratio}: 预测准确率 = {pred_acc[0]:.4f}")
    
    # 绘制结果
    predictor.plot_results('accuracy_prediction_results.png')
    return predictor

if __name__ == "__main__":
    predictor = main()
    # 预测
    pred_acc = predictor.predict(0.86)
    print(f"保留比例 0.86: 预测准确率 = {pred_acc[0]:.4f}")