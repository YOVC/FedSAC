import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd

# 读取数据
def load_data():
    """从CSV文件加载MACs和时间数据"""
    with open('docs/macs-time.csv', 'r') as f:
        lines = f.readlines()
    
    # 解析第一行（MACs值）
    macs_str = lines[0].strip().split(',')
    macs = [float(x.strip()) for x in macs_str]
    
    # 解析第二行（时间值）
    time_str = lines[1].strip().split(',')
    time = [float(x.strip()) for x in time_str]
    
    return np.array(macs), np.array(time)

# 定义多种拟合函数
def linear_func(x, a, b):
    """线性函数: y = ax + b"""
    return a * x + b

def quadratic_func(x, a, b, c):
    """二次函数: y = ax² + bx + c"""
    return a * x**2 + b * x + c

def exponential_func(x, a, b, c):
    """指数函数: y = a * exp(bx) + c"""
    return a * np.exp(b * x) + c

def power_func(x, a, b, c):
    """幂函数: y = a * x^b + c"""
    return a * np.power(x, b) + c

def logarithmic_func(x, a, b, c):
    """对数函数: y = a * log(bx) + c"""
    return a * np.log(b * x) + c

# 拟合函数并评估
def fit_and_evaluate(macs, time):
    """拟合多种函数并评估性能"""
    functions = {
        'Linear': (linear_func, 2),
        'Quadratic': (quadratic_func, 3),
        'Exponential': (exponential_func, 3),
        'Power': (power_func, 3),
        'Logarithmic': (logarithmic_func, 3)
    }
    
    results = {}
    
    for name, (func, param_count) in functions.items():
        try:
            # 拟合函数
            if name == 'Exponential':
                # 指数函数需要特殊的初始猜测
                popt, pcov = curve_fit(func, macs, time, p0=[1, 0.01, 10], maxfev=5000)
            elif name == 'Power':
                # 幂函数需要特殊的初始猜测
                popt, pcov = curve_fit(func, macs, time, p0=[1, 1, 0], maxfev=5000)
            elif name == 'Logarithmic':
                # 对数函数需要特殊的初始猜测
                popt, pcov = curve_fit(func, macs, time, p0=[10, 1, 0], maxfev=5000)
            else:
                popt, pcov = curve_fit(func, macs, time, maxfev=5000)
            
            # 预测值
            y_pred = func(macs, *popt)
            
            # 计算R²
            r2 = r2_score(time, y_pred)
            
            # 计算均方根误差
            rmse = np.sqrt(np.mean((time - y_pred)**2))
            
            results[name] = {
                'params': popt,
                'r2': r2,
                'rmse': rmse,
                'func': func,
                'y_pred': y_pred
            }
            
        except Exception as e:
            print(f"拟合 {name} 函数时出错: {e}")
            continue
    
    return results

# 可视化结果
def plot_results(macs, time, results):
    """绘制原始数据和拟合结果"""
    plt.figure(figsize=(15, 10))
    
    # 创建更密集的x值用于绘制平滑曲线
    x_smooth = np.linspace(macs.min(), macs.max(), 100)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(2, 3, i+1)
        
        # 绘制原始数据点
        plt.scatter(macs, time, color='black', s=50, alpha=0.7, label='原始数据')
        
        # 绘制拟合曲线
        try:
            y_smooth = result['func'](x_smooth, *result['params'])
            plt.plot(x_smooth, y_smooth, color=colors[i], linewidth=2, 
                    label=f'{name} (R²={result["r2"]:.4f})')
        except:
            # 如果无法绘制平滑曲线，就绘制数据点
            plt.plot(macs, result['y_pred'], 'o-', color=colors[i], 
                    label=f'{name} (R²={result["r2"]:.4f})')
        
        plt.xlabel('MACs')
        plt.ylabel('训练时间')
        plt.title(f'{name} 拟合')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 比较所有模型
    plt.subplot(2, 3, 6)
    plt.scatter(macs, time, color='black', s=50, alpha=0.7, label='原始数据')
    
    for i, (name, result) in enumerate(results.items()):
        try:
            y_smooth = result['func'](x_smooth, *result['params'])
            plt.plot(x_smooth, y_smooth, color=colors[i], linewidth=2, 
                    label=f'{name} (R²={result["r2"]:.4f})')
        except:
            plt.plot(macs, result['y_pred'], 'o-', color=colors[i], 
                    label=f'{name} (R²={result["r2"]:.4f})')
    
    plt.xlabel('MACs')
    plt.ylabel('训练时间')
    plt.title('所有模型比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/macs_time_fitting_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 生成预测函数
def generate_prediction_function(best_model_name, best_params, best_func):
    """生成可用的预测函数"""
    print(f"\n最佳拟合模型: {best_model_name}")
    print(f"参数: {best_params}")
    
    if best_model_name == 'Linear':
        print(f"\n预测函数: time = {best_params[0]:.6f} * macs + {best_params[1]:.6f}")
        return lambda macs: best_params[0] * macs + best_params[1]
    
    elif best_model_name == 'Quadratic':
        print(f"\n预测函数: time = {best_params[0]:.6f} * macs² + {best_params[1]:.6f} * macs + {best_params[2]:.6f}")
        return lambda macs: best_params[0] * macs**2 + best_params[1] * macs + best_params[2]
    
    elif best_model_name == 'Exponential':
        print(f"\n预测函数: time = {best_params[0]:.6f} * exp({best_params[1]:.6f} * macs) + {best_params[2]:.6f}")
        return lambda macs: best_params[0] * np.exp(best_params[1] * macs) + best_params[2]
    
    elif best_model_name == 'Power':
        print(f"\n预测函数: time = {best_params[0]:.6f} * macs^{best_params[1]:.6f} + {best_params[2]:.6f}")
        return lambda macs: best_params[0] * np.power(macs, best_params[1]) + best_params[2]
    
    elif best_model_name == 'Logarithmic':
        print(f"\n预测函数: time = {best_params[0]:.6f} * log({best_params[1]:.6f} * macs) + {best_params[2]:.6f}")
        return lambda macs: best_params[0] * np.log(best_params[1] * macs) + best_params[2]

def main():
    """主函数"""
    print("MACs到训练时间的函数拟合分析")
    print("="*50)
    
    # 加载数据
    macs, time = load_data()
    print(f"数据点数量: {len(macs)}")
    print(f"MACs范围: {macs.min():.2f} - {macs.max():.2f}")
    print(f"时间范围: {time.min():.2f} - {time.max():.2f}")
    
    # 显示数据
    print("\n原始数据:")
    for i in range(len(macs)):
        print(f"MACs: {macs[i]:6.2f}, 时间: {time[i]:6.2f}")
    
    # 拟合函数
    print("\n开始拟合函数...")
    results = fit_and_evaluate(macs, time)
    
    # 显示结果
    print("\n拟合结果:")
    print("-"*60)
    print(f"{'模型':<12} {'R²':<10} {'RMSE':<10} {'参数'}")
    print("-"*60)
    
    best_r2 = -1
    best_model = None
    best_name = None
    
    for name, result in results.items():
        print(f"{name:<12} {result['r2']:<10.6f} {result['rmse']:<10.4f} {result['params']}")
        
        if result['r2'] > best_r2:
            best_r2 = result['r2']
            best_model = result
            best_name = name
    
    # 生成预测函数
    if best_model:
        predict_func = generate_prediction_function(best_name, best_model['params'], best_model['func'])
        
        # 测试预测函数
        print(f"\n预测函数测试:")
        test_macs = [15, 25, 35, 45, 55]
        for test_mac in test_macs:
            predicted_time = predict_func(test_mac)
            print(f"MACs = {test_mac}, 预测时间 = {predicted_time:.2f}")
    
    # 绘制结果
    plot_results(macs, time, results)
    
    return predict_func if best_model else None

if __name__ == "__main__":
    prediction_function = main()