import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 导入配置文件
try:
    from analysis_config import *
except ImportError:
    # 如果配置文件不存在，使用默认设置
    MAX_ROUNDS = 50
    FIGURE_WIDTH = 18
    FIGURE_HEIGHT = 6
    DPI = 300
    ALGORITHM_LABELS = {
        'aware.csv': 'Aware',
        'awareWeight.csv': 'AwareWeight', 
        'Grad.csv': 'Grad',
        'GradAndWeight.csv': 'GradAndWeight',
        'neuron.csv': 'Neuron'
    }
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    VERBOSE = True

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_csv_data(file_path):
    """加载CSV数据并处理缺失值"""
    try:
        df = pd.read_csv(file_path)
        # 删除包含NaN值的行
        df = df.dropna()
        
        # 如果设置了MAX_ROUNDS，则只使用前MAX_ROUNDS条数据
        if MAX_ROUNDS is not None:
            # 根据round列进行筛选，取前MAX_ROUNDS轮
            max_round = df['round'].min() + MAX_ROUNDS - 1
            df = df[df['round'] <= max_round]
            if VERBOSE:
                print(f"  限制使用前{MAX_ROUNDS}轮数据，实际使用{len(df)}条记录")
        
        return df
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def plot_accuracy_comparison():
    """绘制三个准确率对比图"""
    
    print(f"=== 数据分析设置 ===")
    if MAX_ROUNDS is not None:
        print(f"使用前 {MAX_ROUNDS} 轮数据进行分析")
    else:
        print("使用全部数据进行分析")
    print()
    
    # 使用配置文件中的算法标签
    files = ALGORITHM_LABELS
    
    # 数据存储
    data = {}
    
    # 加载所有数据
    for filename, label in files.items():
        file_path = os.path.join('docs', filename)
        df = load_csv_data(file_path)
        if df is not None:
            data[label] = df
            print(f"成功加载 {filename}: {len(df)} 轮数据")
        else:
            print(f"无法加载 {filename}")
    
    if not data:
        print("没有成功加载任何数据文件")
        return
    
    # 创建三个子图
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # 设置主标题，包含数据范围信息
    if MAX_ROUNDS is not None:
        main_title = f'联邦学习准确率对比分析 (前{MAX_ROUNDS}轮)'
    else:
        main_title = '联邦学习准确率对比分析 (全部数据)'
    
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # 使用配置文件中的颜色
    colors = COLORS
    
    # 准确率类型
    accuracy_types = [
        ('client_max_accuracy', '客户端最大准确率'),
        ('client_avg_accuracy', '客户端平均准确率'), 
        ('global_accuracy', '全局准确率')
    ]
    
    for idx, (acc_type, title) in enumerate(accuracy_types):
        ax = axes[idx]
        
        color_idx = 0
        for label, df in data.items():
            if acc_type in df.columns:
                # 获取轮数和准确率数据
                rounds = df['round'].values
                accuracy = df[acc_type].values
                
                # 绘制线条
                ax.plot(rounds, accuracy, 
                       label=label, 
                       color=colors[color_idx % len(colors)],
                       linewidth=2, 
                       marker='o', 
                       markersize=3,
                       alpha=0.8)
                color_idx += 1
        
        # 设置图表属性
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('训练轮数', fontsize=12)
        ax.set_ylabel('准确率', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # 设置y轴范围，使图表更清晰
        if data:
            all_values = []
            for label, df in data.items():
                if acc_type in df.columns:
                    all_values.extend(df[acc_type].values)
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                margin = (max_val - min_val) * 0.05
                ax.set_ylim(min_val - margin, max_val + margin)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片，文件名包含数据范围信息
    if MAX_ROUNDS is not None:
        output_path = f'accuracy_comparison_top{MAX_ROUNDS}rounds.png'
    else:
        output_path = 'accuracy_comparison_all_data.png'
    
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"图表已保存为: {output_path}")
    
    # 显示图表
    plt.show()
    
    # 打印数据统计信息
    print("\n=== 数据统计信息 ===")
    for label, df in data.items():
        print(f"\n{label}:")
        print(f"  轮数范围: {df['round'].min()} - {df['round'].max()}")
        for acc_type, title in accuracy_types:
            if acc_type in df.columns:
                values = df[acc_type].values
                print(f"  {title}: 最小值={values.min():.4f}, 最大值={values.max():.4f}, 平均值={values.mean():.4f}")

if __name__ == "__main__":
    # 确保在正确的目录下运行
    if not os.path.exists('docs'):
        print("错误: 找不到docs目录，请确保在FedAVE-main目录下运行此脚本")
    else:
        plot_accuracy_comparison()