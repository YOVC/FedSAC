# 联邦学习准确率分析配置文件
# Federal Learning Accuracy Analysis Configuration

# ===== 数据设置 =====
# 设置使用前多少条数据进行分析
# 可选值：
#   - 数字（如30, 50, 100）：使用前N轮数据
#   - None：使用全部数据
MAX_ROUNDS = 50

# ===== 图表设置 =====
# 图表尺寸设置
FIGURE_WIDTH = 18
FIGURE_HEIGHT = 6

# 图表DPI设置（影响图片清晰度）
DPI = 300

# ===== 算法标签设置 =====
# 可以修改算法在图表中显示的名称
ALGORITHM_LABELS = {
    'aware.csv': 'Aware',
    'weight.csv': 'Weight', 
    'Grad.csv': 'Grad',
    'GradAndWeight.csv': 'GradWeight',
    'neuron.csv': 'Neuron'
}

# ===== 颜色设置 =====
# 图表中各算法使用的颜色
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# ===== 输出设置 =====
# 是否显示详细的加载信息
VERBOSE = True

# 是否自动打开生成的图片
AUTO_OPEN_IMAGES = False

# ===== 使用说明 =====
"""
使用方法：
1. 修改上面的配置参数
2. 运行 python plot_accuracy_comparison.py 生成基础对比图

常用配置示例：
- 分析前30轮：MAX_ROUNDS = 30
- 分析前50轮：MAX_ROUNDS = 50  
- 分析全部数据：MAX_ROUNDS = None
"""