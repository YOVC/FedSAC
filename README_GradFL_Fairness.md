# GradFL算法公平性评估

本项目实现了GradFL算法的公平性评估功能，用于分析不同子模型生成策略和子模型比例对联邦学习系统公平性的影响。

## 公平性评估概述

在联邦学习中，公平性是指系统能够为所有参与方提供相似的性能收益，而不偏向特定的客户端。GradFL算法通过为不同客户端分配不同大小的子模型，可能会影响系统的公平性。本项目实现了两种公平性评估指标：

1. **客户端性能与独立训练性能的相关性**：衡量联邦学习模型在各客户端上的性能与这些客户端独立训练模型性能之间的相关性。相关系数越高，表示公平性越好。

2. **子模型比例与客户端性能的相关性**：分析子模型比例与客户端性能之间的关系，评估资源分配的公平性。这一指标特别适用于GradFL算法。

## 功能特点

- 支持多种子模型生成策略的公平性评估：
  - `awareGrad`：基于梯度的子模型生成
  - `roll`：滚动选择通道
  - `rand`：随机选择通道
  - `hetero`：基于固定顺序选择通道
  - `fedavg`：完整模型（FedAvg基准）

- 支持多种模型架构：
  - `conv`：简单卷积神经网络
  - `resnet`：残差网络
  - `vgg`：VGG网络

- 支持不同子模型比例的公平性评估

- 提供可视化工具，展示公平性指标随训练轮次的变化

## 使用方法

### 运行单次GradFL实验

```bash
python run_gradfl.py
```

这将使用默认参数运行GradFL算法，并在训练过程中计算和记录公平性指标。

### 运行公平性评估实验

```bash
python evaluate_fairness.py
```

这将对不同模型和子模型生成策略进行公平性评估，并生成比较图表。

### 自定义参数

您可以修改`run_gradfl.py`或`evaluate_fairness.py`中的参数来自定义实验：

```python
# 基本参数设置
task = 'cifar10_cnum10_dist0_skew0_seed0'  # 任务名称
algorithm = 'GradFL'  # 算法名称
model = 'cnn'  # 模型名称
gradfl_model = 'conv'  # GradFL使用的模型：conv, resnet, vgg

# GradFL特定参数
mode = 'awareGrad'  # 子模型生成策略：awareGrad, roll, rand, hetero, fedavg
select_mode = 'absmax'  # 选择模式：absmax, probs, absmin
client_state = 'fix'  # 客户端状态：fix, dynamic
submodel_rate = '0.3 0.5 0.7 0.9'  # 子模型比例列表
probs = '0.25 0.25 0.25 0.25'  # 各比例的概率
```

## 结果分析

运行评估脚本后，结果将保存在`fairness_results`目录中，包括：

- `fairness_results.json`：包含所有实验的公平性指标数据
- `fairness_comparison.png`：不同模型和子模型生成策略的公平性比较图
- `fairness_over_rounds_<model>.png`：公平性指标随训练轮次变化的曲线图

## 公平性指标解释

- **皮尔逊相关系数**：衡量两个变量之间的线性相关性，取值范围为[-1, 1]。
  - 接近1：强正相关，表示公平性好
  - 接近0：无相关性，表示公平性差
  - 接近-1：强负相关，表示公平性差且存在反向关系

- **p值**：统计显著性指标，通常p < 0.05表示相关性具有统计学意义。

## 代码结构

- `algorithm/GradFL.py`：GradFL算法实现，包含公平性评估相关代码
- `main.py`：主程序，包含日志记录和公平性指标保存功能
- `run_gradfl.py`：运行单次GradFL实验的脚本
- `evaluate_fairness.py`：公平性评估脚本
- `benchmark/cifar10/model/gradfl_models/`：GradFL使用的模型实现

## 注意事项

- 公平性评估需要足够多的客户端参与，建议至少使用5个以上的客户端
- 对于`fedavg`策略，所有客户端使用相同的完整模型，因此子模型比例与性能的相关性分析不适用
- 评估结果可能受到数据分布、模型架构、优化器等因素的影响