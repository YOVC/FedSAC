# 客户端资源模拟器

## 概述

客户端资源模拟器是一个基于MACs（Multiply-Accumulate Operations）和独立高斯分布CPU效率的联邦学习资源异构性建模工具。该模拟器为每个客户端分配独立的CPU效率高斯分布，并在每轮训练中从该分布采样当前轮的计算效率，从而更真实地模拟客户端设备的动态性能变化。

## 核心功能

### 1. MACs计算
- **自动计算模型计算量**：基于PyTorch模型自动计算MACs
- **支持多种架构**：CNN、全连接、残差网络等
- **回退方案**：当MACs计算失败时，使用参数数量估算

### 2. 独立高斯分布CPU效率建模
- **个性化分布**：每个客户端拥有独立的CPU效率高斯分布
- **动态采样**：每轮训练从客户端分布中采样当前效率
- **可控异构性**：通过全局分布和变异系数控制客户端差异

### 3. 训练时间计算
- **基于真实计算量**：训练时间 = MACs / CPU效率
- **考虑训练轮数**：支持多epoch训练时间计算
- **动态变化**：每轮采样新的CPU效率

### 4. 资源贡献度评估
- **综合评估**：基于CPU效率和训练时间
- **归一化处理**：贡献度范围[0, 1]
- **公平性指标**：用于联邦学习公平性研究

### 5. 训练历史记录
- **详细记录**：每轮训练的资源使用情况
- **统计分析**：支持多维度统计分析
- **可视化支持**：便于分析和调试

## 设计优势

1. **基于真实计算量**：使用MACs而非简单随机数，更准确反映模型复杂度
2. **动态性能建模**：每轮独立采样，模拟设备性能的动态变化
3. **个性化异构性**：每个客户端独立分布，更真实的异构环境
4. **可控参数化**：通过少量参数控制整体异构性程度
5. **扩展性强**：易于集成到现有联邦学习框架

## 文件结构

```
algorithm/
├── client_resource_simulator.py    # 核心资源模拟器
├── GradFL.py                      # 集成资源模拟的GradFL算法
├── test_resource_simulation.py    # 测试脚本
├── example_resource_config.py     # 配置示例
└── README_Resource_Simulation.md  # 本文档
```

## 使用方法

### 1. 创建资源模拟器

有两种方式创建资源模拟器：

#### 方法1: 使用全局分布参数（自动生成客户端分布）

```python
from client_resource_simulator import ClientResourceSimulator

# 使用全局分布参数创建模拟器
simulator = ClientResourceSimulator.from_global_distribution(
    num_clients=10,                      # 客户端数量
    global_cpu_efficiency_mean=10.0,     # 全局CPU效率均值 (GFLOPS)
    global_cpu_efficiency_std=3.0,       # 全局CPU效率标准差
    client_std_ratio=0.2,                # 客户端标准差比例
    seed=42                              # 随机种子
)
```

#### 方法2: 使用外部分布参数（精确控制每个客户端）

```python
# 定义每个客户端的高斯分布参数
client_distributions = {
    0: {'mean': 20.0, 'std': 2.0},  # 高性能客户端
    1: {'mean': 15.0, 'std': 1.5},  # 中等性能客户端
    2: {'mean': 8.0, 'std': 1.0},   # 低性能客户端
    3: {'mean': 12.0, 'std': 3.0},  # 高变异性客户端
    4: {'mean': 6.0, 'std': 0.5}    # 稳定低性能客户端
}

# 使用外部分布参数创建模拟器
simulator = ClientResourceSimulator.from_client_distributions(
    client_distributions=client_distributions,
    seed=42
)
```

### 获取客户端配置

```python
# 获取客户端资源配置
if gradfl.resource_simulator:
    # 打印统计信息
    gradfl.resource_simulator.print_resource_statistics()
    
    # 获取特定客户端的当前CPU效率
    client_id = 0
    current_efficiency = gradfl.resource_simulator.get_current_cpu_efficiency(client_id)
    print(f"客户端 {client_id} 当前CPU效率: {current_efficiency:.3f}")
    
    # 获取客户端的分布参数
    profile = gradfl.resource_simulator.get_client_profile(client_id)
    print(f"客户端 {client_id} 分布参数: 均值={profile.cpu_efficiency_mean:.3f}, "
          f"标准差={profile.cpu_efficiency_std:.3f}")
```

### 计算模型MACs和训练时间

```python
import torch
import torch.nn as nn

# 示例模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 计算模型MACs
model = SimpleCNN()
input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10输入尺寸
macs = gradfl.resource_simulator.calculate_model_macs(model, input_tensor)
print(f"模型MACs: {macs:,}")

# 计算训练时间
client_id = 0
batch_size = 32
num_samples = 1000
training_time = gradfl.resource_simulator.calculate_training_time(
    client_id, macs, batch_size, num_samples
)
print(f"客户端 {client_id} 预估训练时间: {training_time:.2f} 秒")
```

### 4. 在GradFL中启用资源模拟

#### 使用全局分布方式：

```python
# GradFL配置选项
option = {
    # 基本联邦学习参数
    'task': 'cifar10',
    'algorithm': 'gradfl',
    'model': 'resnet18',
    'num_rounds': 100,
    'num_clients': 20,
    
    # 资源模拟参数 - 全局分布方式
    'enable_resource_simulation': True,
    'global_cpu_efficiency_mean': 10.0,
    'global_cpu_efficiency_std': 3.0,
    'client_std_ratio': 0.2,
    'seed': 42
}

# 创建并运行GradFL
gradfl = GradFL(option)
gradfl.run()
```

#### 使用外部分布方式：

```python
# 定义客户端分布参数
client_distributions = {
    0: {'mean': 25.0, 'std': 2.0},  # 高端设备
    1: {'mean': 15.0, 'std': 1.5},  # 中端设备
    2: {'mean': 8.0, 'std': 1.0},   # 低端设备
    3: {'mean': 12.0, 'std': 3.0},  # 高变异性设备
    4: {'mean': 6.0, 'std': 0.5},   # 稳定低性能设备
    # ... 为每个客户端指定分布参数
}

# GradFL配置选项
option = {
    # 基本联邦学习参数
    'task': 'cifar10',
    'algorithm': 'gradfl',
    'model': 'resnet18',
    'num_rounds': 100,
    'num_clients': 20,
    
    # 资源模拟参数 - 外部分布方式
    'enable_resource_simulation': True,
    'client_distributions': client_distributions,
    'seed': 42
}

# 创建并运行GradFL
gradfl = GradFL(option)
gradfl.run()
```

#### 外部分布参数详细配置示例：

```python
import numpy as np

def create_mobile_distributions(num_clients):
    """移动设备异构性模拟"""
    distributions = {}
    
    # 高端设备 (33%)
    for i in range(num_clients // 3):
        distributions[i] = {
            'mean': np.random.uniform(18.0, 22.0),  # 高CPU效率
            'std': np.random.uniform(1.0, 2.0)      # 低变异性
        }
    
    # 中端设备 (33%)
    for i in range(num_clients // 3, 2 * num_clients // 3):
        distributions[i] = {
            'mean': np.random.uniform(10.0, 14.0),  # 中等CPU效率
            'std': np.random.uniform(1.5, 2.5)     # 中等变异性
        }
    
    # 低端设备 (34%)
    for i in range(2 * num_clients // 3, num_clients):
        distributions[i] = {
            'mean': np.random.uniform(3.0, 7.0),   # 低CPU效率
            'std': np.random.uniform(2.0, 4.0)     # 高变异性
        }
    
    return distributions

# 在GradFL中使用
option['client_distributions'] = create_mobile_distributions(20)
```

## 核心算法

### 1. MACs计算

MACs通过模拟前向传播过程计算：

```
MACs = Σ(卷积层MACs + 全连接层MACs)
卷积层MACs = 输出特征图大小 × 卷积核大小 × 输入通道数 × 输出通道数
全连接层MACs = 输入维度 × 输出维度
```

### 2. 独立高斯分布分配

每个客户端的CPU效率分布：

```
客户端i的效率均值 ~ N(全局均值, 全局标准差)
客户端i的效率标准差 = 客户端i的效率均值 × 变异系数
客户端i的当前轮效率 ~ N(客户端i的效率均值, 客户端i的效率标准差)
```

### 3. 训练时间计算

基于MACs和CPU效率的训练时间：

```
训练时间 = (MACs × epochs × 数据批次数) / (CPU效率 × 10^9) + 基础开销
```

### 4. 资源贡献度评估

综合CPU效率和训练时间的贡献度：

```
效率权重 = CPU效率 / max(所有客户端CPU效率)
时间权重 = min(所有客户端训练时间) / 训练时间
资源贡献度 = (效率权重 + 时间权重) / 2
```

## 与原始策略对比

| 特性 | 新策略（独立高斯分布） | 原始策略（简单随机） |
|------|----------------------|-------------------|
| 计算量建模 | 基于真实MACs | 简单随机分布 |
| 异构性建模 | 独立高斯分布，每轮采样 | 固定随机值 |
| 动态性 | 支持每轮变化 | 静态不变 |
| 可控性 | 高度可控参数 | 有限控制 |
| 真实性 | 更接近实际情况 | 较为简化 |
| 复杂度 | 中等 | 简单 |

## 应用场景

1. **联邦学习公平性研究**：评估不同资源分配策略的公平性
2. **客户端选择算法**：基于资源能力选择合适的客户端
3. **负载均衡优化**：根据客户端能力分配训练任务
4. **异构环境模拟**：模拟真实的设备异构性环境
5. **算法性能评估**：在异构环境下评估联邦学习算法

## 配置参数

### 主要参数

- `num_clients`: 客户端数量
- `global_cpu_efficiency_mean`: 全局CPU效率均值 (GFLOPS)
- `global_cpu_efficiency_std`: 全局CPU效率标准差 (GFLOPS)
- `client_std_ratio`: 客户端内部变异系数 (0.1-0.3推荐)

### 推荐配置

```python
# 低异构性环境
low_heterogeneity = {
    'global_cpu_efficiency_mean': 10.0,
    'global_cpu_efficiency_std': 2.0,
    'client_std_ratio': 0.1
}

# 中等异构性环境
medium_heterogeneity = {
    'global_cpu_efficiency_mean': 10.0,
    'global_cpu_efficiency_std': 4.0,
    'client_std_ratio': 0.2
}

# 高异构性环境
high_heterogeneity = {
    'global_cpu_efficiency_mean': 10.0,
    'global_cpu_efficiency_std': 6.0,
    'client_std_ratio': 0.3
}
```

## 输出日志

启用资源模拟后，训练过程中会输出详细的资源信息：

```
客户端 0 资源信息:
  CPU效率分布: 均值=12.34, 标准差=2.47 GFLOPS
  当前轮CPU效率: 11.89 GFLOPS
  模型MACs: 45.67M
  训练时间: 3.84秒
  资源贡献度: 0.756
```

## 扩展定制

### 1. 自定义CPU效率分布

```python
class CustomResourceSimulator(ClientResourceSimulator):
    def _generate_client_profiles(self):
        # 自定义分布逻辑
        pass
```

### 2. 添加新的资源指标

```python
def calculate_memory_usage(self, client_id, model_size):
    # 计算内存使用量
    pass
```

### 3. 自定义贡献度计算

```python
def calculate_custom_contribution(self, client_id, metrics):
    # 自定义贡献度计算
    pass
```

## 注意事项

1. **内存使用**：大量客户端时注意内存占用
2. **随机种子**：设置随机种子确保结果可复现
3. **参数范围**：CPU效率应在合理范围内（1-50 GFLOPS）
4. **MACs计算**：复杂模型可能需要使用回退方案
5. **性能考虑**：频繁采样可能影响训练速度

## 测试

运行测试脚本验证功能：

```bash
python test_resource_simulation.py
```

测试包括：
- 独立高斯分布效率模拟
- MACs计算准确性
- 多轮训练动态变化
- 统计分析功能
- 策略对比分析

## 未来扩展

1. **GPU资源建模**：添加GPU计算能力模拟
2. **网络带宽建模**：模拟通信时间
3. **电池电量建模**：移动设备电量约束
4. **更复杂的分布**：支持多模态分布
5. **实时性能监控**：动态调整资源参数

## 参考文献

1. FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data
2. Towards Fair and Privacy-aware Federated Deep Models
3. FedProx: Federated Optimization in Heterogeneous Networks
4. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning