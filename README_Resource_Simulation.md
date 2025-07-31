# 简化版客户端资源模拟功能

## 概述

本模块提供了一个简化的客户端资源模拟功能，专注于CPU和内存资源的异构性建模。相比复杂的多维度资源模拟，这个版本更加轻量级和易于使用，适合联邦学习公平性研究的基础需求。

## 功能特性

### 1. 核心资源建模
- **CPU资源**: 核心数量(2-16核)和频率(1.5-4.0GHz)
- **内存资源**: 常见配置(4GB, 8GB, 16GB, 32GB)
- **计算能力评分**: 基于CPU和内存的综合评分(0-1)

### 2. 简化的评估指标
- **训练时间计算**: 基于模型大小和客户端计算能力
- **资源贡献度**: 综合计算能力和训练效率的评估
- **训练历史记录**: 记录每轮训练的资源使用情况

### 3. 易于使用
- **简单配置**: 最少参数即可启动
- **轻量级**: 移除了复杂的设备类型和网络建模
- **高效**: 快速的资源计算和评估

## 文件结构

```
algorithm/
├── client_resource_simulator.py  # 核心资源模拟器
├── GradFL.py                     # 集成了资源模拟的GradFL算法
└── fedbase.py                    # 基础联邦学习框架

example_resource_config.py        # 使用示例和配置说明
README_Resource_Simulation.md     # 本文档
```

## 使用方法

### 1. 基本使用

在GradFL算法中启用资源模拟：

```python
# 在option配置中添加以下参数
option = {
    'enable_resource_simulation': True,  # 启用资源模拟
    'num_clients': 10,                   # 客户端数量
    'seed': 42,                          # 随机种子
    'resource_config_file': None,        # 配置文件路径（None使用默认配置）
    # ... 其他参数
}

# 创建服务器实例
server = Server(option, model, clients, test_data, validation)
```

### 2. 自定义资源配置

创建自定义资源配置文件：

```python
# custom_resource_config.py
resource_config = {
    "device_distribution": {
        "smartphone": 0.4,
        "tablet": 0.2,
        "laptop": 0.25,
        "desktop": 0.1,
        "server": 0.05
    },
    "network_distribution": {
        "wifi": 0.6,
        "ethernet": 0.25,
        "4g": 0.1,
        "5g": 0.05
    },
    # ... 更多配置
}
```

### 3. 获取资源信息

在训练过程中，系统会自动记录以下信息：

```python
# 客户端训练时会自动计算和记录：
- training_time          # 训练时间（秒）
- network_time           # 网络传输时间（秒）
- total_time            # 总时间（秒）
- compute_score         # 计算能力分数
- resource_contribution # 资源贡献度
- energy_consumption    # 能耗（焦耳）
- battery_level         # 电池电量（%）
- thermal_state         # 热状态
- network_state         # 网络状态
- is_available          # 是否可用
```

## 核心算法

### 1. 训练时间计算

```python
def calculate_training_time(client_id, model_size_mb, epochs=1):
    """
    基于客户端计算能力和模型大小计算训练时间
    
    考虑因素：
    - CPU性能（核心数 × 频率）
    - 内存容量
    - 模型大小
    - 训练轮数
    - 设备热状态影响
    """
    base_time = (model_size_mb * epochs) / compute_capability
    thermal_factor = 1.0 + thermal_state * 0.3  # 热状态影响性能
    return base_time * thermal_factor
```

### 2. 资源贡献度评估

```python
def calculate_resource_contribution(client_id, training_time):
    """
    综合评估客户端资源贡献度
    
    考虑因素：
    - 计算能力（40%权重）
    - 网络质量（30%权重）
    - 可靠性（20%权重）
    - 训练效率（10%权重）
    """
    contribution = (
        0.4 * compute_score +
        0.3 * network_score +
        0.2 * reliability_score +
        0.1 * efficiency_score
    )
    return contribution
```

### 3. 动态状态更新

```python
def update_client_state(client_id, training_time, energy_consumption):
    """
    更新客户端动态状态
    
    更新内容：
    - 电池电量消耗
    - 设备热状态变化
    - 网络状况波动
    - 训练历史记录
    """
    # 电池消耗
    battery_drain = energy_consumption / battery_capacity * 100
    battery_level -= battery_drain
    
    # 热状态更新
    thermal_increase = training_time * compute_intensity / thermal_capacity
    thermal_state = min(1.0, thermal_state + thermal_increase)
    
    # 网络状况波动
    network_state *= (0.9 + 0.2 * random.random())
```

## 与原始策略对比

| 维度 | 原始策略（高斯分布CPU） | 新策略（综合资源模拟） |
|------|----------------------|----------------------|
| 资源维度 | 仅CPU | CPU、内存、网络、设备类型 |
| 动态性 | 静态 | 动态更新（电池、热状态、网络） |
| 真实性 | 简化模型 | 基于真实设备分布 |
| 评估指标 | 单一CPU资源 | 多维度综合评估 |
| 公平性支持 | 有限 | 全面支持公平性分析 |
| 扩展性 | 低 | 高（易于添加新维度） |

## 应用场景

### 1. 联邦学习公平性研究
- 分析不同资源配置对模型性能的影响
- 评估资源分配的公平性
- 研究资源异构性对收敛的影响

### 2. 客户端选择策略优化
- 基于资源能力的智能客户端选择
- 平衡性能和公平性的选择策略
- 动态调整选择标准

### 3. 资源感知的模型分配
- 根据客户端能力分配不同大小的子模型
- 优化模型分发策略
- 减少资源消耗和训练时间

### 4. 系统性能预测和优化
- 预测训练时间和资源消耗
- 优化通信和计算策略
- 提高系统整体效率

## 配置参数说明

### 设备类型配置
```python
DEVICE_CONFIGS = {
    DeviceType.SMARTPHONE: {
        'cpu_cores': (2, 8),
        'cpu_frequency': (1.5, 3.0),
        'memory_gb': (2, 12),
        'battery_capacity': (2000, 5000),
        'compute_base_score': (0.3, 0.6)
    },
    # ... 其他设备类型
}
```

### 网络类型配置
```python
NETWORK_CONFIGS = {
    NetworkType.WIFI: {
        'bandwidth_mbps': (10, 100),
        'latency_ms': (10, 50),
        'reliability': (0.8, 0.95)
    },
    # ... 其他网络类型
}
```

## 输出和日志

### 1. 资源配置保存
系统会自动将资源配置保存到结果目录：
```
fedtask/{task_name}/csv_results/resource_config.json
```

### 2. 训练日志
每轮训练会记录详细的资源使用信息：
```
客户端 0 资源信息:
  训练时间: 45.23 秒
  网络传输时间: 12.45 秒
  总时间: 57.68 秒
  计算能力分数: 0.75
  资源贡献度: 0.6234
  能耗: 234.56 J
  电池电量: 87.3%
  热状态: 0.23
  网络状态: 0.89
  是否可用: 是
```

### 3. 统计摘要
提供资源分布的统计摘要：
```
=== 客户端资源统计 ===
设备类型分布:
  智能手机: 4 (40.0%)
  平板电脑: 2 (20.0%)
  笔记本电脑: 3 (30.0%)
  台式电脑: 1 (10.0%)

网络类型分布:
  WiFi: 6 (60.0%)
  以太网: 3 (30.0%)
  4G: 1 (10.0%)

计算能力分数: 平均=0.65, 标准差=0.18
网络质量分数: 平均=0.72, 标准差=0.15
```

## 扩展和定制

### 1. 添加新的设备类型
```python
# 在DeviceType枚举中添加新类型
class DeviceType(Enum):
    # ... 现有类型
    IOT_DEVICE = "iot_device"

# 在DEVICE_CONFIGS中添加配置
DEVICE_CONFIGS[DeviceType.IOT_DEVICE] = {
    'cpu_cores': (1, 2),
    'cpu_frequency': (0.5, 1.5),
    # ... 其他配置
}
```

### 2. 自定义资源评估算法
```python
def custom_resource_contribution(self, client_id, training_time):
    """自定义资源贡献度计算"""
    # 实现自定义逻辑
    pass
```

### 3. 添加新的动态因素
```python
def update_custom_state(self, client_id):
    """更新自定义状态"""
    # 例如：网络拥塞、设备负载等
    pass
```

## 注意事项

1. **性能影响**: 资源模拟会增加一定的计算开销，但相对于训练时间来说很小
2. **内存使用**: 每个客户端会维护额外的状态信息，大规模场景下需要注意内存使用
3. **随机性**: 使用固定种子确保实验的可重复性
4. **配置合理性**: 确保设备配置参数符合实际情况

## 未来扩展

1. **更多设备类型**: 支持边缘设备、嵌入式设备等
2. **网络建模**: 更详细的网络拓扑和延迟建模
3. **能耗优化**: 基于能耗的客户端选择和调度
4. **故障模拟**: 设备故障和网络中断的模拟
5. **成本建模**: 考虑计算和通信成本的经济模型

## 参考文献

1. 联邦学习中的资源异构性研究
2. 移动设备性能建模相关工作
3. 网络QoS和带宽估计方法
4. 能耗建模和优化技术