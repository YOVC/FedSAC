# 联邦学习子模型准确率预测器

基于保留比例预测子模型准确率的工具，使用机器学习方法拟合函数关系。

## 数据说明

- **输入数据**: `docs/accs.csv`
  - 第一行：全局模型裁剪成子模型时每一层的保留比例 (0.25-1.0)
  - 后续行：每一轮测试全局模型保留比例后的子模型准确率

## 文件说明

### 1. `accuracy_predictor.py` - 绝对准确率预测器（完整版）
功能最全面的版本，包含：
- 多种拟合方法（多项式、指数、Sigmoid、幂函数）
- 模型性能评估和比较
- 可视化结果
- 详细的性能分析
- **输出**: 直接的准确率数值

### 2. `simple_predictor.py` - 绝对准确率预测器（简化版）
轻量级版本，包含：
- 使用最佳模型（5次多项式）
- 简单的预测接口
- 交互式预测功能
- **输出**: 直接的准确率数值

### 3. `accuracy_retention_predictor.py` - 精度保留率预测器（完整版）⭐
**推荐使用**，功能包含：
- 多种拟合方法（多项式、Sigmoid、幂函数）
- 模型性能评估和比较
- 可视化结果
- **输出**: 子模型保留全局模型精度的百分比

### 5. `single_round_predictor.py` - 单轮次精度保留率预测器（完整版）⭐
**新功能**，基于特定轮次数据拟合，包含：
- 选择任意轮次 (1-15) 进行独立分析
- 多种拟合方法（多项式、指数、幂函数）
- 单轮次模型性能评估和可视化
- **输出**: 基于特定轮次的精度保留率预测

### 6. `simple_single_round_predictor.py` - 单轮次精度保留率预测器（简化版）⭐
**新功能**，轻量级单轮次分析，包含：
- 快速选择特定轮次进行分析
- 使用最佳模型（5次多项式）
- 交互式预测功能
- **输出**: 基于特定轮次的精度保留率预测

### 7. `round_comparator.py` - 轮次比较分析工具⭐
**新功能**，比较不同轮次的性能，包含：
- 多轮次模型性能对比
- 预测结果稳定性分析
- 最佳轮次推荐
- 可视化比较图表

## 使用方法
### 方法一：运行单轮次精度保留率预测器（推荐用于特定轮次分析）⭐
```bash
# 完整版单轮次分析
python single_round_predictor.py

# 简化版单轮次分析
python simple_single_round_predictor.py

# 轮次比较分析
python round_comparator.py
```
这将：
- 选择特定轮次进行独立分析
- 提供基于单轮次数据的精确预测
- 比较不同轮次的性能差异
- 推荐最佳轮次用于预测

### 方法二：运行全数据精度保留率预测器（推荐用于整体分析）⭐
```bash
# 完整版分析
python accuracy_retention_predictor.py

# 简化版交互式预测
python simple_retention_predictor.py
```
这将：
- 输出子模型保留全局模型精度的百分比
- 提供更直观的性能理解
- 支持交互式预测

### 方法二：运行绝对准确率预测器
```bash
# 完整版分析
python accuracy_predictor.py

# 简化版交互式预测
python simple_predictor.py
```
这将：
- 输出直接的准确率数值
- 拟合多种模型并比较性能
- 生成可视化图表

### 方法三：在代码中使用单轮次精度保留率预测器（推荐）⭐
```python
from simple_single_round_predictor import load_and_fit_single_round_model, predict_retention_rate

# 训练特定轮次的模型
round_num = 5  # 选择第5轮数据
model, poly_features, global_accuracy, stats = load_and_fit_single_round_model('docs/accs.csv', round_num)

# 预测精度保留率
retention_rate = predict_retention_rate(0.7, model, poly_features)
print(f"轮次{round_num}，保留比例0.7的精度保留率: {retention_rate:.2f}%")

# 预测多个值
ratios = [0.3, 0.5, 0.7, 0.9]
retention_rates = predict_retention_rate(ratios, model, poly_features)
for ratio, rate in zip(ratios, retention_rates):
    print(f"轮次{round_num}，保留比例{ratio}: 精度保留率{rate:.2f}%")

# 查看轮次统计信息
print(f"轮次{round_num}统计: R²={stats['r2_score']:.4f}, 全局准确率={stats['global_accuracy']:.4f}")
```

### 方法四：在代码中使用轮次比较分析⭐
```python
from round_comparator import RoundComparator

# 创建比较器
comparator = RoundComparator('docs/accs.csv')

# 加载指定轮次
comparator.load_all_rounds([1, 5, 10, 15])

# 比较模型性能
comparison_df = comparator.compare_model_performance()

# 找到最佳轮次
best_round = comparator.find_best_round_for_ratio(0.5)
print(f"保留比例0.5下的最佳轮次: {best_round['round']}")

# 分析稳定性
comparator.analyze_round_stability()
```

### 方法五：在代码中使用全数据精度保留率预测器
```python
from simple_retention_predictor import load_and_fit_retention_model, predict_retention_rate

# 训练模型
model, poly_features, global_acc_stats = load_and_fit_retention_model('docs/accs.csv')

# 预测精度保留率
retention_rate = predict_retention_rate(0.7, model, poly_features)
print(f"保留比例0.7的精度保留率: {retention_rate[0]:.2f}%")

# 预测多个值
ratios = [0.3, 0.5, 0.7, 0.9]
retention_rates = predict_retention_rate(ratios, model, poly_features)
for ratio, rate in zip(ratios, retention_rates):
    print(f"保留比例{ratio}: 精度保留率{rate:.2f}%")
```

### 方法四：在代码中使用绝对准确率预测器
```python
from simple_predictor import load_and_fit_model, predict_accuracy

# 训练模型
model, poly_features = load_and_fit_model('docs/accs.csv')

# 预测单个值
accuracy = predict_accuracy(0.7, model, poly_features)
print(f"保留比例0.7的预测准确率: {accuracy[0]:.4f}")

# 预测多个值
ratios = [0.3, 0.5, 0.7, 0.9]
accuracies = predict_accuracy(ratios, model, poly_features)
for ratio, acc in zip(ratios, accuracies):
    print(f"保留比例{ratio}: {acc:.4f}")
```

## 模型性能

### 单轮次精度保留率预测器（推荐用于特定分析）⭐
- **最佳模型**: 5次多项式拟合
- **R²范围**: 0.9900 - 0.9976 (平均 0.9951)
- **稳定性**: 非常稳定 (变异系数 0.0030)
- **数据点**: 每轮次 76个训练样本
- **优势**: 基于特定轮次，预测更精确
- **输出**: 基于特定轮次的精度保留率

### 全数据精度保留率预测器（推荐用于整体分析）
- **最佳模型**: 5次多项式拟合
- **R²**: 0.9838 (解释了98.38%的方差)
- **MSE**: 13.01 (精度保留率的均方误差)
- **数据点**: 1140个训练样本
- **输出**: 子模型保留全局模型精度的百分比

### 绝对准确率预测器
- **最佳模型**: 5次多项式拟合
- **R²**: 0.9789 (解释了97.89%的方差)
- **MSE**: 0.000636 (准确率的均方误差)
- **数据点**: 1140个训练样本
- **输出**: 直接的准确率数值

根据测试结果，各模型性能如下：

| 模型 | R² | MSE | 说明 |
|------|----|----|------|
| 5次多项式 | 0.9789 | 0.000636 | **最佳模型** |
| Sigmoid | 0.9788 | 0.000638 | 性能接近最佳 |
| 4次多项式 | 0.9788 | 0.000639 | 性能良好 |
| 3次多项式 | 0.9738 | 0.000789 | 性能良好 |
| 2次多项式 | 0.9721 | 0.000841 | 基础模型 |
| 幂函数 | 0.9666 | 0.001007 | 性能一般 |
| 指数函数 | 0.8640 | 0.004101 | 性能较差 |

## 预测示例

### 单轮次精度保留率预测器示例（轮次5）⭐
```
轮次5统计信息:
- 全局模型准确率: 60.40%
- 模型性能: R² = 0.9962, MSE = 2.939

保留比例 -> 精度保留率 -> 预测准确率 -> 性能等级
0.25 -> 15.23% -> 9.20% -> 低性能
0.50 -> 35.67% -> 21.54% -> 中等性能  
0.70 -> 42.31% -> 25.56% -> 中等性能
0.90 -> 76.89% -> 46.44% -> 高性能

应用建议:
- 保留比例 ≥ 0.8: 推荐用于生产环境
- 保留比例 0.5-0.8: 适合测试和开发
- 保留比例 < 0.5: 仅用于资源极度受限场景
```

### 全数据精度保留率预测器示例（推荐）
```
保留比例 -> 精度保留率 -> 应用建议
0.25     -> 14.73%      -> 极度压缩，仅适用于资源极度受限场景
0.30     -> 18.45%      -> 高度压缩，适用于边缘设备
0.50     -> 30.12%      -> 中等压缩，平衡性能与资源
0.70     -> 42.31%      -> 轻度压缩，保持较好性能
0.90     -> 78.45%      -> 微调压缩，接近原始性能
1.00     -> 100.00%     -> 完整模型，最佳性能
```

### 绝对准确率预测示例
使用最佳模型（5次多项式）的预测结果：

| 保留比例 | 预测准确率 | 预测百分比 |
|----------|------------|------------|
| 0.25 | 0.1039 | 10.39% |
| 0.30 | 0.1039 | 10.39% |
| 0.50 | 0.1199 | 11.99% |
| 0.70 | 0.2556 | 25.56% |
| 0.90 | 0.5195 | 51.95% |
| 1.00 | 0.6592 | 65.92% |

## 注意事项

1. **数据范围**: 模型在保留比例0.25-1.0范围内训练，超出此范围的预测可能不准确
2. **模型选择**: 5次多项式模型在当前数据上表现最佳，但可能存在过拟合风险
3. **实际应用**: 建议结合实际业务场景验证预测结果的合理性

## 依赖包

```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

## 在联邦学习公平性研究中的应用

### 单轮次精度保留率预测器的优势⭐
1. **特定轮次分析**: 针对特定训练轮次的精确预测
2. **时间序列分析**: 分析不同轮次的性能变化趋势
3. **最佳轮次选择**: 找到最适合预测的训练轮次
4. **稳定性评估**: 评估模型在不同轮次的稳定性
5. **精确预测**: 基于单轮次数据，预测精度更高

### 联邦学习公平性研究中的应用

#### 1. 轮次特定的公平性分析
```python
from simple_single_round_predictor import load_and_fit_single_round_model
from round_comparator import RoundComparator

# 分析不同轮次的公平性
rounds_to_analyze = [1, 5, 10, 15]
fairness_results = {}

for round_num in rounds_to_analyze:
    model, poly_features, global_acc, stats = load_and_fit_single_round_model('docs/accs.csv', round_num)
    
    # 模拟不同客户端类型
    client_types = {
        'high_resource': 0.9,    # 高资源客户端
        'medium_resource': 0.7,  # 中等资源客户端  
        'low_resource': 0.3      # 低资源客户端
    }
    
    round_fairness = {}
    for client_type, retention_ratio in client_types.items():
        retention_rate = predict_retention_rate(retention_ratio, model, poly_features)
        round_fairness[client_type] = {
            'retention_ratio': retention_ratio,
            'retention_rate': retention_rate,
            'predicted_accuracy': global_acc * retention_rate / 100
        }
    
    fairness_results[f'round_{round_num}'] = round_fairness

# 分析公平性趋势
print("轮次公平性分析:")
for round_key, round_data in fairness_results.items():
    high_acc = round_data['high_resource']['predicted_accuracy']
    low_acc = round_data['low_resource']['predicted_accuracy']
    fairness_gap = high_acc - low_acc
    print(f"{round_key}: 公平性差距 = {fairness_gap:.4f}")
```

#### 2. 最佳轮次选择策略
```python
# 找到公平性最佳的轮次
comparator = RoundComparator('docs/accs.csv')
comparator.load_all_rounds([1, 5, 10, 15])

# 在不同保留比例下找最佳轮次
target_ratios = [0.3, 0.5, 0.7, 0.9]
best_rounds = {}

for ratio in target_ratios:
    best_round_info = comparator.find_best_round_for_ratio(ratio)
    best_rounds[ratio] = best_round_info
    print(f"保留比例{ratio}: 最佳轮次{best_round_info['round']}, "
          f"精度保留率{best_round_info['retention_rate']:.2f}%")
```

#### 3. 动态轮次选择的联邦学习
```python
def dynamic_round_selection_federated_learning():
    """基于轮次性能动态选择预测模型"""
    
    # 加载轮次比较器
    comparator = RoundComparator('docs/accs.csv')
    comparator.load_all_rounds()
    
    # 模拟联邦学习过程
    clients = [
        {'id': 1, 'resource_level': 0.9, 'data_size': 1000},
        {'id': 2, 'resource_level': 0.7, 'data_size': 800}, 
        {'id': 3, 'resource_level': 0.5, 'data_size': 600},
        {'id': 4, 'resource_level': 0.3, 'data_size': 400}
    ]
    
    aggregation_weights = []
    
    for client in clients:
        # 为每个客户端找到最佳预测轮次
        best_round_info = comparator.find_best_round_for_ratio(client['resource_level'])
        
        # 基于最佳轮次预测性能
        retention_rate = best_round_info['retention_rate']
        predicted_accuracy = best_round_info['global_accuracy'] * retention_rate / 100
        
        # 计算聚合权重（结合数据量和预测精度）
        weight = client['data_size'] * predicted_accuracy
        aggregation_weights.append({
            'client_id': client['id'],
            'weight': weight,
            'predicted_accuracy': predicted_accuracy,
            'best_round': best_round_info['round'],
            'retention_rate': retention_rate
        })
    
    # 归一化权重
    total_weight = sum(w['weight'] for w in aggregation_weights)
    for w in aggregation_weights:
        w['normalized_weight'] = w['weight'] / total_weight
    
    return aggregation_weights

# 执行动态轮次选择
weights = dynamic_round_selection_federated_learning()
for w in weights:
    print(f"客户端{w['client_id']}: 权重{w['normalized_weight']:.3f}, "
          f"最佳轮次{w['best_round']}, 精度保留率{w['retention_rate']:.2f}%")
```

### 全数据精度保留率预测器的优势⭐
1. **直观理解**: 输出百分比形式，更容易理解子模型相对于全局模型的性能损失
2. **公平性分析**: 可以直接比较不同客户端在相同保留比例下的性能保留情况
3. **资源分配**: 根据目标精度保留率，合理分配计算资源
4. **性能基准**: 提供统一的性能评估标准

### 应用场景

#### 1. 公平性分析
```python
# 分析不同保留比例下的公平性
retention_rates = predict_retention_rate([0.3, 0.5, 0.7], model, poly_features)
print("公平性分析:")
for ratio, rate in zip([0.3, 0.5, 0.7], retention_rates):
    print(f"保留比例{ratio}: 精度保留率{rate:.1f}%")
```

#### 2. 资源优化
```python
# 寻找达到目标精度保留率的最小保留比例
target_retention = 40.0  # 目标保留40%精度
# 通过二分搜索或优化算法找到最优保留比例
```

#### 3. 客户端选择
- 根据客户端计算能力，选择合适的保留比例
- 确保所有客户端都能达到最低精度保留率要求

#### 4. 模型压缩策略
- 制定分层压缩策略
- 平衡模型大小与性能损失

#### 5. 联邦学习轮次优化
- 根据精度保留率调整聚合权重
- 优化全局模型更新策略