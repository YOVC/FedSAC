"""
全局神经元重要性预计算模块
在服务器端预先计算神经元重要性，避免客户端重复计算
"""
import torch
import torch.nn as nn
import copy
import logging
from collections import OrderedDict
import pickle
import os


class GlobalNeuronImportanceCalculator:
    """全局神经元重要性计算器"""
    
    def __init__(self, model, calculator, cache_dir="./neuron_importance_cache"):
        self.model = model
        self.calculator = calculator
        self.cache_dir = cache_dir
        self.neuron_importance_cache = {}
        self.layer_mapping = {}
        self._build_layer_mapping()
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
    
    def _build_layer_mapping(self):
        """构建层名称到模块的映射"""
        self.layer_mapping = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.layer_mapping[name] = module
    
    def _get_cache_key(self, model_state_dict, data_loader_hash):
        """生成缓存键"""
        # 使用模型参数的哈希值和数据加载器的哈希值作为缓存键
        model_hash = hash(str(model_state_dict))
        return f"importance_{model_hash}_{data_loader_hash}"
    
    def _calculate_data_loader_hash(self, data_loader):
        """计算数据加载器的哈希值（简化版本）"""
        # 这里可以根据需要实现更复杂的哈希逻辑
        return hash(str(len(data_loader.dataset)))
    
    def compute_global_neuron_importance(self, data_loader, force_recompute=False):
        """
        计算全局神经元重要性
        Args:
            data_loader: 验证数据加载器
            force_recompute: 是否强制重新计算
        Returns:
            dict: 每层的神经元重要性分数
        """
        # 检查缓存
        # data_hash = self._calculate_data_loader_hash(data_loader)
        # cache_key = self._get_cache_key(self.model.state_dict(), data_hash)
        cache_key = "importance"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not force_recompute and os.path.exists(cache_file):
            logging.info("从缓存加载神经元重要性...")
            with open(cache_file, 'rb') as f:
                self.neuron_importance_cache = pickle.load(f)
            return self.neuron_importance_cache
        
        logging.info("开始计算全局神经元重要性...")
        self.model.eval()
        
        # 计算原始损失
        original_loss = self._calculate_average_loss(data_loader)
        logging.info(f"原始损失: {original_loss:.6f}")
        
        importance_results = {}
        
        # 遍历所有需要评估的层
        for layer_name, module in self.layer_mapping.items():
            logging.info(f"评估层: {layer_name}")
            
            if isinstance(module, nn.Conv2d):
                num_neurons = module.out_channels
            elif isinstance(module, nn.Linear):
                num_neurons = module.out_features
            else:
                continue
            
            # 计算该层的神经元重要性
            layer_importance = self._evaluate_layer_importance(
                module, data_loader, original_loss, num_neurons
            )
            
            importance_results[layer_name] = {
                'importance_scores': layer_importance,
                'num_neurons': num_neurons,
                'layer_type': type(module).__name__
            }
            
            logging.info(f"层 {layer_name} 完成，神经元数: {num_neurons}")
        
        # 缓存结果
        self.neuron_importance_cache = importance_results
        # 如果cache_file存在，先删除
        if os.path.exists(cache_file):
            os.remove(cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(importance_results, f)
        
        logging.info("全局神经元重要性计算完成并已缓存")
        return importance_results
    
    def _calculate_average_loss(self, data_loader):
        """计算模型在数据集上的平均损失"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in data_loader:
                _, loss = self.calculator.test(self.model, batch_data)
                total_loss += loss * len(batch_data[1])
                total_samples += len(batch_data[1])
        
        return total_loss / total_samples if total_samples > 0 else 0
    
    def _evaluate_layer_importance(self, layer, data_loader, original_loss, num_neurons):
        """评估单个层的神经元重要性（批量优化版本）"""
        importance_scores = []
        
        # 保存原始权重
        original_weight = layer.weight.data.clone()
        original_bias = None
        if layer.bias is not None:
            original_bias = layer.bias.data.clone()
        
        # 批量评估神经元重要性（可以进一步优化）
        for neuron_idx in range(num_neurons):
            # 将该神经元的权重设为0
            if isinstance(layer, nn.Conv2d):
                layer.weight.data[neuron_idx, :, :, :] = 0
            elif isinstance(layer, nn.Linear):
                layer.weight.data[neuron_idx, :] = 0
            
            if layer.bias is not None:
                layer.bias.data[neuron_idx] = 0
            
            # 计算修改后的损失
            modified_loss = self._calculate_average_loss(data_loader)
            
            # 计算重要性分数
            importance = max(modified_loss - original_loss, 0)
            importance_scores.append(importance)
            
            # 恢复原始权重
            if isinstance(layer, nn.Conv2d):
                layer.weight.data[neuron_idx, :, :, :] = original_weight[neuron_idx, :, :, :]
            elif isinstance(layer, nn.Linear):
                layer.weight.data[neuron_idx, :] = original_weight[neuron_idx, :]
            
            if layer.bias is not None:
                layer.bias.data[neuron_idx] = original_bias[neuron_idx]
        
        return importance_scores
    
    def get_layer_topk_neurons(self, layer_name, k, mode='importance'):
        """
        获取指定层的top-k神经元索引
        Args:
            layer_name: 层名称
            k: 选择的神经元数量
            mode: 选择模式 ('importance', 'random')
        Returns:
            torch.Tensor: 选中的神经元索引
        """
        if layer_name not in self.neuron_importance_cache:
            raise ValueError(f"层 {layer_name} 的重要性未计算")
        
        layer_info = self.neuron_importance_cache[layer_name]
        importance_scores = layer_info['importance_scores']
        num_neurons = layer_info['num_neurons']
        
        if mode == 'importance':
            importance_tensor = torch.tensor(importance_scores)
            if importance_tensor.sum() == 0:
                # 如果所有重要性都为0，则随机选择
                selected_indices = torch.randperm(num_neurons)[:k]
            else:
                selected_indices = torch.topk(importance_tensor, k)[1]
        elif mode == 'random':
            selected_indices = torch.randperm(num_neurons)[:k]
        else:
            raise ValueError(f"不支持的选择模式: {mode}")
        
        return torch.sort(selected_indices)[0]  # 按原始位置排序
    
    def clear_cache(self):
        """清除缓存"""
        self.neuron_importance_cache.clear()
        # 删除缓存文件
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
    
    def get_importance_summary(self):
        """获取重要性统计摘要"""
        if not self.neuron_importance_cache:
            return "未计算神经元重要性"
        
        summary = []
        total_neurons = 0
        
        for layer_name, layer_info in self.neuron_importance_cache.items():
            importance_scores = layer_info['importance_scores']
            num_neurons = layer_info['num_neurons']
            layer_type = layer_info['layer_type']
            
            importance_tensor = torch.tensor(importance_scores)
            mean_importance = importance_tensor.mean().item()
            std_importance = importance_tensor.std().item()
            max_importance = importance_tensor.max().item()
            min_importance = importance_tensor.min().item()
            
            summary.append(f"层 {layer_name} ({layer_type}): "
                         f"神经元数={num_neurons}, "
                         f"重要性均值={mean_importance:.6f}, "
                         f"标准差={std_importance:.6f}, "
                         f"范围=[{min_importance:.6f}, {max_importance:.6f}]")
            
            total_neurons += num_neurons
        
        summary.insert(0, f"总神经元数: {total_neurons}")
        return "\n".join(summary)


def create_global_importance_calculator(model, calculator, cache_dir="./neuron_importance_cache"):
    """创建全局神经元重要性计算器的工厂函数"""
    return GlobalNeuronImportanceCalculator(model, calculator, cache_dir)