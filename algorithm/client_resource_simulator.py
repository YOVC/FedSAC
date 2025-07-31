"""基于MACs和高斯分布CPU效率的客户端资源模拟器

使用模型的MACs（Multiply-Accumulate Operations）评估计算量，
为每个客户端分配高斯分布的CPU计算效率，
训练时间 = MACs / CPU计算效率

主要功能：
1. 基于高斯分布生成客户端CPU计算效率
2. 计算模型MACs或基于模型大小估算MACs
3. 基于MACs和CPU效率计算训练时间
4. 评估客户端资源贡献度
5. 记录训练历史

作者: AI Assistant
日期: 2024
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

@dataclass
class ResourceProfile:
    """客户端资源配置文件"""
    client_id: int
    cpu_efficiency_mean: float  # 该客户端的CPU效率均值 (GFLOPS)
    cpu_efficiency_std: float   # 该客户端的CPU效率标准差 (GFLOPS)
    current_cpu_efficiency: float = 0.0  # 当前轮的CPU效率
    training_history: List[Dict] = field(default_factory=list)

class ClientResourceSimulator:
    """基于MACs和独立高斯分布CPU效率的客户端资源模拟器"""
    
    def __init__(self, num_clients: int = None, seed: Optional[int] = None, 
                 cpu_efficiency_mean: float = 10.0, 
                 cpu_efficiency_std: float = 3.0,
                 config_file: Optional[str] = None,
                 client_std_ratio: float = 0.3,
                 client_distributions: Dict[int, Dict[str, float]] = None):
        """
        初始化客户端资源模拟器
        
        Args:
            num_clients: 客户端数量（当client_distributions为None时必须提供）
            seed: 随机种子
            cpu_efficiency_mean: 全局CPU计算效率均值 (GFLOPS)
            cpu_efficiency_std: 全局CPU计算效率标准差 (GFLOPS)
            config_file: 配置文件路径（暂未使用）
            client_std_ratio: 客户端内部标准差相对于均值的比例
            client_distributions: 外部传入的客户端分布参数字典
                格式: {client_id: {'mean': float, 'std': float}, ...}
        """
        self.seed = seed
        self.global_cpu_efficiency_mean = cpu_efficiency_mean
        self.global_cpu_efficiency_std = cpu_efficiency_std
        self.client_std_ratio = client_std_ratio
        
        # 处理客户端分布参数
        if client_distributions is not None:
            self.num_clients = len(client_distributions)
            self.client_distributions = client_distributions
            self.use_external_distributions = True
        else:
            if num_clients is None:
                raise ValueError("当client_distributions为None时，必须提供num_clients参数")
            self.num_clients = num_clients
            self.client_distributions = None
            self.use_external_distributions = False
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 初始化客户端资源配置
        self.client_profiles = self._initialize_client_resources()
        
        logging.info(f"初始化了 {self.num_clients} 个客户端的独立CPU效率分布")
        if self.use_external_distributions:
            logging.info("使用外部传入的客户端分布参数")
        else:
            logging.info(f"全局CPU效率分布: 均值={cpu_efficiency_mean:.1f} GFLOPS, 标准差={cpu_efficiency_std:.1f} GFLOPS")
            logging.info(f"客户端内部标准差比例: {client_std_ratio:.1f}")
    
    @classmethod
    def from_client_distributions(cls, client_distributions: Dict[int, Dict[str, float]], 
                                 seed: Optional[int] = None):
        """
        从外部客户端分布参数创建资源模拟器
        
        Args:
            client_distributions: 客户端分布参数字典
                格式: {client_id: {'mean': float, 'std': float}, ...}
            seed: 随机种子
            
        Returns:
            ClientResourceSimulator实例
            
        Example:
            distributions = {
                0: {'mean': 12.0, 'std': 2.0},
                1: {'mean': 8.0, 'std': 1.5},
                2: {'mean': 15.0, 'std': 3.0}
            }
            simulator = ClientResourceSimulator.from_client_distributions(distributions)
        """
        return cls(client_distributions=client_distributions, seed=seed)
    
    @classmethod
    def from_global_distribution(cls, num_clients: int, 
                               global_cpu_efficiency_mean: float = 10.0,
                               global_cpu_efficiency_std: float = 3.0,
                               client_std_ratio: float = 0.2,
                               seed: Optional[int] = None):
        """
        从全局分布参数创建资源模拟器（原有方式）
        
        Args:
            num_clients: 客户端数量
            global_cpu_efficiency_mean: 全局CPU效率均值 (GFLOPS)
            global_cpu_efficiency_std: 全局CPU效率标准差 (GFLOPS)
            client_std_ratio: 客户端内部标准差与均值的比例
            seed: 随机种子
            
        Returns:
            ClientResourceSimulator实例
        """
        return cls(
            num_clients=num_clients,
            cpu_efficiency_mean=global_cpu_efficiency_mean,
            cpu_efficiency_std=global_cpu_efficiency_std,
            client_std_ratio=client_std_ratio,
            seed=seed
        )
    
    def _initialize_client_resources(self) -> List[ResourceProfile]:
        """初始化客户端资源配置"""
        profiles = []
        
        for i in range(self.num_clients):
            if self.use_external_distributions:
                # 使用外部传入的分布参数
                if i not in self.client_distributions:
                    raise ValueError(f"客户端 {i} 的分布参数未在client_distributions中提供")
                
                client_params = self.client_distributions[i]
                client_mean = client_params['mean']
                client_std = client_params['std']
                
                # 验证参数有效性
                if client_mean <= 0:
                    raise ValueError(f"客户端 {i} 的均值必须大于0，当前值: {client_mean}")
                if client_std < 0:
                    raise ValueError(f"客户端 {i} 的标准差必须非负，当前值: {client_std}")
            else:
                # 使用全局分布自动生成参数
                # 客户端的均值从全局分布中采样
                client_mean = np.random.normal(self.global_cpu_efficiency_mean, self.global_cpu_efficiency_std)
                client_mean = max(1.0, client_mean)  # 确保最小值为1 GFLOPS
                
                # 客户端的标准差为其均值的一定比例
                client_std = client_mean * self.client_std_ratio
            
            profile = ResourceProfile(
                client_id=i,
                cpu_efficiency_mean=client_mean,
                cpu_efficiency_std=client_std,
                current_cpu_efficiency=client_mean,  # 初始值设为均值
                training_history=[]
            )
            
            profiles.append(profile)
        
        return profiles
    
    def calculate_model_macs(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """
        计算模型的MACs (Multiply-Accumulate Operations)
        
        Args:
            model: PyTorch模型
            input_shape: 输入张量形状 (不包括batch维度)
            
        Returns:
            MACs数量 (单位: M, 百万次操作)
        """
        def conv_hook(module, input, output):
            batch_size, input_channels, input_height, input_width = input[0].size()
            output_channels, output_height, output_width = output[0].size()
            
            kernel_dims = module.kernel_size
            kernel_flops = kernel_dims[0] * kernel_dims[1] * (input_channels // module.groups)
            
            output_elements = batch_size * output_channels * output_height * output_width
            flops = kernel_flops * output_elements
            
            module.__flops__ += int(flops)
        
        def linear_hook(module, input, output):
            input_last_dim = input[0].size(-1)
            output_last_dim = output[0].size(-1)
            
            num_instances = 1
            for i in range(len(input[0].size()) - 1):
                num_instances *= input[0].size(i)
            
            flops = num_instances * input_last_dim * output_last_dim
            module.__flops__ += int(flops)
        
        def bn_hook(module, input, output):
            input_numel = input[0].numel()
            # BN的FLOPs主要是归一化和缩放操作
            flops = 2 * input_numel
            module.__flops__ += int(flops)
        
        def relu_hook(module, input, output):
            input_numel = input[0].numel()
            # ReLU操作相对简单
            flops = input_numel
            module.__flops__ += int(flops)
        
        # 注册钩子函数
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.__flops__ = 0
                hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                module.__flops__ = 0
                hooks.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.__flops__ = 0
                hooks.append(module.register_forward_hook(bn_hook))
            elif isinstance(module, nn.ReLU):
                module.__flops__ = 0
                hooks.append(module.register_forward_hook(relu_hook))
        
        # 创建输入张量并进行前向传播
        model.eval()
        with torch.no_grad():
            input_tensor = torch.randn(1, *input_shape)
            _ = model(input_tensor)
        
        # 计算总FLOPs
        total_flops = 0
        for name, module in model.named_modules():
            if hasattr(module, '__flops__'):
                total_flops += module.__flops__
        
        # 移除钩子函数
        for hook in hooks:
            hook.remove()
        
        # 转换为MACs (MACs ≈ FLOPs / 2)
        macs = total_flops / 2.0
        
        # 返回以百万为单位的MACs
        return macs / 1e6
    
    def estimate_model_macs_simple(self, model_size_mb: float) -> float:
        """
        简单估算模型MACs（当无法直接计算时使用）
        
        Args:
            model_size_mb: 模型大小 (MB)
            
        Returns:
            估算的MACs (M)
        """
        # 经验公式：模型大小(MB) * 100 ≈ MACs(M)
        # 这是一个粗略的估算，实际值会根据模型架构有所不同
        return model_size_mb * 100
    
    def sample_cpu_efficiency(self, client_id: int) -> float:
        """
        为指定客户端从其独立的高斯分布中采样当前轮的CPU效率
        
        Args:
            client_id: 客户端ID
            
        Returns:
            当前轮的CPU效率 (GFLOPS)
        """
        profile = self.client_profiles[client_id]
        
        # 从该客户端的独立高斯分布中采样
        current_efficiency = np.random.normal(
            profile.cpu_efficiency_mean, 
            profile.cpu_efficiency_std
        )
        
        # 确保CPU效率为正值，最小值为1.0 GFLOPS
        current_efficiency = max(1.0, current_efficiency)
        
        # 更新客户端的当前CPU效率
        profile.current_cpu_efficiency = current_efficiency
        
        return current_efficiency
    
    def get_current_cpu_efficiency(self, client_id: int) -> float:
        """
        获取客户端当前轮的CPU效率
        
        Args:
            client_id: 客户端ID
            
        Returns:
            当前轮的CPU效率 (GFLOPS)
        """
        return self.client_profiles[client_id].current_cpu_efficiency

    def get_client_profile(self, client_id: int) -> ResourceProfile:
        """获取客户端资源配置"""
        if 0 <= client_id < self.num_clients:
            return self.client_profiles[client_id]
        else:
            raise ValueError(f"客户端ID {client_id} 超出范围 [0, {self.num_clients-1}]")
    
    def calculate_training_time(self, client_id: int, model_macs: float, epochs: int = 1) -> float:
        """
        计算训练时间，使用当前轮采样的CPU效率
        
        Args:
            client_id: 客户端ID
            model_macs: 模型的MACs (M)
            epochs: 训练轮数
            
        Returns:
            训练时间（秒）
        """
        # 为当前轮采样CPU效率
        current_efficiency = self.sample_cpu_efficiency(client_id)
        
        # 训练时间 = MACs / CPU计算效率
        # 考虑到训练比推理复杂，乘以一个因子（通常为3-5倍）
        training_factor = 4.0
        
        # 计算单轮训练时间
        single_epoch_time = (model_macs * training_factor) / current_efficiency
        
        # 计算总训练时间
        total_training_time = single_epoch_time * epochs
        
        # 确保最小训练时间为0.1秒
        return max(0.1, total_training_time)
    
    def calculate_training_time_from_model_size(self, client_id: int, model_size_mb: float, epochs: int = 1) -> float:
        """
        基于模型大小计算训练时间（使用估算的MACs）
        
        Args:
            client_id: 客户端ID
            model_size_mb: 模型大小 (MB)
            epochs: 训练轮数
            
        Returns:
            训练时间（秒）
        """
        # 估算MACs
        estimated_macs = self.estimate_model_macs_simple(model_size_mb)
        
        # 计算训练时间
        return self.calculate_training_time(client_id, estimated_macs, epochs)
    
    def update_training_history(self, client_id: int, round_num: int, training_metrics: Dict):
        """更新训练历史"""
        profile = self.get_client_profile(client_id)
        
        history_entry = {
            'round': round_num,
            'timestamp': training_metrics.get('timestamp'),
            'training_time': training_metrics.get('training_time'),
            'cpu_efficiency': profile.current_cpu_efficiency,
            'contribution': training_metrics.get('contribution')
        }
        
        profile.training_history.append(history_entry)
        
        # 只保留最近50轮的历史
        if len(profile.training_history) > 50:
            profile.training_history = profile.training_history[-50:]
    
    def get_all_profiles(self) -> List[ResourceProfile]:
        """获取所有客户端配置"""
        return self.client_profiles.copy()
    
    def get_statistics(self) -> Dict:
        """
        获取资源模拟器的统计信息
        
        Returns:
            包含统计信息的字典
        """
        if not self.client_profiles:
            return {'num_clients': 0}
        
        # 客户端分布参数统计
        cpu_means = [p.cpu_efficiency_mean for p in self.client_profiles]
        cpu_stds = [p.cpu_efficiency_std for p in self.client_profiles]
        
        # 当前轮效率统计
        current_efficiencies = [p.current_cpu_efficiency for p in self.client_profiles]
        
        stats = {
            'num_clients': self.num_clients,
            'client_distribution_params': {
                'mean_of_means': np.mean(cpu_means),
                'std_of_means': np.std(cpu_means),
                'min_mean': np.min(cpu_means),
                'max_mean': np.max(cpu_means),
                'mean_of_stds': np.mean(cpu_stds),
            },
            'current_round_efficiency': {
                'mean': np.mean(current_efficiencies),
                'std': np.std(current_efficiencies),
                'min': np.min(current_efficiencies),
                'max': np.max(current_efficiencies)
            },
            'distribution_source': 'external' if self.use_external_distributions else 'global'
        }
        
        # 如果使用全局分布，添加全局分布信息
        if not self.use_external_distributions:
            stats['global_distribution'] = {
                'mean': self.global_cpu_efficiency_mean,
                'std': self.global_cpu_efficiency_std
            }
            stats['client_distribution_params']['std_ratio'] = self.client_std_ratio
        
        return stats
    
    def print_resource_statistics(self):
        """打印资源统计信息"""
        stats = self.get_statistics()
        if stats['num_clients'] == 0:
            print("没有客户端资源配置")
            return
        
        cpu_eff = stats['cpu_efficiency']
        dist = stats['distribution']
        
        print("=== 客户端资源统计 ===")
        print(f"客户端数量: {stats['num_clients']}")
        print(f"CPU计算效率(GFLOPS): 平均={cpu_eff['mean']:.2f}, 标准差={cpu_eff['std']:.2f}")
        print(f"CPU效率范围: [{cpu_eff['min']:.1f}, {cpu_eff['max']:.1f}] GFLOPS")
        print(f"分布参数: 均值={dist['mean']:.1f}, 标准差={dist['std']:.1f}")
    
    def save_config(self, file_path: str):
        """保存配置到文件"""
        config_data = {
            'num_clients': self.num_clients,
            'seed': self.seed,
            'cpu_efficiency_mean': self.global_cpu_efficiency_mean,
            'cpu_efficiency_std': self.global_cpu_efficiency_std,
            'client_profiles': []
        }
        
        for profile in self.client_profiles:
            profile_data = {
                'client_id': profile.client_id,
                'cpu_efficiency_mean': profile.cpu_efficiency_mean,
                'cpu_efficiency_std': profile.cpu_efficiency_std
            }
            config_data['client_profiles'].append(profile_data)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"资源配置已保存到: {file_path}")