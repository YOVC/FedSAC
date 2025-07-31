"""
简化的客户端资源模拟器

该模块提供了一个简化的客户端资源模拟功能，主要关注CPU和内存资源的异构性建模。
相比复杂的多维度资源模拟，这个版本更加轻量级和易于使用。

主要功能：
1. 生成异构的客户端CPU和内存配置
2. 计算基于资源的训练时间
3. 评估客户端资源贡献度
4. 记录训练历史

作者: AI Assistant
日期: 2024
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

@dataclass
class ResourceProfile:
    """客户端资源配置"""
    client_id: int
    cpu_cores: int
    cpu_frequency: float  # GHz
    memory_gb: float
    compute_score: float  # 综合计算能力评分 (0-1)
    training_history: List[Dict] = field(default_factory=list)

class ClientResourceSimulator:
    """简化的客户端资源模拟器"""
    
    def __init__(self, num_clients: int, seed: Optional[int] = None, config_file: Optional[str] = None):
        """
        初始化客户端资源模拟器
        
        Args:
            num_clients: 客户端数量
            seed: 随机种子
            config_file: 配置文件路径（暂未使用）
        """
        self.num_clients = num_clients
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 初始化客户端资源配置
        self.client_profiles = self._initialize_client_resources()
        
        logging.info(f"初始化了 {num_clients} 个客户端的资源配置")
    
    def _initialize_client_resources(self) -> List[ResourceProfile]:
        """初始化客户端资源配置"""
        profiles = []
        
        for i in range(self.num_clients):
            # 简单的资源配置生成
            cpu_cores = np.random.randint(2, 17)  # 2-16核
            cpu_frequency = np.random.uniform(1.5, 4.0)  # 1.5-4.0 GHz
            memory_gb = np.random.choice([4, 8, 16, 32])  # 常见内存配置
            
            # 计算综合评分
            compute_score = self._calculate_compute_score(cpu_cores, cpu_frequency, memory_gb)
            
            profile = ResourceProfile(
                cpu_cores=cpu_cores,
                cpu_frequency=cpu_frequency,
                memory_gb=memory_gb,
                compute_score=compute_score,
                training_history=[]
            )
            
            profiles.append(profile)
        
        return profiles
    
    def _calculate_compute_score(self, cpu_cores: int, cpu_frequency: float, memory_gb: float) -> float:
        """计算计算能力综合评分"""
        # 简单的评分公式
        cpu_score = (cpu_cores * cpu_frequency) / (16 * 4.0)  # 标准化到16核4GHz
        memory_score = memory_gb / 32.0  # 标准化到32GB
        
        # 加权平均
        score = 0.7 * cpu_score + 0.3 * memory_score
        return min(1.0, score)  # 限制在0-1范围内
    
    def get_client_profile(self, client_id: int) -> ResourceProfile:
        """获取客户端资源配置"""
        if 0 <= client_id < self.num_clients:
            return self.client_profiles[client_id]
        else:
            raise ValueError(f"客户端ID {client_id} 超出范围 [0, {self.num_clients-1}]")
    
    def calculate_training_time(self, client_id: int, model_size_mb: float, epochs: int = 1) -> float:
        """
        计算训练时间
        
        Args:
            client_id: 客户端ID
            model_size_mb: 模型大小(MB)
            epochs: 训练轮数
            
        Returns:
            训练时间(秒)
        """
        profile = self.get_client_profile(client_id)
        
        # 基础训练时间计算（简化公式）
        base_time_per_mb_per_epoch = 2.0  # 基础时间：每MB每轮2秒
        
        # 根据计算能力调整
        time_factor = 1.0 / max(0.1, profile.compute_score)  # 计算能力越强，时间越短
        
        training_time = model_size_mb * epochs * base_time_per_mb_per_epoch * time_factor
        
        return max(1.0, training_time)  # 最少1秒
    
    def calculate_resource_contribution(self, client_id: int, training_time: float) -> float:
        """
        计算资源贡献度
        
        Args:
            client_id: 客户端ID
            training_time: 训练时间
            
        Returns:
            资源贡献度 (0-1)
        """
        profile = self.get_client_profile(client_id)
        
        # 简化的贡献度计算
        # 主要基于计算能力和训练效率
        efficiency_score = 1.0 / (1.0 + training_time / 100.0)  # 训练时间越短效率越高
        contribution = 0.7 * profile.compute_score + 0.3 * efficiency_score
        
        return min(1.0, contribution)
    
    def update_training_history(self, client_id: int, round_num: int, training_time: float, 
                              model_size_mb: float, contribution: float):
        """更新训练历史"""
        profile = self.get_client_profile(client_id)
        
        history_entry = {
            'round': round_num,
            'training_time': training_time,
            'model_size_mb': model_size_mb,
            'contribution': contribution,
            'compute_score': profile.compute_score
        }
        
        profile.training_history.append(history_entry)
    
    def print_resource_statistics(self):
        """打印资源统计信息"""
        if not self.client_profiles:
            print("没有客户端资源配置")
            return
        
        cpu_cores = [p.cpu_cores for p in self.client_profiles]
        cpu_freq = [p.cpu_frequency for p in self.client_profiles]
        memory = [p.memory_gb for p in self.client_profiles]
        compute_scores = [p.compute_score for p in self.client_profiles]
        
        print("=== 客户端资源统计 ===")
        print(f"CPU核心数: 平均={np.mean(cpu_cores):.1f}, 范围=[{min(cpu_cores)}, {max(cpu_cores)}]")
        print(f"CPU频率(GHz): 平均={np.mean(cpu_freq):.2f}, 范围=[{min(cpu_freq):.1f}, {max(cpu_freq):.1f}]")
        print(f"内存(GB): 平均={np.mean(memory):.1f}, 范围=[{min(memory)}, {max(memory)}]")
        print(f"计算能力评分: 平均={np.mean(compute_scores):.3f}, 标准差={np.std(compute_scores):.3f}")
    
    def save_config(self, file_path: str):
        """保存配置到文件"""
        config_data = {
            'num_clients': self.num_clients,
            'seed': self.seed,
            'client_profiles': []
        }
        
        for profile in self.client_profiles:
            profile_data = {
                'cpu_cores': profile.cpu_cores,
                'cpu_frequency': profile.cpu_frequency,
                'memory_gb': profile.memory_gb,
                'compute_score': profile.compute_score
            }
            config_data['client_profiles'].append(profile_data)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"资源配置已保存到: {file_path}")