from utils import fmodule
from .fedbase import BasicServer, BasicClient
import copy
import math
from utils.fmodule import add_gradient_updates, flatten, \
            mask_grad_update_by_order, add_update_to_model, compute_grad_update, unflatten
import torch
import torch.nn.functional as F
import numpy as np
from torch.linalg import norm
from main import logger
import utils.fflow as flw
import os
import logging
from collections import OrderedDict
import random
from benchmark.cifar10.model.gradfl_models.resnet import resnet18

# 配置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None, validation=None):
        super(Server, self).__init__(option, model, clients, test_data, validation)
        
        # GradFL specific parameters
        self.mode = option.get('mode', 'awareGrad')  # 子模型生成策略：awareGrad, roll, rand, hetero, fedavg
        self.select_mode = option.get('select_mode', 'absmax')  # 选择模式：absmax, probs, absmin

        self.fixed_keep_ratios = [0.8, 0.5, 0.9, 0.7, 0.6, 0.75, 0.4, 0.55, 1.0, 0.45]
        
        # 为每个客户端维护子模型和子模型形状
        self.client_submodels = [copy.deepcopy(model) for _ in range(self.num_clients)]
        self.clients_models_shape = {i: {} for i in range(self.num_clients)}
        
        # 记录上一轮客户端信息
        self.last_client_info = {i: [] for i in range(self.num_clients)}
        
        # 数据集类别列表（用于梯度感知模式）
        self.class_list = None
        
    def run(self):
        """
        Start the federated learning system where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        # 用于存储相关系数的字典
        corrs_agg = {}
        
        for round in range(self.num_rounds + 1):
            logging.info(f"--------------Round {round}--------------")
            logger.time_start('Time Cost')
            self.iterate(round)
            self.global_lr_scheduler(round)
            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): 
                logger.log(self, round=round, corrs_agg=corrs_agg)

        logging.info("==================== 训练完成 ====================")
        logger.time_end('Total Time Cost')
        # save results as .json file
        logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
        return

    def iterate(self, t):
        """
        GradFL核心迭代过程
        """
        # 1. 选择活跃客户端
        self.selected_clients = [i for i in range(self.num_clients)]
        
        # 2. 如果是第一轮，所有客户端使用全局模型进行训练
        if t == 0:
            logging.info(f"第一轮：所有客户端使用全局模型进行训练")
            # 为每个客户端设置完整的全局模型
            for idx in self.selected_clients:
                self.construct_fedavg_submodel(idx)
        else:
            # 非第一轮，使用上一轮已经构建好的子模型进行训练
            # 这里不需要额外操作，因为子模型已经在上一轮的最后构建好了
            logging.info(f"使用上一轮构建的子模型进行训练")
        
        # 3. 客户端本地训练
        logging.info(f"开始客户端本地训练")
        trained_models, _ = self.communicate(self.selected_clients)
        
        # 4. 聚合客户端模型更新全局模型
        logging.info(f"开始聚合客户端模型")
        self.aggregate_client_models(trained_models, self.selected_clients)
        
        # 5. 测试全局模型性能
        logging.info(f"评估全局模型性能")
        self.test()
        
        # 6. 为下一轮构建客户端子模型（第一轮之后）
        if t < self.num_rounds:  # 确保不在最后一轮构建子模型
            logging.info(f"为下一轮构建客户端子模型")
            self.construct_client_submodels(self.selected_clients)
        
        # 7. 更新客户端信息
        for idx in self.selected_clients:
            if not self.last_client_info[idx]:
                # 如果是首次参与，初始化列表
                self.last_client_info[idx] = []
            else:
                # 清空之前的信息
                self.last_client_info[idx].clear()
            # 添加新信息：子模型比例和模型参数
            rate = self.get_client_submodel_rate(idx)
            self.last_client_info[idx].extend([rate, copy.deepcopy(trained_models[idx].state_dict())])
        
        return
    
    def get_client_submodel_rate(self, client_idx):
        """获取客户端子模型比例"""
        return self.fixed_keep_ratios[client_idx]
    
    def construct_client_submodels(self, client_indices):
        """为选中的客户端构建子模型"""
        for idx in client_indices:
            # 获取客户端子模型比例
            rate = self.get_client_submodel_rate(idx)
            
            # 根据不同的模式生成子模型索引
            if self.mode == 'awareGrad':
                # 基于梯度的子模型生成
                self.construct_gradient_aware_submodel(idx, rate)
            elif self.mode == 'roll':
                # 滚动选择通道
                self.construct_roll_submodel(idx, rate)
            elif self.mode == 'rand':
                # 随机选择通道
                self.construct_random_submodel(idx, rate)
            elif self.mode == 'hetero':
                # 基于固定顺序选择通道
                self.construct_hetero_submodel(idx, rate)
            elif self.mode == 'fedavg':
                # 完整模型（FedAvg基准）
                self.construct_fedavg_submodel(idx)
            else:
                raise ValueError(f"不支持的子模型生成模式: {self.mode}")
    
    def get_model(model_name, mode_rate):
        if model_name == 'resnet18':
            # ResNet18配置：4个残差块组，每组通道数分别为64,128,256,512
            hidden_size = [64, 128, 256, 512]
            num_blocks = [2, 2, 2, 2]  # 每组包含2个残差块
            model = resnet18(hidden_size, num_blocks=num_blocks, num_classes=10, model_rate=mode_rate)
        return model

    def construct_gradient_aware_submodel(self, client_idx, rate):
        """基于梯度的子模型生成"""
        # 这里需要实现基于梯度的子模型生成逻辑
        # 在实际实现中，需要获取客户端数据样本，计算梯度，然后基于梯度选择重要的通道
        # 由于我们无法直接访问客户端数据，这里简化处理
        
        # 如果有上一轮的客户端信息，使用它来推断客户端的类别
        if self.last_client_info[client_idx]:
            # 使用上一轮的模型参数来推断客户端的类别
            # 实际实现中，这里应该使用公共数据集来评估模型在各类别上的性能
            pass
        
        # 简化实现：随机生成梯度
        gradient = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and (param.dim() > 1):
                gradient[name] = torch.rand_like(param)
        
        # 使用模型的get_idx_aware_grad方法基于梯度生成子模型索引
        self.model.get_idx_aware_grad(rate, self.select_mode, gradient)
        client_model_idx = copy.deepcopy(self.model.idx)
        self.model.clear_idx()
        
        # 从全局模型中提取子模型参数
        client_model_params = self.get_model_params(self.model, client_model_idx)
        self.client_submodels[client_idx] = self.get_model(self.model, mode_rate=rate)
        # 加载参数到客户端子模型
        self.client_submodels[client_idx].load_state_dict(client_model_params)
        
        # 保存子模型形状信息
        self.clients_models_shape[client_idx] = copy.deepcopy(client_model_idx)
    
    def construct_roll_submodel(self, client_idx, rate):
        """滚动选择通道的子模型生成"""
        # 使用模型的get_idx_roll方法生成子模型索引
        if not hasattr(self.model, 'roll'):
            self.model.roll = 0
            
        self.model.get_idx_roll(rate)
        client_model_idx = copy.deepcopy(self.model.idx)
        self.model.roll += 1  # 更新滚动位置
        self.model.clear_idx()
        
        # 从全局模型中提取子模型参数
        client_model_params = self.get_model_params(self.model, client_model_idx)
        
        # 加载参数到客户端子模型
        self.client_submodels[client_idx].load_state_dict(client_model_params)
        
        # 保存子模型形状信息
        self.clients_models_shape[client_idx] = copy.deepcopy(client_model_idx)
    
    def construct_random_submodel(self, client_idx, rate):
        """随机选择通道的子模型生成"""
        # 使用模型的get_idx_rand方法生成子模型索引
        self.model.get_idx_rand(rate)
        client_model_idx = copy.deepcopy(self.model.idx)
        self.model.clear_idx()
        
        # 从全局模型中提取子模型参数
        client_model_params = self.get_model_params(self.model, client_model_idx)
        
        # 加载参数到客户端子模型
        self.client_submodels[client_idx].load_state_dict(client_model_params)
        
        # 保存子模型形状信息
        self.clients_models_shape[client_idx] = copy.deepcopy(client_model_idx)
    
    def construct_hetero_submodel(self, client_idx, rate):
        """基于固定顺序选择通道的子模型生成"""
        # 使用模型的get_idx_hetero方法生成子模型索引
        self.model.get_idx_hetero(rate)
        client_model_idx = copy.deepcopy(self.model.idx)
        self.model.clear_idx()
        
        # 从全局模型中提取子模型参数
        client_model_params = self.get_model_params(self.model, client_model_idx)
        
        # 加载参数到客户端子模型
        self.client_submodels[client_idx].load_state_dict(client_model_params)
        
        # 保存子模型形状信息
        self.clients_models_shape[client_idx] = copy.deepcopy(client_model_idx)
    
    def construct_fedavg_submodel(self, client_idx):
        """完整模型（FedAvg基准）的子模型生成"""
        # 使用模型的get_idx_hetero方法生成完整模型索引（比例为1）
        self.model.get_idx_hetero(1)
        client_model_idx = copy.deepcopy(self.model.idx)
        self.model.clear_idx()
        
        # 从全局模型中提取子模型参数
        client_model_params = self.get_model_params(self.model, client_model_idx)
        
        # 加载参数到客户端子模型
        self.client_submodels[client_idx].load_state_dict(client_model_params)
        
        # 保存子模型形状信息
        self.clients_models_shape[client_idx] = copy.deepcopy(client_model_idx)
    
    def get_model_params(self, global_model, client_model_idx):
        """从全局模型中提取子模型参数"""
        client_model_params = OrderedDict()
        for k, v in global_model.state_dict().items():
            if k in client_model_idx:
                if v.dim() > 1:
                    client_model_params[k] = copy.deepcopy(v[torch.meshgrid(client_model_idx[k], indexing='ij')])
                else:
                    client_model_params[k] = copy.deepcopy(v[client_model_idx[k]])
            else:
                raise NameError(f"无法匹配参数: {k}")
        return client_model_params
    
    def aggregate_client_models(self, client_models, client_indices):
        """聚合客户端模型更新全局模型"""
        # 初始化全局模型参数
        global_temp_params = OrderedDict()
        client_num_model_param = OrderedDict()
        
        # 初始化参数计数器
        for k, v in self.model.state_dict().items():
            global_temp_params[k] = copy.deepcopy(v)
            client_num_model_param[k] = torch.ones_like(v)
        
        # 聚合客户端模型参数
        for idx in client_indices:
            for k, v in client_models[idx].state_dict().items():
                temp_shape = self.clients_models_shape[idx][k]
                if k in global_temp_params:
                    if v.dim() > 1:
                        global_temp_params[k][torch.meshgrid(temp_shape, indexing='ij')] += v
                        client_num_model_param[k][torch.meshgrid(temp_shape, indexing='ij')] += 1
                    else:
                        global_temp_params[k][temp_shape] += v
                        client_num_model_param[k][temp_shape] += 1
                else:
                    raise NameError(f"无法匹配参数: {k}")
        
        # 计算平均值
        for k in global_temp_params:
            global_temp_params[k] /= client_num_model_param[k]
        
        # 更新全局模型
        self.model.load_state_dict(global_temp_params)
    
    def pack(self, client_id):
        """打包发送给客户端的信息"""
        return {"model" : self.client_submodels[client_id]}

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def train(self, model):
        """客户端本地训练过程"""
        logger.time_start('Train Time Cost')
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, 
                                                lr=self.learning_rate, 
                                                weight_decay=self.weight_decay, 
                                                momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data)
                loss.backward()
                optimizer.step()
        logger.time_end('Train Time Cost')
        return