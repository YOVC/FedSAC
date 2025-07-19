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

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None, validation=None):
        super(Server, self).__init__(option, model, clients, test_data, validation)
        
        # FedSAC specific parameters
        self.beta = option.get('beta', 1.0)  # 超参数β，用于声誉计算
        self.neuron_importance_eval_interval = option.get('neuron_eval_interval', 10)  # 神经元重要性评估间隔
        
        # 客户端声誉和贡献度
        self.client_reputations = [1.0 for _ in range(len(self.clients))]
        self.client_contributions = [0.0 for _ in range(len(self.clients))]
        
        # 神经元重要性分数
        self.neuron_importance = None
        self.neuron_importance_percentiles = None
        
        # 为每个客户端维护子模型
        self.client_submodels = [copy.deepcopy(model) for _ in range(len(self.clients))]
        
        # 神经元聚合频率统计
        self.neuron_aggregation_frequency = None
        
    def run(self):
        """
        Start the federated learning system where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        for round in range(self.num_rounds + 1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')
            self.iterate(round)
            self.global_lr_scheduler(round)
            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): 
                logger.log(self, round=round)

        print("=================End==================")
        logger.time_end('Total Time Cost')
        # save results as .json file
        logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
        return

    def iterate(self, t):
        """
        FedSAC核心迭代过程
        """
        self.selected_clients = [i for i in range(self.num_clients)]
        
        # 1. 神经元重要性评估（每10轮评估一次）
        if t % self.neuron_importance_eval_interval == 0:
            self.evaluate_neuron_importance()
        
        # 2. 计算客户端贡献度
        self.compute_client_contributions()
        
        # 3. 更新客户端声誉
        self.update_client_reputations()
        
        # 4. 子模型分配模块：为每个客户端构建子模型
        self.allocate_submodels()
        
        # 5. 客户端本地训练
        submodels, losses = self.communicate(self.selected_clients)
        
        # 6. 动态聚合模块：基于神经元频率的动态聚合
        self.dynamic_aggregation(submodels)
        
        return

    def evaluate_neuron_importance(self):
        """
        神经元重要性评估模块
        使用Taylor-FO方法评估每个神经元的重要性
        """
        if self.validation is None:
            return
            
        self.model.eval()
        data_loader = self.calculator.get_data_loader(self.validation, batch_size=64)
        
        # 计算原始损失
        total_loss = 0
        total_samples = 0
        for batch_data in data_loader:
            _, loss = self.calculator.test(self.model, batch_data)
            total_loss += loss * len(batch_data[1])
            total_samples += len(batch_data[1])
        original_loss = total_loss / total_samples
        
        # 计算每个神经元的重要性
        neuron_importance = []
        
        # 遍历模型的每一层
        for layer_idx, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # 对于线性层和卷积层，计算每个神经元的重要性
                if isinstance(module, torch.nn.Linear):
                    num_neurons = module.out_features
                elif isinstance(module, torch.nn.Conv2d):
                    num_neurons = module.out_channels
                
                layer_importance = []
                original_weight = module.weight.data.clone()
                
                for neuron_idx in range(num_neurons):
                    # 将该神经元的权重设为0
                    if isinstance(module, torch.nn.Linear):
                        module.weight.data[neuron_idx, :] = 0
                        if module.bias is not None:
                            original_bias = module.bias.data[neuron_idx].clone()
                            module.bias.data[neuron_idx] = 0
                    elif isinstance(module, torch.nn.Conv2d):
                        module.weight.data[neuron_idx, :, :, :] = 0
                        if module.bias is not None:
                            original_bias = module.bias.data[neuron_idx].clone()
                            module.bias.data[neuron_idx] = 0
                    
                    # 计算移除该神经元后的损失
                    total_loss_modified = 0
                    total_samples_modified = 0
                    for batch_data in data_loader:
                        _, loss = self.calculator.test(self.model, batch_data)
                        total_loss_modified += loss * len(batch_data[1])
                        total_samples_modified += len(batch_data[1])
                    modified_loss = total_loss_modified / total_samples_modified
                    
                    # 计算重要性分数
                    importance = modified_loss - original_loss
                    layer_importance.append(importance)
                    
                    # 恢复原始权重
                    if isinstance(module, torch.nn.Linear):
                        module.weight.data[neuron_idx, :] = original_weight[neuron_idx, :]
                        if module.bias is not None:
                            module.bias.data[neuron_idx] = original_bias
                    elif isinstance(module, torch.nn.Conv2d):
                        module.weight.data[neuron_idx, :, :, :] = original_weight[neuron_idx, :, :, :]
                        if module.bias is not None:
                            module.bias.data[neuron_idx] = original_bias
                
                neuron_importance.extend(layer_importance)
        
        # 归一化重要性分数
        self.neuron_importance = torch.tensor(neuron_importance)
        if len(self.neuron_importance) > 0:
            self.neuron_importance = self.neuron_importance / self.neuron_importance.sum() * 100
            # 计算重要性百分位数，用于子模型构建
            self.neuron_importance_percentiles = torch.argsort(self.neuron_importance)

    def compute_client_contributions(self):
        """
        计算客户端贡献度
        基于验证集性能来评估贡献度
        """
        if self.validation is None:
            # 如果没有验证集，使用数据量作为贡献度
            total_data = sum(len(client.train_data) for client in self.clients)
            for i, client in enumerate(self.clients):
                self.client_contributions[i] = len(client.train_data) / total_data
            return
        
        # 使用验证集性能评估贡献度
        contributions = []
        for i, client in enumerate(self.clients):
            # 使用客户端当前子模型在验证集上的性能作为贡献度
            eval_metric, _ = client.test(self.client_submodels[i], 'valid')
            contributions.append(eval_metric)
        
        # 归一化贡献度
        if sum(contributions) > 0:
            self.client_contributions = [c / sum(contributions) for c in contributions]
        else:
            self.client_contributions = [1.0 / len(self.clients) for _ in self.clients]

    def update_client_reputations(self):
        """
        更新客户端声誉
        r_i = e^(c_i * β)，然后归一化
        """
        # 计算声誉
        raw_reputations = [math.exp(c * self.beta) for c in self.client_contributions]
        
        # 归一化到[0, 100]
        if max(raw_reputations) > 0:
            self.client_reputations = [r / max(raw_reputations) * 100 for r in raw_reputations]
        else:
            self.client_reputations = [100.0 / len(self.clients) for _ in self.clients]

    def allocate_submodels(self):
        """
        子模型分配模块
        根据客户端声誉分配不同性能的子模型
        """
        if self.neuron_importance is None or len(self.neuron_importance) == 0:
            # 如果没有神经元重要性信息，直接使用全局模型
            for i in range(len(self.clients)):
                self.client_submodels[i] = copy.deepcopy(self.model)
            return
        
        for i, reputation in enumerate(self.client_reputations):
            # 根据声誉确定子模型应包含的神经元比例
            neuron_ratio = reputation / 100.0
            num_neurons_to_keep = max(1, int(len(self.neuron_importance) * neuron_ratio))
            
            # 选择重要性最高的神经元
            important_neuron_indices = self.neuron_importance_percentiles[-num_neurons_to_keep:]
            
            # 构建子模型（通过掩码实现）
            submodel = copy.deepcopy(self.model)
            self.apply_neuron_mask(submodel, important_neuron_indices)
            self.client_submodels[i] = submodel

    def apply_neuron_mask(self, model, keep_indices):
        """
        对模型应用神经元掩码，只保留指定的神经元
        """
        neuron_idx = 0
        keep_indices_set = set(keep_indices.tolist())
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                if isinstance(module, torch.nn.Linear):
                    num_neurons = module.out_features
                elif isinstance(module, torch.nn.Conv2d):
                    num_neurons = module.out_channels
                
                # 为不在保留列表中的神经元设置权重为0
                for local_idx in range(num_neurons):
                    global_idx = neuron_idx + local_idx
                    if global_idx not in keep_indices_set:
                        if isinstance(module, torch.nn.Linear):
                            module.weight.data[local_idx, :] = 0
                            if module.bias is not None:
                                module.bias.data[local_idx] = 0
                        elif isinstance(module, torch.nn.Conv2d):
                            module.weight.data[local_idx, :, :, :] = 0
                            if module.bias is not None:
                                module.bias.data[local_idx] = 0
                
                neuron_idx += num_neurons

    def dynamic_aggregation(self, client_submodels):
        """
        动态聚合模块
        基于神经元被聚合频率的加权机制
        """
        # 初始化聚合频率统计
        if self.neuron_aggregation_frequency is None:
            self.neuron_aggregation_frequency = {}
            for name, param in self.model.named_parameters():
                self.neuron_aggregation_frequency[name] = torch.zeros_like(param)
        
        # 计算当前轮次每个参数的聚合频率
        current_frequency = {}
        for name, param in self.model.named_parameters():
            current_frequency[name] = torch.zeros_like(param)
        
        # 统计每个参数在多少个客户端模型中非零
        for submodel in client_submodels:
            for name, param in submodel.named_parameters():
                mask = (param != 0).float()
                current_frequency[name] += mask
        
        # 更新全局聚合频率
        for name in current_frequency:
            self.neuron_aggregation_frequency[name] += current_frequency[name]
        
        # 动态聚合：频率越高的参数权重越小
        aggregated_params = {}
        for name, param in self.model.named_parameters():
            weighted_sum = torch.zeros_like(param)
            weight_sum = torch.zeros_like(param)
            
            for submodel in client_submodels:
                submodel_param = dict(submodel.named_parameters())[name]
                # 使用频率的倒数作为权重
                frequency = current_frequency[name] + 1e-8  # 避免除零
                weight = 1.0 / frequency
                
                # 只对非零参数进行聚合
                mask = (submodel_param != 0).float()
                weighted_sum += submodel_param * weight * mask
                weight_sum += weight * mask
            
            # 避免除零
            weight_sum = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
            aggregated_params[name] = weighted_sum / weight_sum
        
        # 更新全局模型
        for name, param in self.model.named_parameters():
            param.data = aggregated_params[name]
        
        # 更新所有客户端子模型的基础
        for i in range(len(self.clients)):
            for name, param in self.client_submodels[i].named_parameters():
                if name in aggregated_params:
                    # 保持子模型的掩码结构
                    mask = (param != 0).float()
                    param.data = aggregated_params[name] * mask

    def pack(self, client_id):
        """
        Pack the necessary information for the client's local training.
        """
        return {"model": copy.deepcopy(self.client_submodels[client_id])}

    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        """
        if model is None:
            model = self.model
        
        if self.test_data:
            model.eval()
            loss = 0
            eval_metric = 0
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
                loss += bmean_loss * len(batch_data[1])
                eval_metric += bmean_eval_metric * len(batch_data[1])
            eval_metric /= len(self.test_data)
            loss /= len(self.test_data)
            return eval_metric, loss
        else:
            return -1, -1

    def test_on_clients(self, round, dataflag='valid'):
        """
        Validate accuracies and losses on clients' local datasets
        """
        evals, losses = [], []
        for i, client in enumerate(self.clients):
            eval_value, loss = client.test(self.client_submodels[i], dataflag)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        """
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
        return

    def test(self, model, dataflag='valid'):
        """
        Evaluate the model with local data.
        """
        dataset = self.train_data if dataflag == 'train' else self.valid_data
        if dataset is None:
            return 0.0, 0.0
            
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        
        total_samples = 0
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
            total_samples += len(batch_data[1])
        
        if total_samples > 0:
            eval_metric /= total_samples
            loss /= total_samples
        
        return eval_metric, loss