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
        self.client_reputations = [0.0 for _ in range(len(self.clients))]
        self.client_contributions = [0.3102, 0.387, 0.4048, 0.1554, 0.4775, 0.4996, 0.5291, 0.515, 0.5361, 0.5653]
        
        # 神经元重要性分数
        self.neuron_importance = None
        self.neuron_importance_percentiles = None
        
        # 为每个客户端维护子模型
        self.client_submodels = [copy.deepcopy(model) for _ in range(len(self.clients))]
        
        # 神经元聚合频率统计
        self.neuron_aggregation_frequency = None
        
        # 归一化贡献度（选择性操作）
        # self.compute_client_contributions()

        # 更新客户端声誉（贡献度在初始化时已计算，保持不变）
        self.update_client_reputations()
        
    def run(self):
        """
        Start the federated learning system where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        # 用于存储相关系数的字典
        corrs_agg = {}
        for round in range(self.num_rounds + 1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')
            self.iterate(round)
            self.global_lr_scheduler(round)
            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): 
                logger.log(self, round=round, corrs_agg=corrs_agg)

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
        print("开始第{}轮训练".format(t))
        # 1. 客户端本地训练
        trained_models, losses = self.communicate(self.selected_clients)
        print("开始进行动态聚合")
        # 2. 动态聚合模块：基于神经元频率的动态聚合
        self.dynamic_aggregation(trained_models, round_num=t)
        # 测试本地训练完成后的模型精度和聚合后的全局模型精度
        test_results = self.test_local_and_global_models(trained_models, t)
        print("开始进行神经元重要性评估")
        # 3. 神经元重要性评估（每10轮评估一次）
        if t % self.neuron_importance_eval_interval == 0:
            self.evaluate_neuron_importance()
        print("开始进行子模型分配")
        # 4. 子模型分配模块：为每个客户端构建子模型（基于更新后的神经元重要性）
        self.allocate_submodels()
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
                    # TODO 使用ReLU函数确保重要性为非负值
                    importance = max(importance, 0)
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
            # 处理全零情况
            if self.neuron_importance.sum() == 0:
                # 如果所有重要性都为0，则赋予一个较小的重要性
                self.neuron_importance = torch.ones_like(self.neuron_importance) * 0.0001
                print("警告：所有神经元重要性评估为零，使用较小值代替")
            self.neuron_importance = self.neuron_importance / self.neuron_importance.sum() * 100
            # 计算重要性百分位数，用于子模型构建
            self.neuron_importance_percentiles = torch.argsort(self.neuron_importance)
    def compute_client_contributions(self):
        # 归一化贡献度
        if sum(self.client_contributions) > 0:
            self.client_contributions = [c / sum(self.client_contributions) for c in self.client_contributions]
        else:
            self.client_contributions = [1.0 / len(self.clients) for _ in self.clients]
        print("Client contributions :", self.client_contributions)

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
        实现公式：θ_i = quantity(r_i, ∑_{n_i ∈ S} I_{n_i})
        其中，quantity表示当r_i = ∑_{n_i ∈ S} I_{n_i}时的子模型θ_i
        S代表模型中所有神经元的位置，按照从最不重要到最重要的顺序排列
        """
        if self.neuron_importance is None or len(self.neuron_importance) == 0:
            # 如果没有神经元重要性信息，直接使用全局模型
            for i in range(len(self.clients)):
                self.client_submodels[i] = copy.deepcopy(self.model)
            return
        # 获取按重要性排序的神经元索引（从最不重要到最重要）
        sorted_neuron_indices = self.neuron_importance_percentiles.tolist()
        for i, reputation in enumerate(self.client_reputations):
            # 根据声誉值r_i确定要保留的神经元
            # 从最重要的神经元开始选择，直到累积重要性达到声誉值
            cumulative_importance = 0.0
            neurons_to_keep = []
            # 从最不重要的神经元开始（倒序遍历）
            for idx in sorted_neuron_indices:
                neurons_to_keep.append(idx)
                cumulative_importance += self.neuron_importance[idx].item()
                # 当累积重要性达到或超过声誉值时停止
                if cumulative_importance >= reputation:
                    break
            # 确保至少保留一个神经元
            if not neurons_to_keep:
                neurons_to_keep = [sorted_neuron_indices[-1]]  # 保留最重要的神经元
            # 构建子模型（通过掩码实现）
            submodel = copy.deepcopy(self.model)
            self.apply_neuron_mask(submodel, torch.tensor(neurons_to_keep))
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

    def dynamic_aggregation(self, client_trained_models, round_num=0):
        """
        动态聚合模块
        基于神经元被聚合频率的加权机制
        """
        # 计算当前轮次每个参数的聚合频率
        current_frequency = {}
        for name, param in self.model.named_parameters():
            current_frequency[name] = torch.zeros_like(param)
        # 第一轮特殊处理：所有神经元的分配频次都为客户端数量
        if round_num == 0:
            # 第一轮时，所有神经元都被所有客户端使用
            for name, param in self.model.named_parameters():
                current_frequency[name] = torch.ones_like(param) * len(self.clients)
        else:
            # 后续轮次：使用上一轮分配的子模型来计算掩码
            for i, allocated_submodel in enumerate(self.client_submodels):
                if allocated_submodel is not None:
                    for name, param in allocated_submodel.named_parameters():
                        mask = (param != 0).float()
                        current_frequency[name] += mask

        # 动态聚合：频率越高的参数权重越小
        aggregated_params = {}
        for name, param in self.model.named_parameters():
            weighted_sum = torch.zeros_like(param)
            if round_num == 0:
                # 第一轮：简单平均聚合
                for i, trained_model in enumerate(client_trained_models):
                    trained_param = dict(trained_model.named_parameters())[name]
                    weighted_sum += trained_param
                aggregated_params[name] = weighted_sum / len(client_trained_models)
                    
            else:
                # 后续轮次：使用训练后的模型进行聚合，但权重基于上一轮分配的子模型掩码
                for i, trained_model in enumerate(client_trained_models):
                    trained_param = dict(trained_model.named_parameters())[name]
                    
                    allocated_param = dict(self.client_submodels[i].named_parameters())[name]
                    # 只聚合被分配的神经元（非零的部分）
                    allocation_mask = (allocated_param != 0).float()
                    # 使用频率的倒数作为权重
                    frequency = current_frequency[name] + 1e-8  # 避免除零
                    weight = allocation_mask / frequency  # 只对分配的神经元计算权重
                    
                    weighted_sum += trained_param * weight
                aggregated_params[name] = weighted_sum
        
        # 检查聚合后的参数是否合理
        if torch.isnan(aggregated_params[name]).any():
            print(f"警告：参数 {name} 聚合后包含NaN值")
        if torch.isinf(aggregated_params[name]).any():
            print(f"警告：参数 {name} 聚合后包含无穷值")

        # 更新全局模型
        for name, param in self.model.named_parameters():
            param.data = aggregated_params[name]
        
        # # 聚合BatchNorm层的统计信息（running_mean和running_var）
        # aggregated_buffers = {}
        # for name, buffer in self.model.named_buffers():
        #     if 'running_mean' in name or 'running_var' in name:
        #         weighted_sum = torch.zeros_like(buffer)
                
        #         # 对于BatchNorm统计信息，始终使用简单平均聚合
        #         for i, trained_model in enumerate(client_trained_models):
        #             trained_buffer = dict(trained_model.named_buffers())[name]
        #             weighted_sum += trained_buffer
                
        #         aggregated_buffers[name] = weighted_sum / len(client_trained_models)
        #     elif 'num_batches_tracked' in name:
        #         # 对于num_batches_tracked，取最大值
        #         max_batches = torch.zeros_like(buffer)
        #         for i, trained_model in enumerate(client_trained_models):
        #             trained_buffer = dict(trained_model.named_buffers())[name]
        #             max_batches = torch.max(max_batches, trained_buffer)
        #         aggregated_buffers[name] = max_batches
        
        # # 更新全局模型的缓冲区
        # for name, buffer in self.model.named_buffers():
        #     if name in aggregated_buffers:
        #         buffer.data = aggregated_buffers[name]
        
    def pack(self, client_id):
        """
        Pack the necessary information for the client's local training.
        """
        return {"model": copy.deepcopy(self.client_submodels[client_id])}

    def test(self, model=None):
        """
        Evaluate each client's submodel on the test dataset owned by the server.
        Returns a list of evaluation metrics for each client's submodel to support fairness analysis.
        """
        if self.test_data:
            eval_metrics, losses = [], []
            
            # 评估每个客户端的子模型
            for i in range(len(self.clients)):
                submodel = self.client_submodels[i] if hasattr(self, 'client_submodels') and i < len(self.client_submodels) else self.model
                
                submodel.eval()
                loss = 0
                eval_metric = 0
                data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
                
                for batch_id, batch_data in enumerate(data_loader):
                    bmean_eval_metric, bmean_loss = self.calculator.test(submodel, batch_data)
                    loss += bmean_loss * len(batch_data[1])
                    eval_metric += bmean_eval_metric * len(batch_data[1])
                
                eval_metric /= len(self.test_data)
                loss /= len(self.test_data)
                
                eval_metrics.append(eval_metric)
                losses.append(loss)
            
            return eval_metrics, losses
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

    def test_local_and_global_models(self, trained_models, round_num):
        """
        测试本地训练完成后的模型精度和聚合后的全局模型精度
        
        Args:
            trained_models: 客户端本地训练完成后的模型列表
            round_num: 当前轮次
            
        Returns:
            dict: 包含本地模型和全局模型测试结果的字典
        """
        results = {
            'local_models': {'eval_metrics': [], 'losses': []},
            'global_model': {'eval_metric': 0.0, 'loss': 0.0}
        }
        
        if not self.test_data:
            print("警告：没有测试数据，无法进行模型评估")
            return results
        
        # 1. 测试本地训练完成后的模型精度
        print(f"轮次 {round_num}: 测试本地训练完成后的模型精度...")
        for i, trained_model in enumerate(trained_models):
            trained_model.eval()
            loss = 0
            eval_metric = 0
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            
            total_samples = 0
            for batch_id, batch_data in enumerate(data_loader):
                bmean_eval_metric, bmean_loss = self.calculator.test(trained_model, batch_data)
                loss += bmean_loss * len(batch_data[1])
                eval_metric += bmean_eval_metric * len(batch_data[1])
                total_samples += len(batch_data[1])
            
            if total_samples > 0:
                eval_metric /= total_samples
                loss /= total_samples
            
            results['local_models']['eval_metrics'].append(eval_metric)
            results['local_models']['losses'].append(loss)
            
            print(f"  客户端 {i}: 精度 = {eval_metric:.4f}, 损失 = {loss:.4f}")
        
        # 计算本地模型的平均性能
        if results['local_models']['eval_metrics']:
            avg_local_metric = sum(results['local_models']['eval_metrics']) / len(results['local_models']['eval_metrics'])
            avg_local_loss = sum(results['local_models']['losses']) / len(results['local_models']['losses'])
            print(f"  本地模型平均: 精度 = {avg_local_metric:.4f}, 损失 = {avg_local_loss:.4f}")
        
        # 2. 测试聚合后的全局模型精度
        print(f"轮次 {round_num}: 测试聚合后的全局模型精度...")
        self.model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
        
        total_samples = 0
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss = self.calculator.test(self.model, batch_data)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
            total_samples += len(batch_data[1])
        
        if total_samples > 0:
            eval_metric /= total_samples
            loss /= total_samples
        
        results['global_model']['eval_metric'] = eval_metric
        results['global_model']['loss'] = loss
        
        print(f"  全局模型: 精度 = {eval_metric:.4f}, 损失 = {loss:.4f}")
        
        return results


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