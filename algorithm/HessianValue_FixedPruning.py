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
        # standalone accuracy [0.4018, 0.2967, 0.3491, 0.4671, 0.3021, 0.3817, 0.389, 0.4454, 0.3589, 0.281]
        # 固定的裁剪比例设置（10个客户端）
        self.fixed_pruning_ratios = [0.2, 0.5, 0.4, 0.0, 0.45, 0.3, 0.25, 0.1, 0.35, 0.6]
        
        print(f"客户端裁剪比例设置: {self.fixed_pruning_ratios}")
        
        # 神经元重要性评估间隔
        self.neuron_importance_eval_interval = option.get('neuron_eval_interval', 1)
        
        # 神经元重要性分数
        self.neuron_importance = None
        self.neuron_importance_percentiles = None
        
        # 为每个客户端维护子模型
        self.client_submodels = [copy.deepcopy(model) for _ in range(len(self.clients))]
        
        # Hessian矩阵计算相关参数
        self.hessian_batch_size = option.get('hessian_batch_size', 32)  # 用于计算Hessian矩阵的批次大小
        self.hessian_samples = option.get('hessian_samples', 100)  # 用于计算Hessian矩阵的样本数量
        self.damping = option.get('damping', 1e-4)  # 阻尼系数，防止Hessian矩阵奇异
        
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
        核心迭代过程
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
        print("开始进行Hessian矩阵评估")
        
        # 3. Hessian矩阵评估（每1轮评估一次）
        if t % self.neuron_importance_eval_interval == 0:
            self.evaluate_neuron_importance_hessian()
        print("开始进行子模型分配")
        
        # 4. 子模型分配模块：为每个客户端构建子模型（基于固定裁剪比例）
        self.allocate_submodels_with_fixed_pruning()
        return

    def evaluate_neuron_importance_hessian(self):
        """
        神经元重要性评估模块
        使用Hessian矩阵对角线元素（二阶导数）评估每个神经元的重要性
        Hessian矩阵的对角线元素表示损失函数对参数的二阶导数，
        数值越大表示该参数对损失函数的影响越大，即神经元越重要
        """
        self.model.eval()
        
        # 如果没有验证数据，则无法计算Hessian矩阵
        if not self.validation:
            print("警告：没有验证数据，无法计算Hessian矩阵，将使用权重幅度作为替代")
            return
        
        # 准备用于计算Hessian矩阵的数据
        data_loader = self.calculator.get_data_loader(self.validation, batch_size=self.hessian_batch_size)
        samples = []
        for batch_id, batch_data in enumerate(data_loader):
            samples.append(batch_data)
            if len(samples) * self.hessian_batch_size >= self.hessian_samples:
                break
        
        print(f"使用{len(samples)}个批次（约{len(samples) * self.hessian_batch_size}个样本）计算Hessian矩阵")
        
        # 计算每个神经元的重要性（基于Hessian矩阵对角线元素）
        neuron_importance = []
        neuron_to_param_mapping = {}  # 记录每个神经元对应的参数索引
        param_index = 0
        
        # 首先构建神经元到参数的映射
        neuron_idx = 0
        for layer_idx, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                if isinstance(module, torch.nn.Linear):
                    num_neurons = module.out_features
                    params_per_neuron = module.in_features + (1 if module.bias is not None else 0)
                elif isinstance(module, torch.nn.Conv2d):
                    num_neurons = module.out_channels
                    params_per_neuron = module.in_channels * module.kernel_size[0] * module.kernel_size[1] + (1 if module.bias is not None else 0)
                
                for i in range(num_neurons):
                    neuron_to_param_mapping[neuron_idx + i] = list(range(param_index, param_index + params_per_neuron))
                    param_index += params_per_neuron
                
                neuron_idx += num_neurons
        
        # 计算损失函数对每个参数的梯度
        for batch_data in samples:
            inputs, targets = batch_data
            inputs, targets = inputs.to(self.model.get_device()), targets.to(self.model.get_device())
            
            # 计算损失
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # 计算梯度
            loss.backward(create_graph=True)
            
            # 收集所有参数的梯度
            all_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads.append(param.grad.view(-1))
                else:
                    print("警告：某些参数未参与梯度计算，将跳过")
            
            all_grads = torch.cat(all_grads)
            
            # 计算Hessian矩阵对角线元素（使用有限差分近似）
            diag_hessian = torch.zeros_like(all_grads)
            for i in range(len(all_grads)):
                # 对第i个参数的梯度计算二阶导数
                grad_i = all_grads[i]
                
                # 计算梯度对该参数的导数（二阶导数）
                self.model.zero_grad()
                grad_i.backward(retain_graph=True)
                
                # 收集所有参数的二阶导数
                hessian_i = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        hessian_i.append(param.grad.view(-1))
                    else:
                        print("警告：某些参数未参与二阶导数计算，将跳过")
                
                hessian_i = torch.cat(hessian_i)
                diag_hessian[i] = hessian_i[i]  # 只保留对角线元素
            
            # 为每个神经元计算Hessian矩阵对角线元素的平均值
            for neuron_id, param_indices in neuron_to_param_mapping.items():
                neuron_hessian = torch.mean(torch.abs(diag_hessian[param_indices])).item()
                neuron_importance.append(neuron_hessian)
        
        # 转换为tensor并归一化重要性分数
        self.neuron_importance = torch.tensor(neuron_importance)
        if len(self.neuron_importance) > 0:
            # 处理全零情况
            if self.neuron_importance.sum() == 0:
                # 如果所有重要性都为0，则赋予一个较小的重要性
                self.neuron_importance = torch.ones_like(self.neuron_importance) * 0.0001
                print("警告：所有神经元Hessian值为零，使用较小值代替")
            else:
                # 归一化到0-100范围
                self.neuron_importance = self.neuron_importance / self.neuron_importance.sum() * 100
            
            # 计算重要性百分位数，用于子模型构建
            # 注意：这里使用正序排序（从小到大），优先裁剪Hessian值较小的神经元
            self.neuron_importance_percentiles = torch.argsort(self.neuron_importance)
            print(f"神经元Hessian评估完成，共{len(self.neuron_importance)}个神经元")
            print(f"Hessian值范围: 最小={self.neuron_importance.min().item():.6f}, 最大={self.neuron_importance.max().item():.6f}")
            print(f"Hessian值均值: {self.neuron_importance.mean().item():.6f}, 标准差: {self.neuron_importance.std().item():.6f}")


    def allocate_submodels_with_fixed_pruning(self):
        """
        基于固定裁剪比例的子模型分配模块
        按照神经元重要性正序排序，优先裁剪重要性较小的参数
        """
        if self.neuron_importance is None or len(self.neuron_importance) == 0:
            # 如果没有神经元重要性信息，直接使用全局模型
            for i in range(len(self.clients)):
                self.client_submodels[i] = copy.deepcopy(self.model)
            return
        
        # 获取按神经元重要性排序的神经元索引（从最小到最大，正序）
        sorted_neuron_indices = self.neuron_importance_percentiles.tolist()
        total_neurons = len(sorted_neuron_indices)
        
        for i, pruning_ratio in enumerate(self.fixed_pruning_ratios):
            # 计算要裁剪的神经元数量
            num_neurons_to_prune = int(total_neurons * pruning_ratio)
            
            # 选择要保留的神经元（裁剪重要性最小的神经元）
            if num_neurons_to_prune >= total_neurons:
                # 如果裁剪比例过大，至少保留一个重要性最大的神经元
                neurons_to_keep = [sorted_neuron_indices[-1]]
                print(f"警告：客户端{i}的裁剪比例({pruning_ratio})过大，只保留重要性最大的神经元")
            else:
                # 保留重要性较大的神经元（从索引num_neurons_to_prune开始到末尾）
                neurons_to_keep = sorted_neuron_indices[num_neurons_to_prune:]
            
            print(f"客户端{i}: 裁剪比例={pruning_ratio:.1%}, 裁剪{num_neurons_to_prune}个神经元, 保留{len(neurons_to_keep)}个神经元")
            
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
        基于神经元被聚合频率的加权机制（与FedSAC一致）
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
            
            print(f"  客户端 {i} (裁剪比例{self.fixed_pruning_ratios[i]:.1%}): 精度 = {eval_metric:.4f}, 损失 = {loss:.4f}")
        
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