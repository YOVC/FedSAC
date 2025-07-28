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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 配置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None, validation=None):
        super(Server, self).__init__(option, model, clients, test_data, validation)
        
        # GradFL specific parameters
        self.mode = option.get('mode', 'awareGrad')  # 子模型生成策略：awareGrad, roll, rand, hetero, fedavg
        self.select_mode = option.get('select_mode', 'absmax')  # 选择模式：absmax, probs, absmin
        self.model_name = option.get('model', 'resnet18')
        self.fixed_keep_ratios = [0.8, 0.5, 0.9, 0.7, 0.6, 0.75, 0.4, 0.55, 1.0, 0.45]
        
        # 为每个客户端维护子模型和子模型形状
        self.client_submodels = [copy.deepcopy(model) for _ in range(self.num_clients)]
        self.clients_models_shape = {i: {} for i in range(self.num_clients)}
        
        # 记录上一轮客户端信息
        self.last_client_info = {i: [] for i in range(self.num_clients)}
        
        # 数据集类别列表（用于梯度感知模式）
        self.class_list = None
        
        # 用于绘图的数据存储
        self.training_history = {
            'rounds': [],
            'local_training': {
                'client_accuracies': [],  # 每轮每个客户端的准确率
                'avg_accuracy': [],       # 每轮客户端平均准确率
                'global_accuracy': []     # 每轮聚合后全局模型准确率
            },
            'submodel_assignment': {
                'client_accuracies': [],  # 每轮子模型分配后每个客户端的准确率
                'avg_accuracy': [],       # 每轮子模型分配后客户端平均准确率
                'global_accuracy': []     # 每轮子模型分配后全局模型准确率（如果有的话）
            }
        }
        
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
        
        # 绘制训练过程图表
        self.plot_training_progress()
        
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
        
        # 3. 客户端本地训练
        logging.info(f"开始客户端本地训练")
        trained_models, _ = self.communicate(self.selected_clients)
        
        # 4. 聚合客户端模型更新全局模型
        logging.info(f"开始聚合客户端模型")
        self.aggregate_client_models(trained_models, self.selected_clients, t, isUseGlobal=False)
        
        # 5. 测试模型性能
        self.test_local_and_global_models(trained_models, t)

        # 6. 为下一轮构建客户端子模型（第一轮之后）
        if t < self.num_rounds:  # 确保不在最后一轮构建子模型
            logging.info(f"开始构建客户端子模型")
            self.construct_client_submodels(self.selected_clients)

        logging.info(f"开始测试子模型精度")
        self.test_local_and_global_models(self.client_submodels, t, isTestGlobal=False)
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
            elif self.mode == 'awareWeight':
                # 基于权重大小的子模型生成
                self.construct_weight_aware_submodel(idx, rate)
            else:
                raise ValueError(f"不支持的子模型生成模式: {self.mode}")
    
    def get_model(self, model_name, mode_rate):
        if model_name == 'resnet18':
            # ResNet18配置：4个残差块组，每组通道数分别为64,128,256,512
            hidden_size = [64, 128, 256, 512]
            num_blocks = [2, 2, 2, 2]  # 每组包含2个残差块
            # 添加datashape参数，确保输入通道数正确设置为3（RGB图像）
            model = resnet18(hidden_size=hidden_size, num_blocks=num_blocks, num_classes=10, model_rate=mode_rate)
            # 确保模型在正确的设备上
            model = model.to(fmodule.device)
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
        self.client_submodels[client_idx] = self.get_model(self.model_name, rate).to(fmodule.device)
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
        
        # 创建新的子模型实例
        self.client_submodels[client_idx] = self.get_model(self.model_name, rate)
        
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
        
        # 创建新的子模型实例
        self.client_submodels[client_idx] = self.get_model(self.model_name, rate)
        
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
        
        # 创建新的子模型实例
        self.client_submodels[client_idx] = self.get_model(self.model_name, rate)
        
        # 加载参数到客户端子模型
        self.client_submodels[client_idx].load_state_dict(client_model_params)
        
        # 保存子模型形状信息
        self.clients_models_shape[client_idx] = copy.deepcopy(client_model_idx)
    
    def construct_fedavg_submodel(self, client_idx, rate=1):
        """FedAVG方式的子模型生成（使用完整模型）"""
        # 使用模型的get_idx_hetero方法生成子模型索引，但使用rate=1表示使用完整模型
        self.model.get_idx_hetero(1)
        client_model_idx = copy.deepcopy(self.model.idx)
        self.model.clear_idx()
        
        # 从全局模型中提取子模型参数
        client_model_params = self.get_model_params(self.model, client_model_idx)
        
        # 创建新的子模型实例（使用完整模型）
        self.client_submodels[client_idx] = self.get_model(self.model_name, rate)
        
        # 加载参数到客户端子模型
        self.client_submodels[client_idx].load_state_dict(client_model_params)
        
        # 保存子模型形状信息
        self.clients_models_shape[client_idx] = copy.deepcopy(client_model_idx)
    
    def construct_weight_aware_submodel(self, client_idx, rate):
        """基于权重大小的子模型生成"""
        # 收集模型权重信息
        weights = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and (param.dim() > 1):
                # 使用权重的绝对值作为重要性指标
                weights[name] = param.data.abs().mean(dim=tuple(range(1, param.dim()))).clone()
        
        # 使用模型的get_idx_aware_weight方法基于权重大小生成子模型索引
        self.model.get_idx_aware_weight(rate, self.select_mode, weights)
        client_model_idx = copy.deepcopy(self.model.idx)
        self.model.clear_idx()
        
        # 从全局模型中提取子模型参数
        client_model_params = self.get_model_params(self.model, client_model_idx)
        
        # 创建新的子模型实例
        self.client_submodels[client_idx] = self.get_model(self.model_name, rate)

        
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
                    # 确保所有张量在同一设备上
                    indices = [idx.to(v.device) for idx in client_model_idx[k]]
                    client_model_params[k] = copy.deepcopy(v[torch.meshgrid(indices, indexing='ij')])
                else:
                    # 确保索引在与参数相同的设备上
                    client_model_params[k] = copy.deepcopy(v[client_model_idx[k].to(v.device)])
            else:
                raise NameError(f"无法匹配参数: {k}")
        return client_model_params
    
    def aggregate_client_models(self, client_models, client_indices, t, isUseGlobal=True):
        """聚合客户端模型更新全局模型"""
        # 初始化全局模型参数
        global_temp_params = OrderedDict()
        client_num_model_param = OrderedDict()
        
        # 初始化参数计数器
        for k, v in self.model.state_dict().items():
            if isUseGlobal and t != 0:
                global_temp_params[k] = copy.deepcopy(v)
                client_num_model_param[k] = torch.ones_like(v)
            else:
                global_temp_params[k] = torch.zeros_like(v)
                client_num_model_param[k] = torch.zeros_like(v)
        
        # 聚合客户端模型参数
        for idx in client_indices:
            for k, v in client_models[idx].state_dict().items():
                temp_shape = self.clients_models_shape[idx][k]
                if k in global_temp_params:
                    if v.dim() > 1:
                        # 确保所有张量在同一设备上
                        indices = [idx_tensor.to(global_temp_params[k].device) for idx_tensor in temp_shape]
                        global_temp_params[k][torch.meshgrid(indices, indexing='ij')] += v
                        client_num_model_param[k][torch.meshgrid(indices, indexing='ij')] += 1
                    else:
                        # 确保索引在与参数相同的设备上
                        global_temp_params[k][temp_shape.to(global_temp_params[k].device)] += v
                        client_num_model_param[k][temp_shape.to(client_num_model_param[k].device)] += 1
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
            
    def test_local_and_global_models(self, trained_models, round_num, isTestGlobal=True):
        results = {
            'local_models': {'eval_metrics': [], 'losses': []},
            'global_model': {'eval_metric': 0.0, 'loss': 0.0}
        }
        
        if not self.test_data:
            logging.warning("没有测试数据，无法进行模型评估")
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
            
            logging.info(f"客户端 {i} (保留比例{self.fixed_keep_ratios[i]:.1%}): 精度={eval_metric:.4f}, 损失={loss:.4f}")
        
        # 计算本地模型的平均性能
        if results['local_models']['eval_metrics']:
            avg_local_metric = sum(results['local_models']['eval_metrics']) / len(results['local_models']['eval_metrics'])
            avg_local_loss = sum(results['local_models']['losses']) / len(results['local_models']['losses'])
            logging.info(f"本地模型平均性能 - 精度: {avg_local_metric:.4f}, 损失: {avg_local_loss:.4f}")
            
            # 记录本地训练数据
            if isTestGlobal:  # 只有在测试本地训练后的模型时才记录
                self.training_history['local_training']['client_accuracies'].append(results['local_models']['eval_metrics'].copy())
                self.training_history['local_training']['avg_accuracy'].append(avg_local_metric)
        
        if isTestGlobal:
            # 2. 测试聚合后的全局模型精度
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
            
            logging.info(f"聚合后 全局模型性能 - 精度: {eval_metric:.4f}, 损失: {loss:.4f}")
            
            # 记录全局模型数据
            self.training_history['local_training']['global_accuracy'].append(eval_metric)
            # 记录轮次
            if round_num not in self.training_history['rounds']:
                self.training_history['rounds'].append(round_num)
        else:
            # 记录子模型分配后的数据
            if results['local_models']['eval_metrics']:
                avg_local_metric = sum(results['local_models']['eval_metrics']) / len(results['local_models']['eval_metrics'])
                self.training_history['submodel_assignment']['client_accuracies'].append(results['local_models']['eval_metrics'].copy())
                self.training_history['submodel_assignment']['avg_accuracy'].append(avg_local_metric)
        
        return results

    def plot_training_progress(self):
        """绘制训练过程图表"""
        if not self.training_history['rounds']:
            logging.warning("没有训练数据可供绘制")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建保存图片的目录
        save_dir = os.path.join('fedtask', self.option['task'], 'plots')
        os.makedirs(save_dir, exist_ok=True)
        
        rounds = self.training_history['rounds']
        
        # 第一张图：本地训练后的准确率
        plt.figure(figsize=(12, 8))
        
        # 绘制每个客户端的准确率曲线
        client_accuracies = self.training_history['local_training']['client_accuracies']
        if client_accuracies:
            for client_idx in range(len(client_accuracies[0])):
                client_acc_over_rounds = [round_acc[client_idx] for round_acc in client_accuracies]
                plt.plot(rounds, client_acc_over_rounds, 
                        label=f'客户端 {client_idx} (保留比例{self.fixed_keep_ratios[client_idx]:.1%})', 
                        marker='o', markersize=4, alpha=0.7)
        
        # 绘制平均准确率
        if self.training_history['local_training']['avg_accuracy']:
            plt.plot(rounds, self.training_history['local_training']['avg_accuracy'], 
                    label='客户端平均准确率', linewidth=3, color='red', marker='s', markersize=6)
        
        # 绘制全局模型准确率
        if self.training_history['local_training']['global_accuracy']:
            plt.plot(rounds, self.training_history['local_training']['global_accuracy'], 
                    label='聚合后全局模型准确率', linewidth=3, color='black', marker='^', markersize=6)
        
        plt.xlabel('训练轮次')
        plt.ylabel('准确率')
        plt.title('本地训练后模型准确率变化')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存第一张图
        plot1_path = os.path.join(save_dir, 'local_training_accuracy.png')
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"本地训练准确率图表已保存至: {plot1_path}")
        
        # 第二张图：子模型分配后的准确率
        plt.figure(figsize=(12, 8))
        
        # 绘制每个客户端的子模型准确率曲线
        submodel_accuracies = self.training_history['submodel_assignment']['client_accuracies']
        if submodel_accuracies:
            for client_idx in range(len(submodel_accuracies[0])):
                client_acc_over_rounds = [round_acc[client_idx] for round_acc in submodel_accuracies]
                # 子模型分配是在第1轮之后开始的，所以轮次从1开始
                submodel_rounds = rounds[1:] if len(rounds) > 1 else rounds
                if len(client_acc_over_rounds) == len(submodel_rounds):
                    plt.plot(submodel_rounds, client_acc_over_rounds, 
                            label=f'客户端 {client_idx} (保留比例{self.fixed_keep_ratios[client_idx]:.1%})', 
                            marker='o', markersize=4, alpha=0.7)
        
        # 绘制子模型平均准确率
        if self.training_history['submodel_assignment']['avg_accuracy']:
            submodel_rounds = rounds[1:] if len(rounds) > 1 else rounds
            if len(self.training_history['submodel_assignment']['avg_accuracy']) == len(submodel_rounds):
                plt.plot(submodel_rounds, self.training_history['submodel_assignment']['avg_accuracy'], 
                        label='子模型平均准确率', linewidth=3, color='red', marker='s', markersize=6)
        
        plt.xlabel('训练轮次')
        plt.ylabel('准确率')
        plt.title('子模型分配后模型准确率变化')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存第二张图
        plot2_path = os.path.join(save_dir, 'submodel_assignment_accuracy.png')
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"子模型分配准确率图表已保存至: {plot2_path}")
        
        # 创建综合对比图
        plt.figure(figsize=(15, 10))
        
        # 子图1：本地训练后准确率
        plt.subplot(2, 1, 1)
        if self.training_history['local_training']['avg_accuracy']:
            plt.plot(rounds, self.training_history['local_training']['avg_accuracy'], 
                    label='本地训练后平均准确率', linewidth=2, color='blue', marker='o')
        if self.training_history['local_training']['global_accuracy']:
            plt.plot(rounds, self.training_history['local_training']['global_accuracy'], 
                    label='聚合后全局模型准确率', linewidth=2, color='green', marker='^')
        plt.xlabel('训练轮次')
        plt.ylabel('准确率')
        plt.title('本地训练后模型准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：子模型分配后准确率
        plt.subplot(2, 1, 2)
        if self.training_history['submodel_assignment']['avg_accuracy']:
            submodel_rounds = rounds[1:] if len(rounds) > 1 else rounds
            if len(self.training_history['submodel_assignment']['avg_accuracy']) == len(submodel_rounds):
                plt.plot(submodel_rounds, self.training_history['submodel_assignment']['avg_accuracy'], 
                        label='子模型分配后平均准确率', linewidth=2, color='red', marker='s')
        plt.xlabel('训练轮次')
        plt.ylabel('准确率')
        plt.title('子模型分配后模型准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存综合对比图
        plot3_path = os.path.join(save_dir, 'training_progress_comparison.png')
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"训练过程对比图表已保存至: {plot3_path}")

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