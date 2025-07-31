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
import os
import csv
from benchmark.cifar10.model.gradfl_models.global_neuron_importance import create_global_importance_calculator
from .client_resource_simulator import ClientResourceSimulator
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
        self.fixed_keep_ratios = [0.8, 0.4, 1.0, 0.7, 0.5, 0.75, 0.55, 0.6, 0.9, 0.45]
        
        # 为每个客户端维护子模型和子模型形状
        self.client_submodels = [copy.deepcopy(model) for _ in range(self.num_clients)]
        self.clients_models_shape = {i: {} for i in range(self.num_clients)}
        
        # 记录上一轮客户端信息
        self.last_client_info = {i: [] for i in range(self.num_clients)}
        
        # 数据集类别列表（用于梯度感知模式）
        self.class_list = None
        
        # 全局神经元重要性计算器
        self.global_importance_calculator = None
        self.global_importance_cache = None
        if self.mode == 'neuron' and validation is not None:
            self.global_importance_calculator = create_global_importance_calculator(
                model, self.calculator, cache_dir="./neuron_importance_cache"
            )
            logging.info("已初始化全局神经元重要性计算器")
        
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
        
        # CSV数据保存相关
        self.csv_save_dir = os.path.join('fedtask', option.get('task', 'default_task'), 'csv_results')
        os.makedirs(self.csv_save_dir, exist_ok=True)
        self.csv_file_path = os.path.join(self.csv_save_dir, '{}.csv'.format(self.mode))
        self._initialize_csv_file()
        
        # 初始化客户端资源模拟器
        self.enable_resource_simulation = option.get('enable_resource_simulation', True)
        if self.enable_resource_simulation:
            client_distributions = option.get('client_distributions', None)
            client_distributions = {
                    0: {'mean': 25.0, 'std': 1.5},  # 高端设备
                    1: {'mean': 25.0, 'std': 3.0},  # 高端设备 稳定性差
                    2: {'mean': 15.0, 'std': 1.5},  # 中端设备
                    3: {'mean': 15.0, 'std': 3},  # 中端设备 稳定性差
                    4: {'mean': 8.0, 'std': 1.5},   # 低端设备
                    5: {'mean': 8.0, 'std': 3.0},   # 低端设备 稳定性差
                    6: {'mean': 12.0, 'std': 2.0},  # 中间值
                    7: {'mean': 12.0, 'std': 2.0},  # 中间值
                    8: {'mean': 6.0, 'std': 1.5},   # 低性能设备
                    9: {'mean': 6.0, 'std': 0.5},   # 稳定低性能设备
                }
            # 根据是否提供外部分布参数选择创建方式
            if client_distributions is not None:
                # 使用外部传入的客户端分布参数
                logging.info("使用外部传入的客户端CPU效率分布参数")
                self.resource_simulator = ClientResourceSimulator.from_client_distributions(
                    client_distributions=client_distributions,
                    seed=option.get('seed', 42),
                )
            else:
                # 使用全局分布参数自动生成客户端分布
                logging.info("使用全局分布参数自动生成客户端CPU效率分布")
                self.resource_simulator = ClientResourceSimulator.from_global_distribution(
                    num_clients=self.num_clients,
                    global_mean=option.get('global_cpu_mean', 1.0),
                    global_std=option.get('global_cpu_std', 0.3),
                    std_ratio=option.get('cpu_std_ratio', 0.2),
                    seed=option.get('seed', 42),
                )            
            # 保存资源配置
            resource_config_path = os.path.join(self.csv_save_dir, 'resource_config.json')
            self.resource_simulator.save_config(resource_config_path)
        else:
            self.resource_simulator = None
            logging.info("资源模拟器已禁用")
        
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
            
            # 每5轮绘制一次图片（从第5轮开始），以便更快看到训练效果
            if round > 0 and round % 5 == 0:
                logging.info(f"第 {round} 轮：绘制训练进度图表")
                self.plot_training_progress()

        logging.info("==================== 训练完成 ====================")
        logger.time_end('Total Time Cost')
        
        # 绘制最终的训练过程图表
        self.plot_training_progress()
        
        # save results as .json file
        logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
        return

    def _initialize_csv_file(self):
        """初始化CSV文件，写入表头"""
        try:
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # 写入表头
                writer.writerow([
                    'round', 
                    'client_max_accuracy', 
                    'client_avg_accuracy', 
                    'global_accuracy'
                ])
            logging.info(f"CSV文件已初始化: {self.csv_file_path}")
        except Exception as e:
            logging.error(f"初始化CSV文件失败: {e}")

    def _save_round_data_to_csv(self, round_num, client_max_acc, client_avg_acc, global_acc):
        """保存单轮数据到CSV文件"""
        try:
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    round_num,
                    f"{client_max_acc:.6f}",
                    f"{client_avg_acc:.6f}",
                    f"{global_acc:.6f}"
                ])
        except Exception as e:
            logging.error(f"保存第{round_num}轮数据到CSV失败: {e}")

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
        # self.test_local_and_global_models(trained_models, t)

        # 6. 为下一轮构建客户端子模型（第一轮之后）
        if t < self.num_rounds:  # 确保不在最后一轮构建子模型
            # 6.1 如果使用神经元重要性模式，先计算全局神经元重要性
            if self.mode == 'neuron' and self.global_importance_calculator is not None:
                logging.info(f"开始计算全局神经元重要性")
                data_loader = self.calculator.get_data_loader(self.validation, batch_size=64)
                self.global_importance_cache = self.global_importance_calculator.compute_global_neuron_importance(
                    data_loader, force_recompute=(t % 5 == 0)
                )
                # 打印重要性统计摘要
                summary = self.global_importance_calculator.get_importance_summary()
            
            logging.info(f"开始构建客户端子模型")
            self.construct_client_submodels(self.selected_clients)
            
            logging.info(f"开始测试子模型精度")
            # self.test_local_and_global_models(self.client_submodels, t, isTestGlobal=False)
        # 7. 更新客户端信息
        if t < self.num_rounds:  # 只在非最后一轮更新客户端信息
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
            if self.mode == 'hetero':
                # 基于固定顺序选择通道
                self.construct_hetero_submodel(idx, rate)
            elif self.mode == 'fedavg':
                # 完整模型（FedAvg基准）
                self.construct_fedavg_submodel(idx)
            elif self.mode == 'awareWeight':
                # 基于权重大小的子模型生成
                self.construct_weight_submodel(idx, rate)
            elif self.mode == 'neuron':
                # 基于神经元重要性的子模型生成
                if self.global_importance_cache is None:
                    logging.warning(f"全局神经元重要性缓存为空，客户端 {idx} 将使用随机选择")
                self.construct_neuron_submodel(idx, rate)
            else:
                raise ValueError(f"不支持的子模型生成模式: {self.mode}")
    
    def get_model(self, model_name, model_rate):
        if model_name == 'resnet18':
            # ResNet18配置：4个残差块组，每组通道数分别为64,128,256,512
            hidden_size = [64, 128, 256, 512]
            num_blocks = [2, 2, 2, 2]  # 每组包含2个残差块
            # 添加datashape参数，确保输入通道数正确设置为3（RGB图像）
            model = resnet18(hidden_size=hidden_size, num_blocks=num_blocks, num_classes=10, model_rate=model_rate)
            # 确保模型在正确的设备上
            model = model.to(fmodule.device)
        return model
    
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
    
    def construct_fedavg_submodel(self, client_idx):
        """FedAVG方式的子模型生成（使用完整模型）"""
        # 使用模型的get_idx_hetero方法生成子模型索引，但使用rate=1表示使用完整模型
        self.model.get_idx_hetero(1)
        client_model_idx = copy.deepcopy(self.model.idx)
        self.model.clear_idx()
        
        # 从全局模型中提取子模型参数
        client_model_params = self.get_model_params(self.model, client_model_idx)
        
        # 创建新的子模型实例（使用完整模型）
        self.client_submodels[client_idx] = self.get_model(self.model_name, 1)
        
        # 加载参数到客户端子模型
        self.client_submodels[client_idx].load_state_dict(client_model_params)
        
        # 保存子模型形状信息
        self.clients_models_shape[client_idx] = copy.deepcopy(client_model_idx)
    
    def construct_weight_submodel(self, client_idx, rate):
        """基于权重大小的子模型生成"""
        # 使用模型的get_idx_weight方法生成子模型索引
        self.model.get_idx_weight(rate, 'l2')
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
    
    def construct_neuron_submodel(self, client_idx, rate):
        """基于神经元重要性的子模型生成"""
        # 使用模型的get_idx_neuron_from_global方法生成子模型索引
        self.model.get_idx_neuron_from_global(rate, self.global_importance_cache or {})
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
    
    def communicate_with(self, client_id):
        """
        与指定客户端通信，设置资源信息并进行训练
        """
        # 为客户端设置资源信息
        if hasattr(self, 'resource_simulator') and self.resource_simulator is not None:
            self.clients[client_id].set_resource_info(client_id, self.resource_simulator)
        
        # 调用父类的communicate_with方法
        return super().communicate_with(client_id)

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
            
            # 保存数据到CSV文件
            if results['local_models']['eval_metrics']:
                client_max_acc = max(results['local_models']['eval_metrics'])
                client_avg_acc = sum(results['local_models']['eval_metrics']) / len(results['local_models']['eval_metrics'])
                self._save_round_data_to_csv(round_num, client_max_acc, client_avg_acc, eval_metric)
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
        
        # 设置字体和样式
        self._setup_plot_style()
        
        # 创建保存图片的目录
        save_dir = os.path.join('fedtask', self.option['task'], 'plots')
        os.makedirs(save_dir, exist_ok=True)
        
        rounds = self.training_history['rounds']
        current_round = rounds[-1] if rounds else 0
        
        # 第一张图：本地训练后的准确率统计
        self._plot_local_training_statistics(rounds, save_dir, current_round)
        
        # 第二张图：子模型分配后的准确率统计
        self._plot_submodel_statistics(rounds, save_dir, current_round)
        
    def _setup_plot_style(self):
        """设置绘图样式和字体"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 设置现代化样式
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                plt.style.use('default')
        
        # 设置全局参数
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': '#f8f9fa',
            'axes.edgecolor': '#dee2e6',
            'axes.linewidth': 1.2,
            'grid.color': '#e9ecef',
            'grid.alpha': 0.7,
            'text.color': '#212529',
            'axes.labelcolor': '#495057',
            'xtick.color': '#6c757d',
            'ytick.color': '#6c757d',
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.titlesize': 18
        })
    
    def _plot_local_training_statistics(self, rounds, save_dir, current_round):
        """绘制本地训练后的准确率统计（最大值、最小值、平均值）和聚合模型准确率"""
        if not self.training_history['local_training']['client_accuracies']:
            logging.warning("没有本地训练数据可供绘制")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 计算客户端准确率的统计信息
        client_accuracies = self.training_history['local_training']['client_accuracies']
        max_accuracies = [max(round_acc) for round_acc in client_accuracies]
        min_accuracies = [min(round_acc) for round_acc in client_accuracies]
        avg_accuracies = self.training_history['local_training']['avg_accuracy']
        global_accuracies = self.training_history['local_training']['global_accuracy']
        
        # 绘制线条
        ax.plot(rounds, max_accuracies, 'o-', color='#FF6B6B', linewidth=2.5, 
                markersize=6, label='client_max', alpha=0.8)
        ax.plot(rounds, min_accuracies, 'o-', color='#4ECDC4', linewidth=2.5, 
                markersize=6, label='client_min', alpha=0.8)
        ax.plot(rounds, avg_accuracies, 'o-', color='#45B7D1', linewidth=2.5, 
                markersize=6, label='client_avg', alpha=0.8)
        ax.plot(rounds, global_accuracies, 's-', color='#FFD700', linewidth=3, 
                markersize=7, label='global', alpha=0.9)
        
        # 设置图表样式
        ax.set_xlabel('round', fontsize=14, fontweight='bold')
        ax.set_ylabel('accuracy', fontsize=14, fontweight='bold')
        ax.set_title('local_training', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)
        
        # 保存图片
        filename = f'local_training_statistics_round_{current_round}.png'
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logging.info(f"本地训练统计图已保存至: {filepath}")
    
    def _plot_submodel_statistics(self, rounds, save_dir, current_round):
        """绘制子模型分配后的准确率统计（最大值、最小值、平均值）和聚合模型准确率"""
        if not self.training_history['submodel_assignment']['client_accuracies']:
            logging.warning("没有子模型分配数据可供绘制")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 计算子模型准确率的统计信息
        client_accuracies = self.training_history['submodel_assignment']['client_accuracies']
        max_accuracies = [max(round_acc) for round_acc in client_accuracies]
        min_accuracies = [min(round_acc) for round_acc in client_accuracies]
        avg_accuracies = self.training_history['submodel_assignment']['avg_accuracy']
        global_accuracies = self.training_history['local_training']['global_accuracy']
        
        # 子模型数据可能从第二轮开始，需要调整轮次
        submodel_rounds = rounds[-len(client_accuracies):] if len(rounds) > len(client_accuracies) else rounds
        
        # 绘制线条
        ax.plot(submodel_rounds, max_accuracies, 'o-', color='#FF9F43', linewidth=2.5, 
                markersize=6, label='submodel_max', alpha=0.8)
        ax.plot(submodel_rounds, min_accuracies, 'o-', color='#10AC84', linewidth=2.5, 
                markersize=6, label='submodel_min', alpha=0.8)
        ax.plot(submodel_rounds, avg_accuracies, 'o-', color='#5F27CD', linewidth=2.5, 
                markersize=6, label='submodel_avg', alpha=0.8)
        
        # 如果有子模型的全局准确率数据，也绘制出来
        if global_accuracies and len(global_accuracies) == len(submodel_rounds):
            ax.plot(submodel_rounds, global_accuracies, 's-', color='#FFD700', linewidth=3, 
                    markersize=7, label='global', alpha=0.9)
        
        # 设置图表样式
        ax.set_xlabel('round', fontsize=14, fontweight='bold')
        ax.set_ylabel('accuracy', fontsize=14, fontweight='bold')
        ax.set_title('submodel_accuracy', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)
        
        # 保存图片
        filename = f'submodel_statistics_round_{current_round}.png'
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logging.info(f"子模型统计图已保存至: {filepath}")
    
class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.client_id = None  # 将在服务器端设置
        self.resource_simulator = None  # 将在服务器端设置

    def set_resource_info(self, client_id, resource_simulator):
        """设置客户端资源信息"""
        self.client_id = client_id
        self.resource_simulator = resource_simulator

    def train(self, model, dataset=None, optimizer=None):
        """客户端训练方法，集成资源模拟""" 
        # 如果有资源模拟器，计算资源相关信息
        if hasattr(self, 'resource_simulator') and self.resource_simulator is not None and self.client_id is not None:
            # 计算模型MACs
            # 尝试获取输入形状（根据数据集类型）
            if dataset is not None and hasattr(dataset, '__iter__'):
                # 从数据加载器获取样本
                for batch_data in dataset:
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) > 0:
                        input_shape = batch_data[0].shape
                    else:
                        input_shape = batch_data.shape
                    break
                # 计算模型MACs
                model_macs = self.resource_simulator.calculate_model_macs(model, input_shape)
            elif self.train_data is not None and len(self.train_data) > 0:
                # 从训练数据获取样本
                sample_data = self.train_data[0]
                if isinstance(sample_data, (list, tuple)) and len(sample_data) > 0:
                    input_shape = sample_data[0].shape
                else:
                    input_shape = sample_data.shape
                model_macs = self.resource_simulator.calculate_model_macs(model, input_shape)

            # 计算训练时间
            training_time = self.resource_simulator.calculate_training_time(
                client_id=self.client_id,
                model_macs=model_macs,
                epochs=self.epochs
            )
            
            # 为当前轮采样CPU效率
            current_cpu_efficiency = self.resource_simulator.sample_cpu_efficiency(self.client_id)
            
            
            # 获取客户端配置
            profile = self.resource_simulator.get_client_profile(self.client_id)
            
            # 记录资源信息到日志
            logging.info(f"客户端 {self.client_id} 资源信息:")
            logging.info(f"  模型MACs: {model_macs:.2f}M")
            logging.info(f"  训练时间: {training_time:.2f}秒")
            logging.info(f"  CPU效率分布: 均值={profile.cpu_efficiency_mean:.2f}, 标准差={profile.cpu_efficiency_std:.2f} GFLOPS")
            logging.info(f"  当前轮CPU效率: {current_cpu_efficiency:.2f} GFLOPS")
            
            # 更新训练历史
            self.resource_simulator.update_training_history(
                client_id=self.client_id,
                round_num=getattr(self, 'current_round', 0),
                training_metrics={
                    'training_time': training_time,
                    'model_macs': model_macs,
                    'cpu_efficiency_mean': profile.cpu_efficiency_mean,
                    'cpu_efficiency_std': profile.cpu_efficiency_std,
                    'current_cpu_efficiency': current_cpu_efficiency
                }
            )
        
        logger.time_start(f'client {self.client_id} Train Time Cost')
        # 执行实际训练
        model.train()
        
        # 使用传入的dataset或默认的train_data
        if dataset is not None:
            data_loader = dataset
        else:
            data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        
        # 使用传入的optimizer或创建新的
        if optimizer is None:
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
        logger.time_end(f'client {self.client_id} Train Time Cost')
        return
    
    def _calculate_model_size(self, model):
        """计算模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            # nelement()获取参数的元素总数，element_size()获取每个元素的字节大小
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb