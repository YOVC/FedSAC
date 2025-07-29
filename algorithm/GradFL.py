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
        """绘制美化的训练过程图表"""
        if not self.training_history['rounds']:
            logging.warning("没有训练数据可供绘制")
            return
        
        # 设置字体和样式
        self._setup_plot_style()
        
        # 创建保存图片的目录
        save_dir = os.path.join('fedtask', self.option['task'], 'plots')
        os.makedirs(save_dir, exist_ok=True)
        
        rounds = self.training_history['rounds']
        
        # 第一张图：本地训练后的准确率
        self._plot_local_training_accuracy(rounds, save_dir)
        
        # 第二张图：子模型分配后的准确率
        self._plot_submodel_assignment_accuracy(rounds, save_dir)
        
    
    def _setup_plot_style(self):
        """设置绘图样式和字体"""
        # 强制设置中文字体
        import matplotlib.font_manager as fm
        
        # 清除字体缓存并重建
        try:
            fm._rebuild()
        except:
            pass
        
        # 直接设置中文字体，优先使用Microsoft YaHei
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 强制设置字体编码
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
        
        # 验证字体设置
        try:
            # 创建一个临时图形来测试中文字体
            test_fig, test_ax = plt.subplots(figsize=(1, 1))
            test_ax.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(test_fig)
        except Exception as e:
            # 如果中文字体仍然有问题，回退到英文标签
            print(f"中文字体设置警告: {e}")
            self.use_english_labels = True
        else:
            self.use_english_labels = False
    
    def _get_color_palette(self, n_colors):
        """获取美观的颜色调色板"""
        # 使用专业的颜色调色板
        colors = [
            '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
            '#577590', '#F3722C', '#F8961E', '#F9844A', '#F9C74F',
            '#90E0EF', '#00B4D8', '#0077B6', '#023E8A', '#03045E'
        ]
        return colors[:n_colors] if n_colors <= len(colors) else colors * (n_colors // len(colors) + 1)
    
    def _plot_local_training_accuracy(self, rounds, save_dir):
        """绘制本地训练后的准确率图"""
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # 设置背景渐变
        ax.set_facecolor('#f8f9fa')
        
        client_accuracies = self.training_history['local_training']['client_accuracies']
        colors = self._get_color_palette(len(client_accuracies[0]) if client_accuracies else 0)
        
        # 根据字体设置选择标签语言
        if hasattr(self, 'use_english_labels') and self.use_english_labels:
            # 英文标签
            xlabel = 'Training Round'
            ylabel = 'Accuracy'
            title = 'Model Accuracy After Local Training'
            legend_title = 'Models'
            client_label = lambda i: f'Client {i} (Keep: {self.fixed_keep_ratios[i]:.1%})'
            avg_label = 'Average Client Accuracy'
            global_label = 'Global Model Accuracy'
        else:
            # 中文标签
            xlabel = '训练轮次'
            ylabel = '准确率'
            title = '本地训练后模型准确率'
            legend_title = '模型类型'
            client_label = lambda i: f'客户端 {i} (保留比例: {self.fixed_keep_ratios[i]:.1%})'
            avg_label = '客户端平均准确率'
            global_label = '聚合模型准确率'
        
        # 绘制每个客户端的准确率曲线
        if client_accuracies:
            for client_idx in range(len(client_accuracies[0])):
                client_acc_over_rounds = [round_acc[client_idx] for round_acc in client_accuracies]
                ax.plot(rounds, client_acc_over_rounds, 
                       label=client_label(client_idx), 
                       color=colors[client_idx], marker='o', markersize=5, 
                       linewidth=2.5, alpha=0.8, markerfacecolor='white', 
                       markeredgewidth=2, markeredgecolor=colors[client_idx])
        
        # 绘制平均准确率 - 突出显示
        if self.training_history['local_training']['avg_accuracy']:
            ax.plot(rounds, self.training_history['local_training']['avg_accuracy'], 
                   label=avg_label, linewidth=4, color='#E63946', 
                   marker='s', markersize=8, markerfacecolor='white', 
                   markeredgewidth=3, markeredgecolor='#E63946', alpha=0.9)
        
        # 绘制全局模型准确率 - 突出显示
        if self.training_history['local_training']['global_accuracy']:
            ax.plot(rounds, self.training_history['local_training']['global_accuracy'], 
                   label=global_label, linewidth=4, color='#2A9D8F', 
                   marker='^', markersize=8, markerfacecolor='white', 
                   markeredgewidth=3, markeredgecolor='#2A9D8F', alpha=0.9)
        
        # 美化图表
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 设置图例
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          frameon=True, fancybox=True, shadow=True, 
                          fontsize=11, title=legend_title, title_fontsize=12)
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.9)
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax.set_axisbelow(True)
        
        # 设置坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        
        # 保存图片
        plot1_path = os.path.join(save_dir, 'local_training_accuracy.png')
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        logging.info(f"本地训练准确率图表已保存至: {plot1_path}")
    
    def _plot_submodel_assignment_accuracy(self, rounds, save_dir):
        """绘制子模型分配后的准确率图"""
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # 设置背景
        ax.set_facecolor('#f8f9fa')
        
        submodel_accuracies = self.training_history['submodel_assignment']['client_accuracies']
        colors = self._get_color_palette(len(submodel_accuracies[0]) if submodel_accuracies else 0)
        
        # 根据字体设置选择标签语言
        if hasattr(self, 'use_english_labels') and self.use_english_labels:
            # 英文标签
            xlabel = 'Training Round'
            ylabel = 'Accuracy'
            title = 'Model Accuracy After Submodel Assignment'
            legend_title = 'Submodels'
            client_label = lambda i: f'Client {i} (Keep: {self.fixed_keep_ratios[i]:.1%})'
            avg_label = 'Average Submodel Accuracy'
        else:
            # 中文标签
            xlabel = '训练轮次'
            ylabel = '准确率'
            title = '子模型分配后模型准确率'
            legend_title = '子模型类型'
            client_label = lambda i: f'客户端 {i} (保留比例: {self.fixed_keep_ratios[i]:.1%})'
            avg_label = '子模型平均准确率'
        
        # 绘制每个客户端的子模型准确率曲线
        if submodel_accuracies:
            for client_idx in range(len(submodel_accuracies[0])):
                client_acc_over_rounds = [round_acc[client_idx] for round_acc in submodel_accuracies]
                submodel_rounds = rounds[1:] if len(rounds) > 1 else rounds
                if len(client_acc_over_rounds) == len(submodel_rounds):
                    ax.plot(submodel_rounds, client_acc_over_rounds, 
                           label=client_label(client_idx), 
                           color=colors[client_idx], marker='o', markersize=5, 
                           linewidth=2.5, alpha=0.8, markerfacecolor='white', 
                           markeredgewidth=2, markeredgecolor=colors[client_idx])
        
        # 绘制子模型平均准确率
        if self.training_history['submodel_assignment']['avg_accuracy']:
            submodel_rounds = rounds[1:] if len(rounds) > 1 else rounds
            if len(self.training_history['submodel_assignment']['avg_accuracy']) == len(submodel_rounds):
                ax.plot(submodel_rounds, self.training_history['submodel_assignment']['avg_accuracy'], 
                       label=avg_label, linewidth=4, color='#F77F00', 
                       marker='s', markersize=8, markerfacecolor='white', 
                       markeredgewidth=3, markeredgecolor='#F77F00', alpha=0.9)
        
        # 美化图表
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 设置图例
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          frameon=True, fancybox=True, shadow=True, 
                          fontsize=11, title=legend_title, title_fontsize=12)
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.9)
        
        # 设置网格和坐标轴
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        
        # 保存图片
        plot2_path = os.path.join(save_dir, 'submodel_assignment_accuracy.png')
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        logging.info(f"子模型分配准确率图表已保存至: {plot2_path}")
    
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