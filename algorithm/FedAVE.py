from hashlib import new
from http import client
from itertools import tee
from re import S
from struct import pack
from telnetlib import SE
from tkinter import E, W
from utils import fmodule  # 联邦学习模块工具
from .fedbase import BasicServer, BasicClient  # 基础服务器和客户端类
import copy  # 深拷贝
import math  # 数学函数
# 导入联邦学习模块中的梯度处理函数
from utils.fmodule import add_gradient_updates, flatten, \
            mask_grad_update_by_order, add_update_to_model, compute_grad_update, unflatten, proportion_grad_update_by_order
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch函数库
import numpy as np  # 数值计算库
from torch.linalg import norm  # 计算范数
from multiprocessing import Pool as ThreadPool  # 多线程处理
import scipy.stats  # 统计函数库，用于计算KL散度
from main import logger  # 日志记录器
import utils.fflow as flw  # 联邦学习流程工具
import os  # 操作系统接口

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None, validation =None):
        """FedAVE服务器初始化
        
        Args:
            option: 配置参数字典
            model: 全局模型
            clients: 客户端列表
            test_data: 测试数据集
            validation: 验证数据集
        """
        super(Server, self).__init__(option, model, clients, test_data, validation)
        
        # 初始化客户端声誉值，初始为0
        self.reputation = [0 for i in range(len(self.clients))]
        # 声誉更新的平滑系数alpha
        self.alpha_reputation = option['alpha_reputatio'] 
        # 为每个客户端创建一个模型副本
        self.model = [model for i in range(len(self.clients))]
        # Gamma参数：控制梯度归一化强度
        self.Gamma = option['Gamma']
        # Beta参数：控制声誉奖励分配的非线性程度
        self.Beta = option['Beta']
        # 记录每个客户端的贡献度
        self.contributions = []
        # 记录每个客户端的数据量
        self.vol_data_clients = [0 for i in range(len(self.clients))]
        

    def run(self):
        """启动联邦学习系统，迭代训练全局模型
        
        整个联邦学习过程的主循环，包括：
        1. 记录总时间
        2. 每轮迭代调用iterate方法
        3. 更新全局学习率
        4. 记录日志
        5. 保存结果
        """
        # 开始计时
        logger.time_start('Total Time Cost')
        # 用于存储相关系数的字典
        corrs_agg = {}
        # 主循环：遍历所有通信轮次
        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')  # 开始记录当前轮次时间
            self.iterate(round)  # 执行当前轮次的迭代
            self.global_lr_scheduler(round)  # 更新全局学习率
            logger.time_end('Time Cost')  # 结束当前轮次时间记录
            # 检查是否需要记录日志
            if logger.check_if_log(round, self.eval_interval): logger.log(self, round=round, corrs_agg=corrs_agg)

        print("=================End==================")
        logger.time_end('Total Time Cost')  # 结束总时间记录
        # 将结果保存为.json文件
        logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
        return
    
    def iterate(self, t):
        """执行一轮联邦学习迭代
        
        Args:
            t: 当前迭代轮次
            
        Returns:
            None
        """
        # 选择所有客户端参与本轮训练
        self.selected_clients = [i for i in range(self.num_clients)]
        # 保存训练前的模型状态
        models_before = copy.deepcopy(self.model)
        # 冻结每个客户端模型的梯度，防止在计算过程中被修改
        for i in range(self.num_clients):
            models_before[i].freeze_grad()

        # 计算服务器验证集上的损失分布
        loss_server = self.KL_test(self.model, 'test')
        # 计算每个客户端训练集上的损失分布
        loss_clients = self.KL_test(self.model, 'train')
        # 用于存储每个客户端的KL散度
        KL_clients = []
        KL = 0
        # 计算每个客户端数据分布与服务器数据分布之间的KL散度
        for i in range(len(self.clients)):
            # 确保比较的损失列表长度一致
            len_data = len(loss_clients[i])
            len_server = len(loss_server[i])
            if len_data < len_server :
                loss_server[i] = loss_server[i][:len_data]  # 截取服务器损失列表
            else:
                loss_clients[i] = loss_clients[i][:len_server]  # 截取客户端损失列表
            # 计算KL散度并取倒数（KL散度越小表示分布越相似，取倒数后值越大表示越相似）
            KL = self.KL_divergence(loss_clients[i], loss_server[i])
            KL = 1/(KL)  # 取倒数，使得分布相似度越高，值越大
            KL_clients.append(KL)
        # 将KL散度列表转换为张量
        KL_clients = torch.tensor(KL_clients)
        # 归一化KL散度，使其总和为1
        KL_clients = torch.div(KL_clients, sum(KL_clients))
        # 再次归一化，使最大值为1
        KL_clients = torch.div(KL_clients, torch.max(KL_clients))

        # 计算每个客户端的数据量相对比例
        max_clients_data = max([len(self.clients[i].train_data) for i in range(self.num_clients)])  # 找出最大数据量
        for i in range(len(self.clients)):
            # 计算每个客户端数据量相对于最大数据量的比例
            self.vol_data_clients[i] = torch.div(len(self.clients[i].train_data), max_clients_data)

        # 客户端本地训练
        ws, losses = self.communicate(self.selected_clients)  # 与选定的客户端通信，获取训练后的模型和损失
        
        # 计算每个客户端的梯度更新
        self.grads = []
        for i in range(len(self.clients)):
            # 计算训练前后模型的梯度差异
            gradient = compute_grad_update(old_model=self.model[i], new_model=ws[i])
            # 将梯度展平为一维向量
            flattened = flatten(gradient)
            # 计算梯度的范数（加小常数避免除零错误）
            norm_value = norm(flattened) + 1e-7 
            # 使用Gamma参数对梯度进行归一化处理，控制梯度更新的强度
            gradient = unflatten(torch.multiply(torch.tensor(self.Gamma), torch.div(flattened, norm_value)), gradient)
            # 添加到梯度列表
            self.grads.append(gradient)
        
        # 在验证集上评估每个客户端的模型性能
        acc__, loss__ = self.validation_(ws)   
        # 计算每个客户端的权重（基于准确率和KL散度）
        weights = [0 for i in range(len(self.clients))]
        for i in range(len(self.clients)):
            # 权重计算：准确率 * KL散度的平方（强调数据分布的重要性）
            weights[i] = acc__[i] * KL_clients[i] * KL_clients[i] 

        # 基于数据量计算客户端权重（FedAvg方式）
        weight_ = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]

        # 注释掉的代码：权重归一化的替代方法
        # sum_weights = sum(weights)
        # for i in range(len(weights)):
        #     weights[i] = weights[i] / sum_weights
        # weights = acc__

        # 聚合所有客户端的梯度
        aggregated_gradient = [torch.zeros(param.shape) for param in self.model[0].parameters()]  # 初始化聚合梯度
        # 按权重聚合每个客户端的梯度
        for gradient_, weight in zip(self.grads, weight_):
            aggregated_gradient = add_gradient_updates(aggregated_gradient, gradient_, weight=weight)

        # 对每个客户端的声誉进行更新（ARC模块：Adaptive Reputation Calculation）
        for i in range(self.num_clients):
            if t == 0:  # 第一轮直接使用计算的权重作为声誉值
                self.reputation[i] = weights[i]
            else:  # 后续轮次使用指数移动平均更新声誉值
                # alpha_reputation控制历史声誉的权重，(1-alpha_reputation)控制当前表现的权重
                self.reputation[i] = self.alpha_reputation * self.reputation[i] + (1-self.alpha_reputation)*weights[i]
            # 转换为张量并设置最小值，避免声誉值过小
            self.reputation[i] = torch.tensor(self.reputation[i])
            self.reputation[i] = torch.clamp(self.reputation[i], min=1e-3)  # 设置最小声誉值为1e-3
        # 将声誉列表转换为张量
        self.reputation = torch.tensor(self.reputation)
        # 归一化声誉值，使总和为1
        self.reputation = torch.div(self.reputation, sum(self.reputation))

        # 使用tanh函数对声誉值进行非线性变换（DGR模块：Dynamic Gradient Reward Distribution）
        # Beta参数控制非线性程度
        q_ratios = torch.tanh(self.Beta * self.reputation)
        # 归一化，使最大值为1
        q_ratios = torch.div(q_ratios, torch.max(q_ratios))

        # 将q_ratios与KL散度相乘，进一步考虑数据分布的影响
        for i in range(len(self.clients)):
            q_ratios[i] = q_ratios[i] * KL_clients[i]

        # 记录本轮的客户端贡献度
        self.contributions.append(q_ratios.numpy())

        # 将声誉张量转回列表形式
        self.reputation = self.reputation.tolist()
        # 从第二轮开始分配奖励（第一轮没有足够信息计算准确的声誉值）
        if t > 0 :
            for i in range(self.num_clients):
                # 根据客户端的q_ratios值，对聚合梯度进行掩码处理
                # mask_percentile决定保留多少比例的梯度更新
                reward_gradient = mask_grad_update_by_order(aggregated_gradient, mask_percentile=q_ratios[i], mode='layer')
                # 将奖励梯度应用到客户端模型
                self.model[i] = copy.deepcopy(add_update_to_model(models_before[i], reward_gradient))
        return 


    def pack(self, client_id):
        """打包客户端本地训练所需的信息
        
        任何压缩或加密操作都应在此处完成
        
        Args:
            client_id: 要与之通信的客户端ID
            
        Returns:
            包含客户端特定模型的字典
        """
        # 返回该客户端对应的模型副本
        return {"model" : copy.deepcopy(self.model[client_id])}
    
    def validation_(self, model_=None):
        """在服务器验证集上评估模型性能
        
        Args:
            model_: 需要评估的模型列表，默认为self.model
            
        Returns:
            eval_metrics: 每个客户端模型在验证集上的评估指标（如准确率）
            losses: 每个客户端模型在验证集上的损失值
        """
        # 如果未提供模型，使用当前服务器模型
        if model_==None: model_=self.model
        eval_metrics, losses = [], [] 
        # 评估每个客户端的模型
        for i in range(self.num_clients):
            model = model_[i]

            # 如果存在验证集
            if self.validation:
                model.eval()  # 设置模型为评估模式
                loss = 0
                eval_metric = 0
                # 获取验证数据加载器
                data_loader = self.calculator.get_data_loader(self.validation, batch_size=64)
                # 遍历验证数据批次
                for batch_id, batch_data in enumerate(data_loader):
                    # 计算批次的评估指标和损失
                    bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
                    # 累加加权损失和评估指标
                    loss += bmean_loss * len(batch_data[1])
                    eval_metric += bmean_eval_metric * len(batch_data[1])
                # 计算平均评估指标和损失
                eval_metric /= len(self.validation)
                loss /= len(self.validation)

                # 添加到结果列表
                eval_metrics.append(eval_metric)
                losses.append(loss)
            else: return -1,-1  # 如果没有验证集，返回-1
        return eval_metrics, losses


    def test(self, model_=None):
        """在服务器测试集上评估模型性能
        
        Args:
            model_: 需要评估的模型列表，默认为self.model
            
        Returns:
            eval_metrics: 每个客户端模型在测试集上的评估指标（如准确率）
            losses: 每个客户端模型在测试集上的损失值
        """
        # 如果未提供模型，使用当前服务器模型
        if model_==None: model_=self.model
        eval_metrics, losses = [], [] 
        # 评估每个客户端的模型
        for i in range(self.num_clients):
            model = model_[i]

            # 如果存在测试集
            if self.test_data:
                model.eval()  # 设置模型为评估模式
                loss = 0
                eval_metric = 0
                # 获取测试数据加载器
                data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
                # 遍历测试数据批次
                for batch_id, batch_data in enumerate(data_loader):
                    # 计算批次的评估指标和损失
                    bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
                    # 累加加权损失和评估指标
                    loss += bmean_loss * len(batch_data[1])
                    eval_metric += bmean_eval_metric * len(batch_data[1])
                # 计算平均评估指标和损失
                eval_metric /= len(self.test_data)
                loss /= len(self.test_data)

                # 添加到结果列表
                eval_metrics.append(eval_metric)
                losses.append(loss)
            else: return -1,-1  # 如果没有测试集，返回-1
        return eval_metrics, losses


    def test_on_clients(self, round, dataflag='valid'):
        """在客户端本地数据集上验证模型性能
        
        Args:
            round: 当前通信轮次
            dataflag: 选择使用训练数据或验证数据进行评估，默认为'valid'
            
        Returns:
            evals: 每个客户端模型在其本地数据集上的评估指标
            losses: 每个客户端模型在其本地数据集上的损失值
        """
        evals, losses = [], []
        # 遍历每个客户端及其对应的模型
        for c, model in zip(self.clients, self.model):
            # 在客户端本地数据上测试模型
            eval_value, loss = c.test(model, dataflag)
            # 收集评估结果
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses

    def KL_test(self, model_=None, data=None):
        """计算模型在不同数据集上的损失分布
        
        用于计算服务器验证集和客户端训练集上的损失分布，以便后续计算KL散度
        
        Args:
            model_: 需要评估的模型列表，默认为self.model
            data: 数据集类型，'test'表示服务器验证集，'train'表示客户端训练集
            
        Returns:
            losses: 每个客户端模型在指定数据集上的损失列表
        """
        # 如果未提供模型，使用当前服务器模型
        if model_==None: model_=self.model
        eval_metrics, losses = [], [] 
        a = [i for i in range(len(self.clients))]
        # 遍历每个客户端及其对应的模型
        for i,c in zip(a, self.clients):
            model = model_[i]

            if data == 'test':  # 在服务器验证集上计算损失
                model.eval()  # 设置模型为评估模式
                loss = 0
                KL_loss = []  # 存储每个批次的损失
                eval_metric = 0
                # 获取验证数据加载器
                data_loader = self.calculator.get_data_loader(self.validation, batch_size=64)
                # 遍历验证数据批次
                for batch_id, batch_data in enumerate(data_loader):
                    # 计算批次的评估指标和损失
                    bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
                    bmean_loss = [bmean_loss]  # 转换为列表
                    KL_loss.extend(bmean_loss)  # 添加到损失列表
                losses.append(KL_loss)  # 添加该客户端的损失列表

            elif data == 'train':  # 在客户端训练集上计算损失
                # 调用客户端的KL_test_方法获取训练集上的损失
                loss = c.KL_test_(model, data)
                losses.append(loss)  # 添加该客户端的损失列表

        return losses
    
    def KL_divergence(self, a, b):
        """计算两个分布之间的KL散度
        
        KL散度是衡量两个概率分布差异的指标，值越小表示分布越相似
        
        Args:
            a: 第一个概率分布（通常是客户端损失分布）
            b: 第二个概率分布（通常是服务器损失分布）
            
        Returns:
            KL: 两个分布之间的KL散度值
        """
        # 使用scipy.stats.entropy计算KL散度
        KL = scipy.stats.entropy(a, b)
        return KL

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        """FedAVE客户端初始化
        
        Args:
            option: 配置参数字典
            name: 客户端名称
            train_data: 训练数据集
            valid_data: 验证数据集
        """
        super(Client, self).__init__(option, name, train_data, valid_data)

    def train(self, model):
        """标准本地训练过程
        
        使用本地训练数据集训练传输的模型
        
        Args:
            model: 全局模型
            
        Returns:
            None
        """
        model.update_grad()  # 更新模型梯度状态
        model.train()  # 设置模型为训练模式
        # 获取训练数据加载器
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        # 获取优化器
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        # 进行多轮本地训练
        for iter in range(self.epochs):
            # 遍历每个批次的数据
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()  # 清除梯度
                # 计算损失
                loss = self.calculator.get_loss(model, batch_data)
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
        return

    def KL_test_(self, model, dataflag='valid'):
        """使用本地数据评估模型并收集损失分布
        
        用于计算客户端本地数据上的损失分布，以便后续计算KL散度
        
        Args:
            model: 要评估的模型
            dataflag: 选择要评估的数据集，'train'表示训练数据，'valid'表示验证数据
            
        Returns:
            losses: 模型在本地数据集上的损失列表
        """
        # 根据dataflag选择数据集
        dataset = self.train_data if dataflag=='train' else self.valid_data
        model.eval()  # 设置模型为评估模式
        loss = 0
        eval_metric = 0
        losses = []  # 存储每个批次的损失
        # 获取数据加载器
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        # 遍历每个批次的数据
        for batch_id, batch_data in enumerate(data_loader):
            # 计算评估指标和损失
            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
            bmean_loss = [bmean_loss]  # 转换为列表
            losses.extend(bmean_loss)  # 添加到损失列表
        return losses  # 返回损失列表

