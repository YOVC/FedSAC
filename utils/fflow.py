import numpy as np        # 导入数值计算库
import argparse           # 导入命令行参数解析库
import random            # 导入随机数生成库
import torch             # 导入PyTorch深度学习库
import os.path           # 导入路径操作库
import importlib         # 导入动态导入模块库
import os                # 导入操作系统接口库
import utils.fmodule     # 导入联邦学习模块工具
import ujson             # 导入高性能JSON处理库
import time              # 导入时间处理库

# 客户端采样策略列表
sample_list=['uniform', 'md', 'active']  # uniform: 均匀采样, md: 基于数据量采样, active: 活跃度采样
# 模型聚合策略列表
agg_list=['uniform', 'weighted_scale', 'weighted_com']  # uniform: 均匀聚合, weighted_scale: 按数据量加权, weighted_com: 按通信代价加权
# 优化器列表
optimizer_list=['SGD', 'Adam']  # 支持的优化器类型

def read_option():
    parser = argparse.ArgumentParser()
    # 基本设置
    parser.add_argument('--task', help='联邦学习任务名称', type=str, default='mnist_cnum10_dist0_skew0_seed0')
    parser.add_argument('--algorithm', help='联邦学习算法名称', type=str, default='FedSAC')
    parser.add_argument('--model', help='模型名称', type=str, default='mlp')

    # 服务器端的客户端采样和模型聚合方法
    parser.add_argument('--sample', help='客户端采样方法', type=str, choices=sample_list, default='md')
    parser.add_argument('--aggregate', help='模型聚合方法', type=str, choices=agg_list, default='uniform')
    parser.add_argument('--learning_rate_decay', help='学习率衰减系数', type=float, default=0.977)
    parser.add_argument('--weight_decay', help='权重衰减系数（L2正则化）', type=float, default=0)
    parser.add_argument('--lr_scheduler', help='全局学习率调度器类型', type=int, default=-1)
    # 服务器端训练的超参数
    parser.add_argument('--num_rounds', help='通信轮次数', type=int, default=20)
    parser.add_argument('--proportion', help='每轮采样的客户端比例', type=float, default=0.2)
    # 本地训练的超参数
    parser.add_argument('--num_epochs', help='客户端本地训练的轮次数', type=int, default=5)
    parser.add_argument('--learning_rate', help='本地求解器的学习率', type=float, default=0.1)
    parser.add_argument('--batch_size', help='客户端训练的批量大小', type=int, default=32)
    parser.add_argument('--optimizer', help='选择梯度下降优化器', type=str, choices=optimizer_list, default='SGD')
    parser.add_argument('--momentum', help='本地更新的动量系数', type=float, default=0)

    # 机器环境设置
    parser.add_argument('--seed', help='随机初始化的种子', type=int, default=0)
    parser.add_argument('--gpu', help='GPU ID，-1表示使用CPU', type=int, default=1)
    parser.add_argument('--eval_interval', help='每隔多少轮进行一次评估', type=int, default=1)
    parser.add_argument('--num_threads', help="客户端计算会话中的线程数量", type=int, default=1)
    parser.add_argument('--train_on_all', help='使用训练数据和验证数据一起训练模型', action="store_true", default=True)

    # 客户端系统模拟设置
    # 构建网络异构性
    parser.add_argument('--net_drop', help="控制每轮通信中被选中客户端的丢弃率，服从Beta(drop,1)分布", type=float, default=0)
    parser.add_argument('--net_active', help="控制客户端活跃概率，服从Beta(active,1)分布", type=float, default=99999)
    # 构建计算能力异构性
    parser.add_argument('--capability', help="控制每个客户端本地计算能力的差异", type=float, default=0)

    # 不同算法的超参数
    # FedAVE算法相关参数
    parser.add_argument('--alpha_reputatio', help='上一轮和本轮声誉的权重', type=float, default='0.95')
    parser.add_argument('--Gamma', help='FedAVE中梯度归一化系数', type=float, default='0.5')
    parser.add_argument('--Beta', help='FedAVE中奖励比例系数', type=float, default='2.0')
    
    # 其他联邦学习算法的超参数
    parser.add_argument('--learning_rate_lambda', help='AFL中λ的学习率η', type=float, default=0)
    parser.add_argument('--q', help='q-FedAvg中的q参数', type=float, default='0.0')
    parser.add_argument('--epsilon', help='FedMGDA+中的ε参数', type=float, default='0.0')
    parser.add_argument('--eta', help='FedMGDA+中的全局学习率', type=float, default='1.0')
    parser.add_argument('--tau', help='FedFAvg中包含的最近历史梯度长度', type=int, default=0)
    parser.add_argument('--alpha', help='FedFV中保持原始方向的客户端比例/FedFA中的alpha参数', type=float, default='0.0')
    parser.add_argument('--beta', help='FedFA中的beta参数',type=float, default='1.0')
    parser.add_argument('--gamma', help='FedFA中的gamma参数', type=float, default='0')
    parser.add_argument('--mu', help='FedProx中的mu参数（近端项系数）', type=float, default='0.0')
    parser.add_argument('--sources', help='cf_TMC_ours中贡献轮次的更新数量', type=int, default='100')
    parser.add_argument('--standalone', help='客户端是否独立训练', type=float, default='0.0')
    parser.add_argument('--num_epochs_standalone', help='独立训练的通信轮次数', type=float, default='500')
    parser.add_argument('--standalone_loss', help='FedProx中的独立损失', type=float, default='0.0')
    # FedSAC和相关算法的参数
    parser.add_argument('--neuron_eval_interval', help='神经元重要性评估间隔', type=int, default=10)
    
    # GradFL算法相关参数
    parser.add_argument('--mode', help='GradFL子模型生成策略：awareGrad, roll, rand, hetero, fedavg', type=str, default='awareGrad')
    parser.add_argument('--select_mode', help='GradFL选择模式：absmax, probs, absmin', type=str, default='absmax')


    try: option = vars(parser.parse_args())       #vars()返回对象object的属性和属性值的字典对象
    except IOError as msg: parser.error(str(msg))
    return option

def setup_seed(seed):
    random.seed(1+seed)                     # 设置Python随机模块种子
    np.random.seed(21+seed)                # 设置NumPy随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子
    torch.manual_seed(12+seed)             # 为CPU设置PyTorch随机种子
    torch.cuda.manual_seed_all(123+seed)   # 为所有GPU设置PyTorch随机种子
    torch.backends.cudnn.benchmark = False    # 关闭cudnn的benchmark模式
    torch.backends.cudnn.deterministic = True  # 确保cudnn的结果是确定的

def initialize(option):
    """初始化联邦学习系统
    
    Args:
        option: 配置参数字典
        
    Returns:
        server: 初始化好的联邦学习服务器实例
    """
    # 初始化联邦学习任务
    print("init fedtask...", end='')
    # 根据基准动态初始化配置
    bmk_name = option['task'][:option['task'].find('cnum')-1].lower()  # 提取基准名称（如mnist, cifar10等）
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])  # 模型路径
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])  # 核心功能路径
    # 设置计算设备（GPU或CPU）
    utils.fmodule.device = torch.device('cuda:{}'.format(option['gpu']) if torch.cuda.is_available() and option['gpu'] != -1 else 'cpu')
    # 动态导入任务计算器
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    # 设置优化器
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), option['optimizer']))
    # 动态导入模型类
    if option['algorithm'] == 'GradFL':
        # 如果是GradFL算法，使用model_selector中的get_model函数选择模型
        model_selector_path = '.'.join(['benchmark', bmk_name, 'model', 'model_selector'])
        get_model = getattr(importlib.import_module(model_selector_path), 'get_model')
        utils.fmodule.Model = lambda: get_model(option['model'])
    else:
        # 其他算法使用原有的模型导入方式
        utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    # 创建任务读取器并读取数据
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', option['task']))
    # 读取训练数据、验证数据、服务器验证集、测试数据和客户端名称
    train_datas, valid_datas, validation, test_data, client_names = task_reader.read_data()

    num_clients = len(client_names)  # 客户端数量
    print("done")

    # 初始化客户端
    print('init clients...', end='')
    client_path = '%s.%s' % ('algorithm', option['algorithm'])  # 构建客户端类的导入路径
    Client=getattr(importlib.import_module(client_path), 'Client')  # 获取客户端类
    # 为每个客户端创建实例，传入配置、名称、训练数据和验证数据
    clients = [Client(option, name = client_names[cid], train_data = train_datas[cid], valid_data = valid_datas[cid]) for cid in range(num_clients)]
    print('done')

    # 初始化服务器
    print("init server...", end='')
    server_path = '%s.%s' % ('algorithm', option['algorithm'])  # 构建服务器类的导入路径
    # 创建服务器实例，传入配置、模型、客户端列表、测试数据和服务器验证集
    server = getattr(importlib.import_module(server_path), 'Server')(option, utils.fmodule.Model().to(utils.fmodule.device), clients, test_data = test_data, validation=validation)
    print('done')
    return server  # 返回初始化好的服务器实例

def output_filename(option, server):
    """生成输出文件名
    
    根据算法参数和服务器参数生成唯一的输出文件名
    
    Args:
        option: 配置参数字典
        server: 服务器实例
        
    Returns:
        str: 格式化的输出文件名
    """
    # 文件名头部：算法名
    header = "{}_".format(option["algorithm"])
    # 添加服务器特定参数
    for para in server.paras_name: header = header + para + "{}_".format(option[para])
    # 添加通用参数信息
    output_name = header + "M{}_R{}_B{}_E{}_LR{:.4f}_P{:.2f}_S{}_LD{:.3f}_WD{:.3f}_DR{:.2f}_AC{:.2f}.json".format(
        option['model'],           # 模型
        option['num_rounds'],      # 通信轮次
        option['batch_size'],      # 批量大小
        option['num_epochs'],      # 本地训练轮次
        option['learning_rate'],   # 学习率
        option['proportion'],      # 客户端采样比例
        option['seed'],            # 随机种子
        option['lr_scheduler']+option['learning_rate_decay'],  # 学习率调度
        option['weight_decay'],    # 权重衰减
        option['net_drop'],        # 网络丢包率
        option['net_active'])      # 客户端活跃度
    return output_name

class Logger:
    """日志记录器类
    
    用于记录联邦学习过程中的各种指标、时间消耗等信息
    """
    def __init__(self):
        """初始化日志记录器"""
        self.output = {}            # 存储输出数据的字典
        self.current_round = -1     # 当前通信轮次
        self.temp = "{:<30s}{:.4f}" # 格式化输出模板
        self.time_costs = []        # 时间消耗记录
        self.time_buf={}           # 时间缓冲区，用于计时

    def check_if_log(self, round, eval_interval=-1):
        """检查是否需要在当前轮次记录日志
        
        每隔'eval_interval'轮评估一次，检查是否在'round'轮记录日志
        
        Args:
            round: 当前轮次
            eval_interval: 评估间隔，默认为-1
            
        Returns:
            bool: 是否需要记录日志
        """
        self.current_round = round
        return eval_interval > 0 and (round == 0 or round % eval_interval == 0)

    def time_start(self, key = ''):
        """创建事件'key'的开始时间戳
        
        Args:
            key: 事件名称
        """
        if key not in [k for k in self.time_buf.keys()]:
            self.time_buf[key] = []
        self.time_buf[key].append(time.time())

    def time_end(self, key = ''):
        """创建事件'key'的结束时间戳并打印事件的时间间隔
        
        Args:
            key: 事件名称
            
        Raises:
            RuntimeError: 如果在开始计时前结束计时
        """
        if key not in [k for k in self.time_buf.keys()]:
            raise RuntimeError("Timer end before start.")
        else:
            self.time_buf[key][-1] =  time.time() - self.time_buf[key][-1]
            print("{:<30s}{:.4f}".format(key+":", self.time_buf[key][-1]) + 's')

    def save(self, filepath):
        """将self.output保存为.json文件
        
        Args:
            filepath: 输出文件路径
        """
        if self.output=={}: return
        with open(filepath, 'w') as outf:
            ujson.dump(self.output, outf)    # ujson.dump()生成一个fp的文件流
            
    def write(self, var_name=None, var_value=None):
        """将变量'var_name'及其值var_value添加到日志记录器
        
        Args:
            var_name: 变量名称
            var_value: 变量值
            
        Raises:
            RuntimeError: 如果缺少要记录的变量名称
            
        Returns:
            None
        """
        if var_name==None: raise RuntimeError("Missing the name of the variable to be logged.")
        if var_name in [key for key in self.output.keys()]:
            self.output[var_name] = []
        self.output[var_name].append(var_value)
        return

    def log(self, sever=None):
        """记录服务器状态（待实现）
        
        Args:
            sever: 服务器实例
        """
        pass
