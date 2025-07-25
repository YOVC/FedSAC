from builtins import print, sum
from torch.utils import tensorboard
import utils.fflow as flw  # 导入联邦学习流程模块
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard可视化工具
from scipy.stats import pearsonr  # 导入皮尔逊相关系数计算函数
import os
import csv


TensorWriter = SummaryWriter('./wzh/log/1001')  # 创建TensorBoard日志写入器
#  自定义日志记录器实现

class MyLogger(flw.Logger):
    def log(self, server=None, round=None,corrs_agg=None):
        """自定义日志记录方法，记录训练过程中的各项指标
        Args:
            server: 联邦学习服务器实例
            round: 当前通信轮次
            corrs_agg: 存储各轮次相关系数的字典
        """
        if server==None: return
        # 初始化输出字典
        if self.output == {}:
            self.output = {
                "meta":server.option,  # 记录配置参数
                "train_losses":[],     # 训练损失
                "train_losses_clients":[],  # 每个客户端的训练损失
                "test_accs":[],        # 测试准确率
                "test_losses":[],      # 测试损失
                "valid_losses":[],     # 验证损失
            }
        # 获取测试集上的性能指标
        test_metric, test_loss = server.test()
        # 获取客户端本地验证集上的性能指标
        valid_metrics, valid_losses = server.test_on_clients(self.current_round, 'valid')
        # 获取客户端本地训练集上的性能指标
        train_metrics, train_losses = server.test_on_clients(self.current_round, 'train')
        # 计算加权平均的训练损失（按客户端数据量加权） TODO client_vols是什么
        self.output['train_losses'].append(1.0*sum([ck * closs for ck, closs in zip(server.client_vols, train_losses)])/server.data_vol)
        # 计算加权平均的验证损失
        self.output['valid_losses'].append(1.0*sum([ck * closs for ck, closs in zip(server.client_vols, valid_losses)])/server.data_vol)
        # 记录每个客户端的训练损失
        self.output['train_losses_clients'].append(train_losses)

        # 记录测试准确率和损失
        self.output['test_accs'].append(test_metric)
        self.output['test_losses'].append(test_loss)
        # 打印当前轮次的性能指标
        print("Training Loss:", self.output['train_losses'][-1])
        print("valid Loss:", self.output['valid_losses'][-1])
        print("Testing Loss:", self.output['test_losses'][-1])
        print("Testing Accuracy:", self.output['test_accs'][-1])  
        print("Mean of testing Accuracy:", np.mean(self.output['test_accs'][-1]))
        print("Max of testing Accuracy:", np.max(self.output['test_accs'][-1]))

        # 独立训练模型的测试准确率（用于计算公平性指标）
        # mnist-10 clients
        # powerlaw分布下的独立训练准确率
        standalone_test_acc = [0.3102, 0.387, 0.4048, 0.1554, 0.4775, 0.4996, 0.5291, 0.515, 0.5361, 0.5653]


        # 计算皮尔逊相关系数（衡量公平性指标）
        # 计算独立训练准确率与联邦学习模型准确率之间的相关性
        # 相关系数越高，表示公平性越好
        corrs = pearsonr(standalone_test_acc, self.output['test_accs'][-1])
        print("corrs:", corrs[0])  # 打印相关系数

        # 使用TensorBoard记录训练指标
        TensorWriter.add_scalar('Training Loss', self.output['train_losses'][-1], round)  # 记录训练损失
        TensorWriter.add_scalar('Mean of testing Accuracy', np.mean(self.output['test_accs'][-1]), round)  # 记录平均测试准确率
        TensorWriter.add_scalar('Max of testing Accuracy', np.max(self.output['test_accs'][-1]), round)  # 记录最大测试准确率
        TensorWriter.add_scalar('corrs', corrs[0], round)  # 记录相关系数（公平性指标）
        TensorWriter.add_scalar('valid_losses', self.output['valid_losses'][-1], round)  # 记录验证损失

        # 存储当前轮次的相关系数
        corrs_agg[round] = corrs[0]
        # 按相关系数降序排序
        corrs_agg = sorted(corrs_agg.items(), key = lambda kv:kv[1], reverse = True)
        # 如果已经有10轮以上的记录，打印相关系数最高的前9轮
        if len(corrs_agg) >= 10 :
            max_corrs = corrs_agg[0:9:1]
            print("max_corrs:", max_corrs)  # 打印最高的相关系数（最佳公平性）

# 创建全局日志记录器实例
logger = MyLogger()

def main(): 
    """主函数，启动联邦学习系统"""
    # 读取命令行参数选项
    option = flw.read_option()
    # 设置随机种子，确保实验可重复性
    flw.setup_seed(option['seed'])
    # 初始化联邦学习服务器
    # 包括加载数据集、创建模型、初始化客户端等
    server = flw.initialize(option)
    # 启动联邦学习优化过程
    server.run()

if __name__ == '__main__':
    main()


