"""resnet in pytorch with GradFL submodel construction support

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from utils.fmodule import FModule


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output
        
def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, rate=1, track=False):
        super().__init__()
        self.idx = OrderedDict()
        self.rate = rate
        self.scaler = Scaler(rate)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # residual function
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n1 = nn.BatchNorm2d(out_channels, momentum=None, track_running_stats=track)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(out_channels * BasicBlock.expansion, momentum=None, track_running_stats=track)

        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                Scaler(rate),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion, momentum=None, track_running_stats=track)
            )

    def forward(self, x):
        out = self.relu(self.n1(self.scaler(self.conv1(x))))
        out = self.n2(self.scaler(self.conv2(out)))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

    def get_idx_aware(self, input, first_channels, rate, param_name, topmode):
        out = self.relu(self.n1(self.scaler(self.conv1(input))))
        second_channels = get_topk_index(out, int(rate * self.out_channels), topmode)
        self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels, torch.arange(3), torch.arange(3))
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels
        out = self.n2(self.scaler(self.conv2(out)))
        third_channels = get_topk_index(out, int(rate * self.out_channels), topmode)
        self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels, torch.arange(3), torch.arange(3))
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        if len(self.shortcut) != 0:
            self.idx[param_name + '.shortcut.0.weight'] = (third_channels, first_channels, torch.arange(1), torch.arange(1))
            self.idx[param_name + '.shortcut.2.weight'], self.idx[param_name + '.shortcut.2.bias'] = third_channels, third_channels
        return third_channels

    def get_idx_hetero(self, first_channels, rate, param_name):
        second_channels = torch.arange(int(rate * self.out_channels))
        self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels, torch.arange(3), torch.arange(3))
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels
        third_channels = torch.arange(int(rate * self.out_channels))
        self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels, torch.arange(3), torch.arange(3))
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        if len(self.shortcut) != 0:
            self.idx[param_name + '.shortcut.0.weight'] = (third_channels, first_channels, torch.arange(1), torch.arange(1))
            self.idx[param_name + '.shortcut.2.weight'], self.idx[param_name + '.shortcut.2.bias'] = third_channels, third_channels
        return third_channels

    def get_idx_weight(self, first_channels, rate, param_name, weight_mode='l2'):
        """
        基于权重大小选择神经元索引
        Args:
            first_channels: 输入通道索引
            rate: 保留比例
            param_name: 参数名称前缀
            weight_mode: 权重选择模式 ('l2', 'l1', 'max')
        """
        # 基于conv1权重选择第二层通道
        conv1_weight = self.conv1.weight.data  # shape: [out_channels, in_channels, 3, 3]
        if weight_mode == 'l2':
            # 计算每个输出通道的L2范数
            weight_importance = torch.norm(conv1_weight, p=2, dim=(1, 2, 3))
        elif weight_mode == 'l1':
            # 计算每个输出通道的L1范数
            weight_importance = torch.norm(conv1_weight, p=1, dim=(1, 2, 3))
        elif weight_mode == 'max':
            # 计算每个输出通道的最大绝对值
            weight_importance = torch.max(torch.abs(conv1_weight).view(conv1_weight.size(0), -1), dim=1)[0]
        else:
            raise ValueError(f'Unsupported weight_mode: {weight_mode}')
        
        # 选择top-k重要的通道，并按原始位置排序
        k = int(rate * self.out_channels)
        second_channels = torch.topk(weight_importance, k)[1]
        second_channels = torch.sort(second_channels)[0]  # 按原始位置排序
        
        self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels, torch.arange(3), torch.arange(3))
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels
        
        # 基于conv2权重选择第三层通道
        conv2_weight = self.conv2.weight.data  # shape: [out_channels, in_channels, 3, 3]
        if weight_mode == 'l2':
            weight_importance = torch.norm(conv2_weight, p=2, dim=(1, 2, 3))
        elif weight_mode == 'l1':
            weight_importance = torch.norm(conv2_weight, p=1, dim=(1, 2, 3))
        elif weight_mode == 'max':
            weight_importance = torch.max(torch.abs(conv2_weight).view(conv2_weight.size(0), -1), dim=1)[0]
        
        third_channels = torch.topk(weight_importance, k)[1]
        third_channels = torch.sort(third_channels)[0]  # 按原始位置排序
        self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels, torch.arange(3), torch.arange(3))
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        
        # 处理shortcut连接
        if len(self.shortcut) != 0:
            self.idx[param_name + '.shortcut.0.weight'] = (third_channels, first_channels, torch.arange(1), torch.arange(1))
            self.idx[param_name + '.shortcut.2.weight'], self.idx[param_name + '.shortcut.2.bias'] = third_channels, third_channels
        
        return third_channels

    def clear_idx(self):
        self.idx.clear()

class Model(FModule):
    def __init__(self,hidden_size = [64, 128, 256, 512], block=BasicBlock, num_block=[2,2,2,2], num_classes=10, rate=1, track=False):
        super().__init__()
        self.idx = OrderedDict()
        self.rate = rate
        self.scaler = Scaler(rate)
        self.in_channels = hidden_size[0]
        self.track = track
        self.roll = 0  # For roll mode

        self.conv = nn.Conv2d(3, hidden_size[0], kernel_size=3, padding=1, bias=False)
        self.n1 = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)

        self.layer1 = self._make_layer(block, hidden_size[0], num_block[0], 1, rate, track)
        self.layer2 = self._make_layer(block, hidden_size[1], num_block[1], 2, rate, track)
        self.layer3 = self._make_layer(block, hidden_size[2], num_block[2], 2, rate, track)
        self.layer4 = self._make_layer(block, hidden_size[3], num_block[3], 2, rate, track)
        
        self.linear = nn.Linear(hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, rate, track):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, rate, track))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.n1(self.scaler(self.conv(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_idx_aware(self, input, rate, topmode):
        start_channels = (torch.arange(3))
        out = F.relu(self.n1(self.scaler(self.conv(input))))
        first_channels = get_topk_index(out, int(rate * self.conv.weight.shape[0]), topmode)
        self.idx['conv.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                first_channels = blc.get_idx_aware(out, first_channels, rate, param_sub_name, topmode)
                self.idx.update(blc.idx)
                out = blc(out)
        
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_hetero(self, rate):
        start_channels = (torch.arange(3))
        first_channels = torch.arange(int(rate * self.conv.weight.shape[0]))
        self.idx['conv.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                first_channels = blc.get_idx_hetero(first_channels, rate, param_sub_name)
                self.idx.update(blc.idx)
        
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_weight(self, rate, weight_mode='l2'):
        """
        基于权重大小选择神经元索引
        Args:
            rate: 保留比例
            weight_mode: 权重选择模式 ('l2', 'l1', 'max')
        """
        start_channels = (torch.arange(3))
        
        # 基于第一个卷积层权重选择通道
        conv_weight = self.conv.weight.data  # shape: [out_channels, in_channels, 3, 3]
        if weight_mode == 'l2':
            weight_importance = torch.norm(conv_weight, p=2, dim=(1, 2, 3))
        elif weight_mode == 'l1':
            weight_importance = torch.norm(conv_weight, p=1, dim=(1, 2, 3))
        elif weight_mode == 'max':
            weight_importance = torch.max(torch.abs(conv_weight).view(conv_weight.size(0), -1), dim=1)[0]
        else:
            raise ValueError(f'Unsupported weight_mode: {weight_mode}')
        
        k = int(rate * self.conv.weight.shape[0])
        first_channels = torch.topk(weight_importance, k)[1]
        first_channels = torch.sort(first_channels)[0]  # 按原始位置排序
        
        self.idx['conv.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        
        # 处理各个layer
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                first_channels = blc.get_idx_weight(first_channels, rate, param_sub_name, weight_mode)
                self.idx.update(blc.idx)
        
        # 处理最后的线性层
        linear_weight = self.linear.weight.data  # shape: [num_classes, in_features]
        if weight_mode == 'l2':
            weight_importance = torch.norm(linear_weight, p=2, dim=0)
        elif weight_mode == 'l1':
            weight_importance = torch.norm(linear_weight, p=1, dim=0)
        elif weight_mode == 'max':
            weight_importance = torch.max(torch.abs(linear_weight), dim=0)[0]
        
        # 对于线性层，我们保留所有输出类别，但可以根据权重重要性选择输入特征
        # 这里我们保持与first_channels一致
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def clear_idx(self):
        self.idx.clear()
        for i, blc in enumerate(self.layer1):
            blc.clear_idx()
        for i, blc in enumerate(self.layer2):
            blc.clear_idx()
        for i, blc in enumerate(self.layer3):
            blc.clear_idx()
        for i, blc in enumerate(self.layer4):
            blc.clear_idx()

    def get_attar(self, attar_name):
        names = attar_name.split('.')
        obj = self
        for x in names:
            obj = getattr(obj, x)
        return obj

# 辅助函数

def resnet18(hidden_size, num_blocks, num_classes, track=False, model_rate=1):
    hidden_size = [int(item * model_rate) for item in hidden_size]
    model = Model(hidden_size, BasicBlock, num_blocks, num_classes, model_rate, track)
    model.apply(init_param)
    return model

def get_topk_index_by_weight(weight_tensor, k, weight_mode='l2'):
    """
    基于权重大小选择top-k索引
    Args:
        weight_tensor: 权重张量
        k: 选择的数量
        weight_mode: 权重计算模式 ('l2', 'l1', 'max')
    Returns:
        top-k索引（按原始位置排序）
    """
    if weight_mode == 'l2':
        if weight_tensor.dim() > 2:  # conv层权重
            importance = torch.norm(weight_tensor, p=2, dim=tuple(range(1, weight_tensor.dim())))
        else:  # linear层权重
            importance = torch.norm(weight_tensor, p=2, dim=1)
    elif weight_mode == 'l1':
        if weight_tensor.dim() > 2:
            importance = torch.norm(weight_tensor, p=1, dim=tuple(range(1, weight_tensor.dim())))
        else:
            importance = torch.norm(weight_tensor, p=1, dim=1)
    elif weight_mode == 'max':
        if weight_tensor.dim() > 2:
            importance = torch.max(torch.abs(weight_tensor).view(weight_tensor.size(0), -1), dim=1)[0]
        else:
            importance = torch.max(torch.abs(weight_tensor), dim=1)[0]
    else:
        raise ValueError(f'Unsupported weight_mode: {weight_mode}')
    
    # 选择top-k索引并按原始位置排序
    topk_indices = torch.topk(importance, k)[1]
    return torch.sort(topk_indices)[0]

def get_topk_index(x, k, topmode):
    if topmode == 'absmax':
        if x.dim() > 2:  # conv
            temp = torch.sum(x, dim=(0, 2, 3))
            return torch.topk(temp, k)[1]
        else:  # linear
            return torch.topk(x, k)[1]
    elif topmode == 'probs':
        if x.dim() > 2:
            temp = torch.sum(x, dim=(0, 2, 3))
            probs = torch.abs(temp) / torch.sum(torch.abs(temp))
        else:
            probs = torch.abs(x) / torch.sum(torch.abs(x))
        samples = torch.multinomial(probs, num_samples=k, replacement=False)
        return samples
    elif topmode == 'absmin':
        if x.dim() > 2:
            temp = -torch.sum(x, dim=(0, 2, 3))
            return torch.topk(temp, k)[1]
        else:
            return torch.topk(-x, k)[1]
    else:
        raise ValueError('no method!')

def get_model_params(global_model, client_model_idx):
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
            raise NameError('Can\'t match {}'.format(k))
    return client_model_params

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)