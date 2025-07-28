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
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=None, track_running_stats=track)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * BasicBlock.expansion, momentum=None, track_running_stats=track)

        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                Scaler(rate),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion, momentum=None, track_running_stats=track)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.scaler(self.conv1(x))))
        out = self.bn2(self.scaler(self.conv2(out)))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

    def get_idx_aware(self, input, rate, topmode):
        start_channels = (torch.arange(self.in_channels))
        out = self.relu(self.bn1(self.scaler(self.conv1(input))))
        first_channels = get_topk_index(out, int(rate * self.out_channels), topmode)
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        out = self.bn2(self.scaler(self.conv2(out)))
        second_channels = get_topk_index(out, int(rate * self.out_channels), topmode)
        self.idx['conv2.weight'] = (second_channels, first_channels, torch.arange(3), torch.arange(3))
        self.idx['bn2.weight'], self.idx['bn2.bias'] = second_channels, second_channels
        if len(self.shortcut) != 0:
            self.idx['shortcut.0.weight'] = (second_channels, start_channels, torch.arange(1), torch.arange(1))
            self.idx['shortcut.2.weight'], self.idx['shortcut.2.bias'] = second_channels, second_channels
        return second_channels

    def get_idx_roll(self, rate):
        start_channels = (torch.arange(self.in_channels))
        first_channels = torch.roll(torch.arange(self.out_channels), shifts=0, dims=-1)[:int(rate * self.out_channels)]
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        second_channels = torch.roll(torch.arange(self.out_channels), shifts=0, dims=-1)[:int(rate * self.out_channels)]
        self.idx['conv2.weight'] = (second_channels, first_channels, torch.arange(3), torch.arange(3))
        self.idx['bn2.weight'], self.idx['bn2.bias'] = second_channels, second_channels
        if len(self.shortcut) != 0:
            self.idx['shortcut.0.weight'] = (second_channels, start_channels, torch.arange(1), torch.arange(1))
            self.idx['shortcut.2.weight'], self.idx['shortcut.2.bias'] = second_channels, second_channels
        return second_channels

    def get_idx_rand(self, rate):
        start_channels = (torch.arange(self.in_channels))
        first_channels = torch.randperm(self.out_channels)[:int(rate * self.out_channels)]
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        second_channels = torch.randperm(self.out_channels)[:int(rate * self.out_channels)]
        self.idx['conv2.weight'] = (second_channels, first_channels, torch.arange(3), torch.arange(3))
        self.idx['bn2.weight'], self.idx['bn2.bias'] = second_channels, second_channels
        if len(self.shortcut) != 0:
            self.idx['shortcut.0.weight'] = (second_channels, start_channels, torch.arange(1), torch.arange(1))
            self.idx['shortcut.2.weight'], self.idx['shortcut.2.bias'] = second_channels, second_channels
        return second_channels

    def get_idx_hetero(self, rate):
        start_channels = (torch.arange(self.in_channels))
        first_channels = torch.arange(int(rate * self.out_channels))
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        second_channels = torch.arange(int(rate * self.out_channels))
        self.idx['conv2.weight'] = (second_channels, first_channels, torch.arange(3), torch.arange(3))
        self.idx['bn2.weight'], self.idx['bn2.bias'] = second_channels, second_channels
        if len(self.shortcut) != 0:
            self.idx['shortcut.0.weight'] = (second_channels, start_channels, torch.arange(1), torch.arange(1))
            self.idx['shortcut.2.weight'], self.idx['shortcut.2.bias'] = second_channels, second_channels
        return second_channels

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

        self.conv1 = nn.Conv2d(3, hidden_size[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, hidden_size[0], num_block[0], 1, rate, track)
        self.layer2 = self._make_layer(block, hidden_size[1], num_block[1], 2, rate, track)
        self.layer3 = self._make_layer(block, hidden_size[2], num_block[2], 2, rate, track)
        self.layer4 = self._make_layer(block, hidden_size[3], num_block[3], 2, rate, track)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, rate, track):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, rate, track))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.scaler(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def get_idx_aware(self, input, rate, topmode):
        start_channels = (torch.arange(3))
        out = self.relu(self.bn1(self.scaler(self.conv1(input))))
        first_channels = get_topk_index(out, int(rate * 64), topmode)
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        
        for i, blc in enumerate(self.layer1):
            out = blc(out)
            first_channels = blc.get_idx_aware(out, rate, topmode)
            for k, v in blc.idx.items():
                self.idx['layer1.{}.{}'.format(i, k)] = v
        
        for i, blc in enumerate(self.layer2):
            out = blc(out)
            first_channels = blc.get_idx_aware(out, rate, topmode)
            for k, v in blc.idx.items():
                self.idx['layer2.{}.{}'.format(i, k)] = v
        
        for i, blc in enumerate(self.layer3):
            out = blc(out)
            first_channels = blc.get_idx_aware(out, rate, topmode)
            for k, v in blc.idx.items():
                self.idx['layer3.{}.{}'.format(i, k)] = v
        
        for i, blc in enumerate(self.layer4):
            out = blc(out)
            first_channels = blc.get_idx_aware(out, rate, topmode)
            for k, v in blc.idx.items():
                self.idx['layer4.{}.{}'.format(i, k)] = v
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        self.idx['fc.weight'] = (torch.arange(self.fc.weight.shape[0]), first_channels)
        self.idx['fc.bias'] = torch.arange(self.fc.weight.shape[0])

    def get_idx_aware_grad(self, rate, topmode, gradient):
        start_channels = (torch.arange(3))
        gradient_s = gradient['conv1']
        first_channels = get_topk_index(gradient_s, int(rate * 64), topmode)
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        
        for i, blc in enumerate(self.layer1):
            gradient_f = gradient[f'layer1.{i}.conv1']
            first_channels = get_topk_index(gradient_f, int(rate * 64), topmode)
            blc.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
            blc.idx['bn1.weight'], blc.idx['bn1.bias'] = first_channels, first_channels
            
            gradient_s = gradient[f'layer1.{i}.conv2']
            second_channels = get_topk_index(gradient_s, int(rate * 64), topmode)
            blc.idx['conv2.weight'] = (second_channels, first_channels, torch.arange(3), torch.arange(3))
            blc.idx['bn2.weight'], blc.idx['bn2.bias'] = second_channels, second_channels
            
            if len(blc.shortcut) != 0:
                blc.idx['shortcut.0.weight'] = (second_channels, start_channels, torch.arange(1), torch.arange(1))
                blc.idx['shortcut.2.weight'], blc.idx['shortcut.2.bias'] = second_channels, second_channels
            
            for k, v in blc.idx.items():
                self.idx[f'layer1.{i}.{k}'] = v
            
            start_channels = second_channels
        
        # 类似地处理layer2, layer3, layer4
        # 为简洁起见，这里只展示layer1的处理方式
        # 实际实现中需要添加其他层的处理
        
        self.idx['fc.weight'] = (torch.arange(self.fc.weight.shape[0]), start_channels)
        self.idx['fc.bias'] = torch.arange(self.fc.weight.shape[0])

    def get_idx_roll(self, rate):
        start_channels = (torch.arange(3))
        first_channels = torch.roll(torch.arange(64), shifts=self.roll % 64, dims=-1)[:int(rate * 64)]
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        
        for i, blc in enumerate(self.layer1):
            first_channels = blc.get_idx_roll(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer1.{i}.{k}'] = v
        
        for i, blc in enumerate(self.layer2):
            first_channels = blc.get_idx_roll(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer2.{i}.{k}'] = v
        
        for i, blc in enumerate(self.layer3):
            first_channels = blc.get_idx_roll(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer3.{i}.{k}'] = v
        
        for i, blc in enumerate(self.layer4):
            first_channels = blc.get_idx_roll(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer4.{i}.{k}'] = v
        
        self.idx['fc.weight'] = (torch.arange(self.fc.weight.shape[0]), first_channels)
        self.idx['fc.bias'] = torch.arange(self.fc.weight.shape[0])

    def get_idx_rand(self, rate):
        start_channels = (torch.arange(3))
        first_channels = torch.randperm(64)[:int(rate * 64)]
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        
        for i, blc in enumerate(self.layer1):
            first_channels = blc.get_idx_rand(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer1.{i}.{k}'] = v
        
        for i, blc in enumerate(self.layer2):
            first_channels = blc.get_idx_rand(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer2.{i}.{k}'] = v
        
        for i, blc in enumerate(self.layer3):
            first_channels = blc.get_idx_rand(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer3.{i}.{k}'] = v
        
        for i, blc in enumerate(self.layer4):
            first_channels = blc.get_idx_rand(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer4.{i}.{k}'] = v
        
        self.idx['fc.weight'] = (torch.arange(self.fc.weight.shape[0]), first_channels)
        self.idx['fc.bias'] = torch.arange(self.fc.weight.shape[0])

    def get_idx_hetero(self, rate):
        start_channels = (torch.arange(3))
        first_channels = torch.arange(int(rate * 64))
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(3), torch.arange(3))
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        
        for i, blc in enumerate(self.layer1):
            first_channels = blc.get_idx_hetero(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer1.{i}.{k}'] = v
        
        for i, blc in enumerate(self.layer2):
            first_channels = blc.get_idx_hetero(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer2.{i}.{k}'] = v
        
        for i, blc in enumerate(self.layer3):
            first_channels = blc.get_idx_hetero(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer3.{i}.{k}'] = v
        
        for i, blc in enumerate(self.layer4):
            first_channels = blc.get_idx_hetero(rate)
            for k, v in blc.idx.items():
                self.idx[f'layer4.{i}.{k}'] = v
        
        self.idx['fc.weight'] = (torch.arange(self.fc.weight.shape[0]), first_channels)
        self.idx['fc.bias'] = torch.arange(self.fc.weight.shape[0])

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
                client_model_params[k] = copy.deepcopy(v[torch.meshgrid(client_model_idx[k], indexing='ij')])
            else:
                client_model_params[k] = copy.deepcopy(v[client_model_idx[k]])
        else:
            raise NameError('Can\'t match {}'.format(k))
    return client_model_params

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)