"""CNN model with GradFL submodel construction support"""
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

class Model(FModule):
    def __init__(self, num_classes=10, rate=1, track=False):
        super().__init__()
        self.idx = OrderedDict()
        self.rate = rate
        self.scaler = Scaler(rate)
        self.track = track
        self.roll = 0  # For roll mode

        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32, momentum=None, track_running_stats=track)
        self.pool1 = nn.MaxPool2d(2)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64, momentum=None, track_running_stats=track)
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.scaler(self.conv1(x)))))
        x = self.pool2(F.relu(self.bn2(self.scaler(self.conv2(x)))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_idx_aware(self, input, rate, topmode):
        start_channels = (torch.arange(3))
        out = self.pool1(F.relu(self.bn1(self.scaler(self.conv1(input)))))
        first_channels = get_topk_index(out, int(rate * 32), topmode)
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(5), torch.arange(5))
        self.idx['conv1.bias'] = first_channels
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        
        out = self.pool2(F.relu(self.bn2(self.scaler(self.conv2(out)))))
        second_channels = get_topk_index(out, int(rate * 64), topmode)
        self.idx['conv2.weight'] = (second_channels, first_channels, torch.arange(5), torch.arange(5))
        self.idx['conv2.bias'] = second_channels
        self.idx['bn2.weight'], self.idx['bn2.bias'] = second_channels, second_channels
        
        out = out.view(-1, 64 * 8 * 8)
        out = F.relu(self.fc1(out))
        fc1_out_channels = torch.arange(int(rate * 512))
        self.idx['fc1.weight'] = (fc1_out_channels, torch.cat([second_channels.repeat(8 * 8)]))
        self.idx['fc1.bias'] = fc1_out_channels
        
        out = F.relu(self.fc2(out))
        fc2_out_channels = torch.arange(int(rate * 128))
        self.idx['fc2.weight'] = (fc2_out_channels, fc1_out_channels)
        self.idx['fc2.bias'] = fc2_out_channels
        
        self.idx['fc3.weight'] = (torch.arange(self.fc3.weight.shape[0]), fc2_out_channels)
        self.idx['fc3.bias'] = torch.arange(self.fc3.weight.shape[0])

    def get_idx_aware_grad(self, rate, topmode, gradient):
        start_channels = (torch.arange(3))
        gradient_conv1 = gradient['conv1']
        first_channels = get_topk_index(gradient_conv1, int(rate * 32), topmode)
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(5), torch.arange(5))
        self.idx['conv1.bias'] = first_channels
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        
        gradient_conv2 = gradient['conv2']
        second_channels = get_topk_index(gradient_conv2, int(rate * 64), topmode)
        self.idx['conv2.weight'] = (second_channels, first_channels, torch.arange(5), torch.arange(5))
        self.idx['conv2.bias'] = second_channels
        self.idx['bn2.weight'], self.idx['bn2.bias'] = second_channels, second_channels
        
        fc1_out_channels = torch.arange(int(rate * 512))
        self.idx['fc1.weight'] = (fc1_out_channels, torch.cat([second_channels.repeat(8 * 8)]))
        self.idx['fc1.bias'] = fc1_out_channels
        
        fc2_out_channels = torch.arange(int(rate * 128))
        self.idx['fc2.weight'] = (fc2_out_channels, fc1_out_channels)
        self.idx['fc2.bias'] = fc2_out_channels
        
        self.idx['fc3.weight'] = (torch.arange(self.fc3.weight.shape[0]), fc2_out_channels)
        self.idx['fc3.bias'] = torch.arange(self.fc3.weight.shape[0])

    def get_idx_roll(self, rate):
        start_channels = (torch.arange(3))
        first_channels = torch.roll(torch.arange(32), shifts=self.roll % 32, dims=-1)[:int(rate * 32)]
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(5), torch.arange(5))
        self.idx['conv1.bias'] = first_channels
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        
        second_channels = torch.roll(torch.arange(64), shifts=self.roll % 64, dims=-1)[:int(rate * 64)]
        self.idx['conv2.weight'] = (second_channels, first_channels, torch.arange(5), torch.arange(5))
        self.idx['conv2.bias'] = second_channels
        self.idx['bn2.weight'], self.idx['bn2.bias'] = second_channels, second_channels
        
        fc1_out_channels = torch.roll(torch.arange(512), shifts=self.roll % 512, dims=-1)[:int(rate * 512)]
        self.idx['fc1.weight'] = (fc1_out_channels, torch.cat([second_channels.repeat(8 * 8)]))
        self.idx['fc1.bias'] = fc1_out_channels
        
        fc2_out_channels = torch.roll(torch.arange(128), shifts=self.roll % 128, dims=-1)[:int(rate * 128)]
        self.idx['fc2.weight'] = (fc2_out_channels, fc1_out_channels)
        self.idx['fc2.bias'] = fc2_out_channels
        
        self.idx['fc3.weight'] = (torch.arange(self.fc3.weight.shape[0]), fc2_out_channels)
        self.idx['fc3.bias'] = torch.arange(self.fc3.weight.shape[0])

    def get_idx_rand(self, rate):
        start_channels = (torch.arange(3))
        first_channels = torch.randperm(32)[:int(rate * 32)]
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(5), torch.arange(5))
        self.idx['conv1.bias'] = first_channels
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        
        second_channels = torch.randperm(64)[:int(rate * 64)]
        self.idx['conv2.weight'] = (second_channels, first_channels, torch.arange(5), torch.arange(5))
        self.idx['conv2.bias'] = second_channels
        self.idx['bn2.weight'], self.idx['bn2.bias'] = second_channels, second_channels
        
        fc1_out_channels = torch.randperm(512)[:int(rate * 512)]
        self.idx['fc1.weight'] = (fc1_out_channels, torch.cat([second_channels.repeat(8 * 8)]))
        self.idx['fc1.bias'] = fc1_out_channels
        
        fc2_out_channels = torch.randperm(128)[:int(rate * 128)]
        self.idx['fc2.weight'] = (fc2_out_channels, fc1_out_channels)
        self.idx['fc2.bias'] = fc2_out_channels
        
        self.idx['fc3.weight'] = (torch.arange(self.fc3.weight.shape[0]), fc2_out_channels)
        self.idx['fc3.bias'] = torch.arange(self.fc3.weight.shape[0])

    def get_idx_hetero(self, rate):
        start_channels = (torch.arange(3))
        first_channels = torch.arange(int(rate * 32))
        self.idx['conv1.weight'] = (first_channels, start_channels, torch.arange(5), torch.arange(5))
        self.idx['conv1.bias'] = first_channels
        self.idx['bn1.weight'], self.idx['bn1.bias'] = first_channels, first_channels
        
        second_channels = torch.arange(int(rate * 64))
        self.idx['conv2.weight'] = (second_channels, first_channels, torch.arange(5), torch.arange(5))
        self.idx['conv2.bias'] = second_channels
        self.idx['bn2.weight'], self.idx['bn2.bias'] = second_channels, second_channels
        
        fc1_out_channels = torch.arange(int(rate * 512))
        self.idx['fc1.weight'] = (fc1_out_channels, torch.cat([second_channels.repeat(8 * 8)]))
        self.idx['fc1.bias'] = fc1_out_channels
        
        fc2_out_channels = torch.arange(int(rate * 128))
        self.idx['fc2.weight'] = (fc2_out_channels, fc1_out_channels)
        self.idx['fc2.bias'] = fc2_out_channels
        
        self.idx['fc3.weight'] = (torch.arange(self.fc3.weight.shape[0]), fc2_out_channels)
        self.idx['fc3.bias'] = torch.arange(self.fc3.weight.shape[0])

    def clear_idx(self):
        self.idx.clear()

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