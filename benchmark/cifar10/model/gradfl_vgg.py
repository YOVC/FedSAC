"""VGG model with GradFL submodel construction support"""
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

class VGG(FModule):
    def __init__(self, features, num_classes=10, rate=1, track=False):
        super().__init__()
        self.idx = OrderedDict()
        self.rate = rate
        self.scaler = Scaler(rate)
        self.track = track
        self.roll = 0  # For roll mode
        
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        # 存储特征层的卷积和BN层，用于子模型构建
        self.conv_layers = []
        self.bn_layers = []
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                self.conv_layers.append(module)
            elif isinstance(module, nn.BatchNorm2d):
                self.bn_layers.append(module)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_idx_aware(self, input, rate, topmode):
        # 初始通道为输入图像的通道数
        start_channels = torch.arange(3)
        out = input
        
        # 处理特征提取层
        layer_idx = 0
        for i, module in enumerate(self.features):
            if isinstance(module, nn.Conv2d):
                out = module(out)
                if isinstance(self.features[i+1], nn.BatchNorm2d):
                    out = self.features[i+1](out)
                out = F.relu(out)
                if isinstance(self.features[i+2], nn.MaxPool2d):
                    out = self.features[i+2](out)
                
                # 获取当前层的输出通道数
                out_channels = module.out_channels
                # 选择重要通道
                selected_channels = get_topk_index(out, int(rate * out_channels), topmode)
                
                # 设置卷积层的索引
                self.idx[f'features.{i}.weight'] = (selected_channels, start_channels, torch.arange(module.kernel_size[0]), torch.arange(module.kernel_size[1]))
                if module.bias is not None:
                    self.idx[f'features.{i}.bias'] = selected_channels
                
                # 如果下一层是BN层，设置BN层的索引
                if isinstance(self.features[i+1], nn.BatchNorm2d):
                    self.idx[f'features.{i+1}.weight'] = selected_channels
                    self.idx[f'features.{i+1}.bias'] = selected_channels
                
                # 更新起始通道为当前选择的通道
                start_channels = selected_channels
                layer_idx += 1
        
        # 处理分类器层
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # 处理第一个全连接层
        fc1_out_channels = torch.arange(int(rate * 4096))
        self.idx['classifier.0.weight'] = (fc1_out_channels, torch.cat([start_channels.repeat(7 * 7)]))
        self.idx['classifier.0.bias'] = fc1_out_channels
        
        # 处理第二个全连接层
        fc2_out_channels = torch.arange(int(rate * 4096))
        self.idx['classifier.3.weight'] = (fc2_out_channels, fc1_out_channels)
        self.idx['classifier.3.bias'] = fc2_out_channels
        
        # 处理最后一个全连接层
        self.idx['classifier.6.weight'] = (torch.arange(self.classifier[6].weight.shape[0]), fc2_out_channels)
        self.idx['classifier.6.bias'] = torch.arange(self.classifier[6].weight.shape[0])

    def get_idx_aware_grad(self, rate, topmode, gradient):
        # 初始通道为输入图像的通道数
        start_channels = torch.arange(3)
        
        # 处理特征提取层
        layer_idx = 0
        for i, module in enumerate(self.features):
            if isinstance(module, nn.Conv2d):
                # 获取当前层的梯度
                gradient_key = f'features.{i}'
                if gradient_key in gradient:
                    gradient_layer = gradient[gradient_key]
                    # 获取当前层的输出通道数
                    out_channels = module.out_channels
                    # 基于梯度选择重要通道
                    selected_channels = get_topk_index(gradient_layer, int(rate * out_channels), topmode)
                    
                    # 设置卷积层的索引
                    self.idx[f'features.{i}.weight'] = (selected_channels, start_channels, torch.arange(module.kernel_size[0]), torch.arange(module.kernel_size[1]))
                    if module.bias is not None:
                        self.idx[f'features.{i}.bias'] = selected_channels
                    
                    # 如果下一层是BN层，设置BN层的索引
                    if isinstance(self.features[i+1], nn.BatchNorm2d):
                        self.idx[f'features.{i+1}.weight'] = selected_channels
                        self.idx[f'features.{i+1}.bias'] = selected_channels
                    
                    # 更新起始通道为当前选择的通道
                    start_channels = selected_channels
                    layer_idx += 1
        
        # 处理分类器层
        # 处理第一个全连接层
        fc1_out_channels = torch.arange(int(rate * 4096))
        self.idx['classifier.0.weight'] = (fc1_out_channels, torch.cat([start_channels.repeat(7 * 7)]))
        self.idx['classifier.0.bias'] = fc1_out_channels
        
        # 处理第二个全连接层
        fc2_out_channels = torch.arange(int(rate * 4096))
        self.idx['classifier.3.weight'] = (fc2_out_channels, fc1_out_channels)
        self.idx['classifier.3.bias'] = fc2_out_channels
        
        # 处理最后一个全连接层
        self.idx['classifier.6.weight'] = (torch.arange(self.classifier[6].weight.shape[0]), fc2_out_channels)
        self.idx['classifier.6.bias'] = torch.arange(self.classifier[6].weight.shape[0])

    def get_idx_roll(self, rate):
        # 初始通道为输入图像的通道数
        start_channels = torch.arange(3)
        
        # 处理特征提取层
        for i, module in enumerate(self.features):
            if isinstance(module, nn.Conv2d):
                # 获取当前层的输出通道数
                out_channels = module.out_channels
                # 使用roll方法选择通道
                selected_channels = torch.roll(torch.arange(out_channels), shifts=self.roll % out_channels, dims=-1)[:int(rate * out_channels)]
                
                # 设置卷积层的索引
                self.idx[f'features.{i}.weight'] = (selected_channels, start_channels, torch.arange(module.kernel_size[0]), torch.arange(module.kernel_size[1]))
                if module.bias is not None:
                    self.idx[f'features.{i}.bias'] = selected_channels
                
                # 如果下一层是BN层，设置BN层的索引
                if i+1 < len(self.features) and isinstance(self.features[i+1], nn.BatchNorm2d):
                    self.idx[f'features.{i+1}.weight'] = selected_channels
                    self.idx[f'features.{i+1}.bias'] = selected_channels
                
                # 更新起始通道为当前选择的通道
                start_channels = selected_channels
        
        # 处理分类器层
        # 处理第一个全连接层
        fc1_out_channels = torch.roll(torch.arange(4096), shifts=self.roll % 4096, dims=-1)[:int(rate * 4096)]
        self.idx['classifier.0.weight'] = (fc1_out_channels, torch.cat([start_channels.repeat(7 * 7)]))
        self.idx['classifier.0.bias'] = fc1_out_channels
        
        # 处理第二个全连接层
        fc2_out_channels = torch.roll(torch.arange(4096), shifts=self.roll % 4096, dims=-1)[:int(rate * 4096)]
        self.idx['classifier.3.weight'] = (fc2_out_channels, fc1_out_channels)
        self.idx['classifier.3.bias'] = fc2_out_channels
        
        # 处理最后一个全连接层
        self.idx['classifier.6.weight'] = (torch.arange(self.classifier[6].weight.shape[0]), fc2_out_channels)
        self.idx['classifier.6.bias'] = torch.arange(self.classifier[6].weight.shape[0])

    def get_idx_rand(self, rate):
        # 初始通道为输入图像的通道数
        start_channels = torch.arange(3)
        
        # 处理特征提取层
        for i, module in enumerate(self.features):
            if isinstance(module, nn.Conv2d):
                # 获取当前层的输出通道数
                out_channels = module.out_channels
                # 随机选择通道
                selected_channels = torch.randperm(out_channels)[:int(rate * out_channels)]
                
                # 设置卷积层的索引
                self.idx[f'features.{i}.weight'] = (selected_channels, start_channels, torch.arange(module.kernel_size[0]), torch.arange(module.kernel_size[1]))
                if module.bias is not None:
                    self.idx[f'features.{i}.bias'] = selected_channels
                
                # 如果下一层是BN层，设置BN层的索引
                if i+1 < len(self.features) and isinstance(self.features[i+1], nn.BatchNorm2d):
                    self.idx[f'features.{i+1}.weight'] = selected_channels
                    self.idx[f'features.{i+1}.bias'] = selected_channels
                
                # 更新起始通道为当前选择的通道
                start_channels = selected_channels
        
        # 处理分类器层
        # 处理第一个全连接层
        fc1_out_channels = torch.randperm(4096)[:int(rate * 4096)]
        self.idx['classifier.0.weight'] = (fc1_out_channels, torch.cat([start_channels.repeat(7 * 7)]))
        self.idx['classifier.0.bias'] = fc1_out_channels
        
        # 处理第二个全连接层
        fc2_out_channels = torch.randperm(4096)[:int(rate * 4096)]
        self.idx['classifier.3.weight'] = (fc2_out_channels, fc1_out_channels)
        self.idx['classifier.3.bias'] = fc2_out_channels
        
        # 处理最后一个全连接层
        self.idx['classifier.6.weight'] = (torch.arange(self.classifier[6].weight.shape[0]), fc2_out_channels)
        self.idx['classifier.6.bias'] = torch.arange(self.classifier[6].weight.shape[0])

    def get_idx_hetero(self, rate):
        # 初始通道为输入图像的通道数
        start_channels = torch.arange(3)
        
        # 处理特征提取层
        for i, module in enumerate(self.features):
            if isinstance(module, nn.Conv2d):
                # 获取当前层的输出通道数
                out_channels = module.out_channels
                # 顺序选择通道
                selected_channels = torch.arange(int(rate * out_channels))
                
                # 设置卷积层的索引
                self.idx[f'features.{i}.weight'] = (selected_channels, start_channels, torch.arange(module.kernel_size[0]), torch.arange(module.kernel_size[1]))
                if module.bias is not None:
                    self.idx[f'features.{i}.bias'] = selected_channels
                
                # 如果下一层是BN层，设置BN层的索引
                if i+1 < len(self.features) and isinstance(self.features[i+1], nn.BatchNorm2d):
                    self.idx[f'features.{i+1}.weight'] = selected_channels
                    self.idx[f'features.{i+1}.bias'] = selected_channels
                
                # 更新起始通道为当前选择的通道
                start_channels = selected_channels
        
        # 处理分类器层
        # 处理第一个全连接层
        fc1_out_channels = torch.arange(int(rate * 4096))
        self.idx['classifier.0.weight'] = (fc1_out_channels, torch.cat([start_channels.repeat(7 * 7)]))
        self.idx['classifier.0.bias'] = fc1_out_channels
        
        # 处理第二个全连接层
        fc2_out_channels = torch.arange(int(rate * 4096))
        self.idx['classifier.3.weight'] = (fc2_out_channels, fc1_out_channels)
        self.idx['classifier.3.bias'] = fc2_out_channels
        
        # 处理最后一个全连接层
        self.idx['classifier.6.weight'] = (torch.arange(self.classifier[6].weight.shape[0]), fc2_out_channels)
        self.idx['classifier.6.bias'] = torch.arange(self.classifier[6].weight.shape[0])

    def clear_idx(self):
        self.idx.clear()

    def get_attar(self, attar_name):
        names = attar_name.split('.')
        obj = self
        for x in names:
            obj = getattr(obj, x)
        return obj

# VGG配置
config = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=True, track=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, momentum=None, track_running_stats=track), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class Model(FModule):
    def __init__(self, num_classes=10, rate=1, track=False):
        super().__init__()
        self.vgg = VGG(make_layers(config['VGG16'], batch_norm=True, track=track), num_classes=num_classes, rate=rate, track=track)
        self.idx = self.vgg.idx

    def forward(self, x):
        return self.vgg(x)

    def get_idx_aware(self, input, rate, topmode):
        self.vgg.get_idx_aware(input, rate, topmode)
        self.idx = self.vgg.idx

    def get_idx_aware_grad(self, rate, topmode, gradient):
        self.vgg.get_idx_aware_grad(rate, topmode, gradient)
        self.idx = self.vgg.idx

    def get_idx_roll(self, rate):
        self.vgg.get_idx_roll(rate)
        self.idx = self.vgg.idx

    def get_idx_rand(self, rate):
        self.vgg.get_idx_rand(rate)
        self.idx = self.vgg.idx

    def get_idx_hetero(self, rate):
        self.vgg.get_idx_hetero(rate)
        self.idx = self.vgg.idx

    def clear_idx(self):
        self.vgg.clear_idx()
        self.idx.clear()

    def get_attar(self, attar_name):
        return self.vgg.get_attar(attar_name)

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