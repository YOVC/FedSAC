import torch
import torch.nn as nn
import copy
from collections import OrderedDict
from utils.fmodule import FModule

if __name__ == '__main__':
    from utils import init_param, Scaler  # When debug use this code
else:
    from .utils import init_param, Scaler  # When run main use it


def conv_layer(chann_in, chann_out, k_size, p_size, rate):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        Scaler(rate),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s, rate):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i], rate) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


class VGG16(FModule):
    def __init__(self, num_classes=10, rate=1):
        super(VGG16, self).__init__()
        self.idx = OrderedDict()
        self.rate = rate
        self.scaler = Scaler(rate)
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2, rate)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2, rate)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2, rate)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2, rate)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2, rate)

        # FC layers
        self.layer6 = nn.Sequential(
            nn.Linear(512, 256),
            Scaler(rate),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(256, 128),
            Scaler(rate),
            nn.ReLU()
        )

        # Final layer
        self.layer8 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        vgg_features = x.view(x.size(0), -1)
        x = self.layer6(vgg_features)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def get_idx_aware(self, input, rate, topmode):
        start_channels = (torch.arange(3))
        out = self.layer1[:2](input)
        first_channels = get_topk_index(out, int(rate * 64), topmode)
        self.idx['layer1.0.0.weight'], self.idx['layer1.0.0.bias'] = (first_channels, start_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        out = self.layer1[2:](out)
        second_channels = get_topk_index(out, int(rate * 64), topmode)
        self.idx['layer1.1.0.weight'], self.idx['layer1.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        out = self.layer2[:2](out)
        first_channels = get_topk_index(out, int(rate * 128), topmode)
        self.idx['layer2.0.0.weight'], self.idx['layer2.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        out = self.layer2[2:](out)
        second_channels = get_topk_index(out, int(rate * 128), topmode)
        self.idx['layer2.1.0.weight'], self.idx['layer2.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        out = self.layer3[:2](out)
        first_channels = get_topk_index(out, int(rate * 256), topmode)
        self.idx['layer3.0.0.weight'], self.idx['layer3.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        out = self.layer3[2:4](out)
        second_channels = get_topk_index(out, int(rate * 256), topmode)
        self.idx['layer3.1.0.weight'], self.idx['layer3.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        out = self.layer3[4:](out)
        third_channels = get_topk_index(out, int(rate * 256), topmode)
        self.idx['layer3.2.0.weight'], self.idx['layer3.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        out = self.layer4[:2](out)
        first_channels = get_topk_index(out, int(rate * 512), topmode)
        self.idx['layer4.0.0.weight'], self.idx['layer4.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        out = self.layer4[2:4](out)
        second_channels = get_topk_index(out, int(rate * 512), topmode)
        self.idx['layer4.1.0.weight'], self.idx['layer4.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        out = self.layer4[4:](out)
        third_channels = get_topk_index(out, int(rate * 512), topmode)
        self.idx['layer4.2.0.weight'], self.idx['layer4.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        out = self.layer5[:2](out)
        first_channels = get_topk_index(out, int(rate * 512), topmode)
        self.idx['layer5.0.0.weight'], self.idx['layer5.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        out = self.layer5[2:4](out)
        second_channels = get_topk_index(out, int(rate * 512), topmode)
        self.idx['layer5.1.0.weight'], self.idx['layer5.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        out = self.layer5[4:](out)
        third_channels = get_topk_index(out, int(rate * 512), topmode)
        self.idx['layer5.2.0.weight'], self.idx['layer5.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        vgg_features = out.view(out.size(0), -1)
        out = self.layer6(vgg_features)
        first_channels = get_topk_index(out, int(rate * 256), topmode)
        self.idx['layer6.0.weight'], self.idx['layer6.0.bias'] = (first_channels, third_channels), first_channels
        out = self.layer7(out)
        second_channels = get_topk_index(out, int(rate * 128), topmode)
        self.idx['layer7.0.weight'], self.idx['layer7.0.bias'] = (second_channels, first_channels), second_channels
        self.idx['layer8.weight'], self.idx['layer8.bias'] = (torch.arange(10), second_channels), torch.arange(10)

    def get_idx_aware_grad(self, rate, topmode):
        start_channels = (torch.arange(3))
        gradient_s = self.get_attar('layer1.0.0.weight.grad')
        first_channels = get_topk_index(gradient_s, int(rate * 64), topmode)
        self.idx['layer1.0.0.weight'], self.idx['layer1.0.0.bias'] = (first_channels, start_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        gradient_s = self.get_attar('layer1.1.0.weight.grad')
        second_channels = get_topk_index(gradient_s, int(rate * 64), topmode)
        self.idx['layer1.1.0.weight'], self.idx['layer1.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        gradient_s = self.get_attar('layer2.0.0.weight.grad')
        first_channels = get_topk_index(gradient_s, int(rate * 128), topmode)
        self.idx['layer2.0.0.weight'], self.idx['layer2.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        gradient_s = self.get_attar('layer2.1.0.weight.grad')
        second_channels = get_topk_index(gradient_s, int(rate * 128), topmode)
        self.idx['layer2.1.0.weight'], self.idx['layer2.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        gradient_s = self.get_attar('layer3.0.0.weight.grad')
        first_channels = get_topk_index(gradient_s, int(rate * 256), topmode)
        self.idx['layer3.0.0.weight'], self.idx['layer3.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        gradient_s = self.get_attar('layer3.1.0.weight.grad')
        second_channels = get_topk_index(gradient_s, int(rate * 256), topmode)
        self.idx['layer3.1.0.weight'], self.idx['layer3.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        gradient_s = self.get_attar('layer3.2.0.weight.grad')
        third_channels = get_topk_index(gradient_s, int(rate * 256), topmode)
        self.idx['layer3.2.0.weight'], self.idx['layer3.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        gradient_s = self.get_attar('layer4.0.0.weight.grad')
        first_channels = get_topk_index(gradient_s, int(rate * 512), topmode)
        self.idx['layer4.0.0.weight'], self.idx['layer4.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        gradient_s = self.get_attar('layer4.1.0.weight.grad')
        second_channels = get_topk_index(gradient_s, int(rate * 512), topmode)
        self.idx['layer4.1.0.weight'], self.idx['layer4.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        gradient_s = self.get_attar('layer4.2.0.weight.grad')
        third_channels = get_topk_index(gradient_s, int(rate * 512), topmode)
        self.idx['layer4.2.0.weight'], self.idx['layer4.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        gradient_s = self.get_attar('layer5.0.0.weight.grad')
        first_channels = get_topk_index(gradient_s, int(rate * 512), topmode)
        self.idx['layer5.0.0.weight'], self.idx['layer5.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        gradient_s = self.get_attar('layer5.1.0.weight.grad')
        second_channels = get_topk_index(gradient_s, int(rate * 512), topmode)
        self.idx['layer5.1.0.weight'], self.idx['layer5.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        gradient_s = self.get_attar('layer5.2.0.weight.grad')
        third_channels = get_topk_index(gradient_s, int(rate * 512), topmode)
        self.idx['layer5.2.0.weight'], self.idx['layer5.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        gradient_s = self.get_attar('layer6.0.weight.grad')
        first_channels = get_topk_index(gradient_s, int(rate * 256), topmode)
        self.idx['layer6.0.weight'], self.idx['layer6.0.bias'] = (first_channels, third_channels), first_channels
        gradient_s = self.get_attar('layer7.0.weight.grad')
        second_channels = get_topk_index(gradient_s, int(rate * 128), topmode)
        self.idx['layer7.0.weight'], self.idx['layer7.0.bias'] = (second_channels, first_channels), second_channels
        self.idx['layer8.weight'], self.idx['layer8.bias'] = (torch.arange(10), second_channels), torch.arange(10)
        
    def get_idx_aware_weight(self, rate, topmode, weights):
        """基于权重大小选择重要通道"""
        start_channels = (torch.arange(3))
        weight_s = weights['layer1.0.0.weight']
        first_channels = get_topk_index(weight_s, int(rate * 64), topmode)
        self.idx['layer1.0.0.weight'], self.idx['layer1.0.0.bias'] = (first_channels, start_channels, torch.arange(3),
                                                                        torch.arange(3)), first_channels
        weight_s = weights['layer1.1.0.weight']
        second_channels = get_topk_index(weight_s, int(rate * 64), topmode)
        self.idx['layer1.1.0.weight'], self.idx['layer1.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                        torch.arange(3)), second_channels
        weight_s = weights['layer2.0.0.weight']
        first_channels = get_topk_index(weight_s, int(rate * 128), topmode)
        self.idx['layer2.0.0.weight'], self.idx['layer2.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                        torch.arange(3)), first_channels
        weight_s = weights['layer2.1.0.weight']
        second_channels = get_topk_index(weight_s, int(rate * 128), topmode)
        self.idx['layer2.1.0.weight'], self.idx['layer2.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                        torch.arange(3)), second_channels
        weight_s = weights['layer3.0.0.weight']
        first_channels = get_topk_index(weight_s, int(rate * 256), topmode)
        self.idx['layer3.0.0.weight'], self.idx['layer3.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                        torch.arange(3)), first_channels
        weight_s = weights['layer3.1.0.weight']
        second_channels = get_topk_index(weight_s, int(rate * 256), topmode)
        self.idx['layer3.1.0.weight'], self.idx['layer3.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                        torch.arange(3)), second_channels
        weight_s = weights['layer3.2.0.weight']
        third_channels = get_topk_index(weight_s, int(rate * 256), topmode)
        self.idx['layer3.2.0.weight'], self.idx['layer3.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                        torch.arange(3)), third_channels
        weight_s = weights['layer4.0.0.weight']
        first_channels = get_topk_index(weight_s, int(rate * 512), topmode)
        self.idx['layer4.0.0.weight'], self.idx['layer4.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                        torch.arange(3)), first_channels
        weight_s = weights['layer4.1.0.weight']
        second_channels = get_topk_index(weight_s, int(rate * 512), topmode)
        self.idx['layer4.1.0.weight'], self.idx['layer4.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                        torch.arange(3)), second_channels
        weight_s = weights['layer4.2.0.weight']
        third_channels = get_topk_index(weight_s, int(rate * 512), topmode)
        self.idx['layer4.2.0.weight'], self.idx['layer4.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                        torch.arange(3)), third_channels
        weight_s = weights['layer5.0.0.weight']
        first_channels = get_topk_index(weight_s, int(rate * 512), topmode)
        self.idx['layer5.0.0.weight'], self.idx['layer5.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                        torch.arange(3)), first_channels
        weight_s = weights['layer5.1.0.weight']
        second_channels = get_topk_index(weight_s, int(rate * 512), topmode)
        self.idx['layer5.1.0.weight'], self.idx['layer5.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                        torch.arange(3)), second_channels
        weight_s = weights['layer5.2.0.weight']
        third_channels = get_topk_index(weight_s, int(rate * 512), topmode)
        self.idx['layer5.2.0.weight'], self.idx['layer5.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                        torch.arange(3)), third_channels
        weight_s = weights['layer6.0.weight']
        first_channels = get_topk_index(weight_s, int(rate * 256), topmode)
        self.idx['layer6.0.weight'], self.idx['layer6.0.bias'] = (first_channels, third_channels), first_channels
        weight_s = weights['layer7.0.weight']
        second_channels = get_topk_index(weight_s, int(rate * 128), topmode)
        self.idx['layer7.0.weight'], self.idx['layer7.0.bias'] = (second_channels, first_channels), second_channels
        self.idx['layer8.weight'], self.idx['layer8.bias'] = (torch.arange(10), second_channels), torch.arange(10)

    def get_idx_roll(self, rate):
        start_channels = (torch.arange(3))
        first_channels = torch.roll(torch.arange(64), shifts=0, dims=-1)[:int(rate * 64)]
        self.idx['layer1.0.0.weight'], self.idx['layer1.0.0.bias'] = (first_channels, start_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.roll(torch.arange(64), shifts=0, dims=-1)[:int(rate * 64)]
        self.idx['layer1.1.0.weight'], self.idx['layer1.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        first_channels = torch.roll(torch.arange(128), shifts=0, dims=-1)[:int(rate * 128)]
        self.idx['layer2.0.0.weight'], self.idx['layer2.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.roll(torch.arange(128), shifts=0, dims=-1)[:int(rate * 128)]
        self.idx['layer2.1.0.weight'], self.idx['layer2.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        first_channels = torch.roll(torch.arange(256), shifts=0, dims=-1)[:int(rate * 256)]
        self.idx['layer3.0.0.weight'], self.idx['layer3.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.roll(torch.arange(256), shifts=0, dims=-1)[:int(rate * 256)]
        self.idx['layer3.1.0.weight'], self.idx['layer3.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        third_channels = torch.roll(torch.arange(256), shifts=0, dims=-1)[:int(rate * 256)]
        self.idx['layer3.2.0.weight'], self.idx['layer3.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        first_channels = torch.roll(torch.arange(512), shifts=0, dims=-1)[:int(rate * 512)]
        self.idx['layer4.0.0.weight'], self.idx['layer4.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.roll(torch.arange(512), shifts=0, dims=-1)[:int(rate * 512)]
        self.idx['layer4.1.0.weight'], self.idx['layer4.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        third_channels = torch.roll(torch.arange(512), shifts=0, dims=-1)[:int(rate * 512)]
        self.idx['layer4.2.0.weight'], self.idx['layer4.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        first_channels = torch.roll(torch.arange(512), shifts=0, dims=-1)[:int(rate * 512)]
        self.idx['layer5.0.0.weight'], self.idx['layer5.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.roll(torch.arange(512), shifts=0, dims=-1)[:int(rate * 512)]
        self.idx['layer5.1.0.weight'], self.idx['layer5.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        third_channels = torch.roll(torch.arange(512), shifts=0, dims=-1)[:int(rate * 512)]
        self.idx['layer5.2.0.weight'], self.idx['layer5.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        first_channels = torch.roll(torch.arange(256), shifts=0, dims=-1)[:int(rate * 256)]
        self.idx['layer6.0.weight'], self.idx['layer6.0.bias'] = (first_channels, third_channels), first_channels
        second_channels = torch.roll(torch.arange(128), shifts=0, dims=-1)[:int(rate * 128)]
        self.idx['layer7.0.weight'], self.idx['layer7.0.bias'] = (second_channels, first_channels), second_channels
        self.idx['layer8.weight'], self.idx['layer8.bias'] = (torch.arange(10), second_channels), torch.arange(10)

    def get_idx_rand(self, rate):
        start_channels = (torch.arange(3))
        first_channels = torch.randperm(64)[:int(rate * 64)]
        self.idx['layer1.0.0.weight'], self.idx['layer1.0.0.bias'] = (first_channels, start_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.randperm(64)[:int(rate * 64)]
        self.idx['layer1.1.0.weight'], self.idx['layer1.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        first_channels = torch.randperm(128)[:int(rate * 128)]
        self.idx['layer2.0.0.weight'], self.idx['layer2.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.randperm(128)[:int(rate * 128)]
        self.idx['layer2.1.0.weight'], self.idx['layer2.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        first_channels = torch.randperm(256)[:int(rate * 256)]
        self.idx['layer3.0.0.weight'], self.idx['layer3.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.randperm(256)[:int(rate * 256)]
        self.idx['layer3.1.0.weight'], self.idx['layer3.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        third_channels = torch.randperm(256)[:int(rate * 256)]
        self.idx['layer3.2.0.weight'], self.idx['layer3.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        first_channels = torch.randperm(512)[:int(rate * 512)]
        self.idx['layer4.0.0.weight'], self.idx['layer4.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.randperm(512)[:int(rate * 512)]
        self.idx['layer4.1.0.weight'], self.idx['layer4.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        third_channels = torch.randperm(512)[:int(rate * 512)]
        self.idx['layer4.2.0.weight'], self.idx['layer4.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        first_channels = torch.randperm(512)[:int(rate * 512)]
        self.idx['layer5.0.0.weight'], self.idx['layer5.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.randperm(512)[:int(rate * 512)]
        self.idx['layer5.1.0.weight'], self.idx['layer5.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        third_channels = torch.randperm(512)[:int(rate * 512)]
        self.idx['layer5.2.0.weight'], self.idx['layer5.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        first_channels = torch.randperm(256)[:int(rate * 256)]
        self.idx['layer6.0.weight'], self.idx['layer6.0.bias'] = (first_channels, third_channels), first_channels
        second_channels = torch.randperm(128)[:int(rate * 128)]
        self.idx['layer7.0.weight'], self.idx['layer7.0.bias'] = (second_channels, first_channels), second_channels
        self.idx['layer8.weight'], self.idx['layer8.bias'] = (torch.arange(10), second_channels), torch.arange(10)

    def get_idx_hetero(self, rate):
        start_channels = (torch.arange(3))
        first_channels = torch.arange(int(rate * 64))
        self.idx['layer1.0.0.weight'], self.idx['layer1.0.0.bias'] = (first_channels, start_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.arange(int(rate * 64))
        self.idx['layer1.1.0.weight'], self.idx['layer1.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        first_channels = torch.arange(int(rate * 128))
        self.idx['layer2.0.0.weight'], self.idx['layer2.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.arange(int(rate * 128))
        self.idx['layer2.1.0.weight'], self.idx['layer2.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        first_channels = torch.arange(int(rate * 256))
        self.idx['layer3.0.0.weight'], self.idx['layer3.0.0.bias'] = (first_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.arange(int(rate * 256))
        self.idx['layer3.1.0.weight'], self.idx['layer3.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        third_channels = torch.arange(int(rate * 256))
        self.idx['layer3.2.0.weight'], self.idx['layer3.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        first_channels = torch.arange(int(rate * 512))
        self.idx['layer4.0.0.weight'], self.idx['layer4.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.arange(int(rate * 512))
        self.idx['layer4.1.0.weight'], self.idx['layer4.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        third_channels = torch.arange(int(rate * 512))
        self.idx['layer4.2.0.weight'], self.idx['layer4.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        first_channels = torch.arange(int(rate * 512))
        self.idx['layer5.0.0.weight'], self.idx['layer5.0.0.bias'] = (first_channels, third_channels, torch.arange(3),
                                                                       torch.arange(3)), first_channels
        second_channels = torch.arange(int(rate * 512))
        self.idx['layer5.1.0.weight'], self.idx['layer5.1.0.bias'] = (second_channels, first_channels, torch.arange(3),
                                                                       torch.arange(3)), second_channels
        third_channels = torch.arange(int(rate * 512))
        self.idx['layer5.2.0.weight'], self.idx['layer5.2.0.bias'] = (third_channels, second_channels, torch.arange(3),
                                                                       torch.arange(3)), third_channels
        first_channels = torch.arange(int(rate * 256))
        self.idx['layer6.0.weight'], self.idx['layer6.0.bias'] = (first_channels, third_channels), first_channels
        second_channels = torch.arange(int(rate * 128))
        self.idx['layer7.0.weight'], self.idx['layer7.0.bias'] = (second_channels, first_channels), second_channels
        self.idx['layer8.weight'], self.idx['layer8.bias'] = (torch.arange(10), second_channels), torch.arange(10)

    def clear_idx(self):
        self.idx.clear()

    def get_attar(self, attar_name):
        names = attar_name.split('.')
        obj = self
        for x in names:
            obj = getattr(obj, x)
        return obj


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


class Model(VGG16):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        # 适配当前框架的参数
        super(Model, self).__init__(out_dim)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)


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