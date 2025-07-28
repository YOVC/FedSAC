import torch
import torch.nn as nn
import copy
from collections import OrderedDict
from utils.fmodule import FModule

if __name__ == '__main__':
    from utils import init_param, Scaler  # When debug use this code
else:
    from .utils import init_param, Scaler  # When run main use it


class Conv(FModule):
    def __init__(self, data_shape, hidden_size, classes_size, rate=1, track=False):
        super().__init__()
        self.idx = OrderedDict()
        self.roll = 0

        self.rate = rate
        self.scaler = Scaler(rate)
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.classes_size = classes_size

        # self.in_channels = data_shape[0]
        self.conv = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = hidden_size[0]
        self.layer1 = self._make_layer(hidden_size[1], track)
        self.layer2 = self._make_layer(hidden_size[2], track)
        self.layer3 = self._make_layer(hidden_size[3], track)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_size[-1], classes_size)

    def _make_layer(self, out_channels, track):
        layer = nn.Sequential(OrderedDict([
            ('maxPool', nn.MaxPool2d(2)),
            ('conv', nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ('scale', Scaler(self.rate)),
            ('norm', nn.BatchNorm2d(out_channels, momentum=None, track_running_stats=track)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        self.in_channels = out_channels
        return layer

    def forward(self, input):
        out = self.relu(self.norm(self.scaler(self.conv(input))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(self.avgpool(out))
        out = self.linear(out)
        return out

    def get_idx_layer_aware(self, input, rate, device):
        start_channels = (torch.arange(self.data_shape[0]))
        out = self.relu(self.norm(self.scaler(self.conv(input))))
        first_channels = self.get_layer_aware_top(out, 0, int(rate * self.hidden_size[0]), device)
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['norm.weight'], self.idx['norm.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = [1, 2, 3]
        for s in seq_index:
            param_name = 'layer' + str(s)
            # exec('out = ' + fun_name + str(s) + '(out)')
            if s == 1:
                blc = self.layer1
            elif s == 2:
                blc = self.layer2
            elif s == 3:
                blc = self.layer3
            else:
                raise ValueError('error')
            out = blc(out)
            second_channels = self.get_layer_aware_top(out, s, int(rate * self.hidden_size[s]), device)
            self.idx[param_name + '.conv.weight'], self.idx[param_name + '.conv.bias'] = \
                (second_channels, first_channels), second_channels
            self.idx[param_name + '.norm.weight'], self.idx[param_name + '.norm.bias'] = \
                second_channels, second_channels
            first_channels = second_channels
        self.idx['linear.weight'] = (torch.arange(self.classes_size), first_channels)
        self.idx['linear.bias'] = torch.arange(self.classes_size)

    def get_layer_aware_top(self, input, name, k, device):
        # TODO()
        if name == 0:
            blc = self.layer1
            return get_diff_channels_sum(input, blc, k, device)
        elif name == 1:
            blc = self.layer2
            return get_diff_channels_sum(input, blc, k, device)
        elif name == 2:
            blc = self.layer3
            return get_diff_channels_sum(input, blc, k, device)
        elif name == 3:
            blc = nn.Sequential(
                self.avgpool,
                self.flatten,
                self.linear,
            )
            return get_diff_channels_sum(input, blc, k, device)
        else:
            raise ValueError('not match conv')

    def get_idx_aware(self, input, rate, topmode):
        start_channels = (torch.arange(self.data_shape[0]))
        out = self.relu(self.norm(self.scaler(self.conv(input))))
        first_channels = get_top_index(out, int(rate * self.hidden_size[0]), topmode)
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['norm.weight'], self.idx['norm.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = [1, 2, 3]
        for s in seq_index:
            param_name = 'layer' + str(s)
            # exec('out = ' + fun_name + str(s) + '(out)')
            if s == 1:
                blc = self.layer1
            elif s == 2:
                blc = self.layer2
            elif s == 3:
                blc = self.layer3
            else:
                raise ValueError('error')
            out = blc(out)
            second_channels = get_top_index(out, int(rate * self.hidden_size[s]), topmode)
            self.idx[param_name + '.conv.weight'], self.idx[param_name + '.conv.bias'] = \
                (second_channels, first_channels), second_channels
            self.idx[param_name + '.norm.weight'], self.idx[param_name + '.norm.bias'] = \
                second_channels, second_channels
            first_channels = second_channels
        self.idx['linear.weight'] = (torch.arange(self.classes_size), first_channels)
        self.idx['linear.bias'] = torch.arange(self.classes_size)

    def get_idx_aware_grad(self, rate, topmode):
        start_channels = (torch.arange(self.data_shape[0]))
        gradient_s = self.get_attar('conv.weight.grad')
        first_channels = get_top_index(gradient_s, int(rate * self.hidden_size[0]), topmode)
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['norm.weight'], self.idx['norm.bias'] = first_channels, first_channels
        seq_index = [1, 2, 3]
        for s in seq_index:
            param_name = 'layer' + str(s)
            # exec('out = ' + fun_name + str(s) + '(out)')
            if s == 1:
                layer_name = 'layer1.'
            elif s == 2:
                layer_name = 'layer2.'
            elif s == 3:
                layer_name = 'layer3.'
            else:
                raise ValueError('error')
            gradient_f = self.get_attar(layer_name + 'conv.grad')
            second_channels = get_top_index(gradient_f, int(rate * self.hidden_size[s]), topmode)
            self.idx[param_name + '.conv.weight'], self.idx[param_name + '.conv.bias'] = \
                (second_channels, first_channels), second_channels
            self.idx[param_name + '.norm.weight'], self.idx[param_name + '.norm.bias'] = \
                second_channels, second_channels
            first_channels = second_channels
        self.idx['linear.weight'] = (torch.arange(self.classes_size), first_channels)
        self.idx['linear.bias'] = torch.arange(self.classes_size)

    def get_idx_aware_grad(self, rate, topmode, gradient):
        start_channels = (torch.arange(self.data_shape[0]))
        gradient_s = gradient['conv']
        first_channels = get_top_index(gradient_s, int(rate * self.hidden_size[0]), topmode)
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['norm.weight'], self.idx['norm.bias'] = first_channels, first_channels
        seq_index = [1, 2, 3]
        for s in seq_index:
            param_name = 'layer' + str(s)
            # exec('out = ' + fun_name + str(s) + '(out)')
            if s == 1:
                layer_name = 'layer1.'
            elif s == 2:
                layer_name = 'layer2.'
            elif s == 3:
                layer_name = 'layer3.'
            else:
                raise ValueError('error')
            gradient_f = gradient[layer_name + 'conv']
            second_channels = get_top_index(gradient_f, int(rate * self.hidden_size[s]), topmode)
            self.idx[param_name + '.conv.weight'], self.idx[param_name + '.conv.bias'] = \
                (second_channels, first_channels), second_channels
            self.idx[param_name + '.norm.weight'], self.idx[param_name + '.norm.bias'] = \
                second_channels, second_channels
            first_channels = second_channels
        self.idx['linear.weight'] = (torch.arange(self.classes_size), first_channels)
        self.idx['linear.bias'] = torch.arange(self.classes_size)

    def get_idx_roll(self, rate):
        start_channels = (torch.arange(self.data_shape[0]))
        first_channels = torch.roll(torch.arange(self.hidden_size[0]),
                                    shifts=self.roll % self.hidden_size[0], dims=-1)[
                         : int(rate * self.hidden_size[0])]
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['norm.weight'], self.idx['norm.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = [1, 2, 3]
        for s in seq_index:
            param_name = 'layer' + str(s)
            second_channels = torch.roll(torch.arange(self.hidden_size[s]),
                                         shifts=self.roll % self.hidden_size[s], dims=-1)[
                              : int(rate * self.hidden_size[s])]
            self.idx[param_name + '.conv.weight'], self.idx[param_name + '.conv.bias'] = \
                (second_channels, first_channels), second_channels
            self.idx[param_name + '.norm.weight'], self.idx[param_name + '.norm.bias'] = \
                second_channels, second_channels
            first_channels = second_channels
        self.idx['linear.weight'] = (torch.arange(self.classes_size), first_channels)
        self.idx['linear.bias'] = torch.arange(self.classes_size)

    def get_idx_rand(self, rate):
        start_channels = (torch.arange(self.data_shape[0]))
        first_channels = torch.randperm(self.hidden_size[0])[:int(rate * self.hidden_size[0])]
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['norm.weight'], self.idx['norm.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = [1, 2, 3]
        for s in seq_index:
            param_name = 'layer' + str(s)
            second_channels = torch.randperm(self.hidden_size[s])[:int(rate * self.hidden_size[s])]
            self.idx[param_name + '.conv.weight'], self.idx[param_name + '.conv.bias'] = \
                (second_channels, first_channels), second_channels
            self.idx[param_name + '.norm.weight'], self.idx[param_name + '.norm.bias'] = \
                second_channels, second_channels
            first_channels = second_channels
        self.idx['linear.weight'] = (torch.arange(self.classes_size), first_channels)
        self.idx['linear.bias'] = torch.arange(self.classes_size)

    def get_idx_hetero(self, rate):
        start_channels = (torch.arange(self.data_shape[0]))
        first_channels = torch.arange(int(rate * self.hidden_size[0]))
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['norm.weight'], self.idx['norm.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = [1, 2, 3]
        for s in seq_index:
            param_name = 'layer' + str(s)
            second_channels = torch.arange(int(rate * self.hidden_size[s]))
            self.idx[param_name + '.conv.weight'], self.idx[param_name + '.conv.bias'] = \
                (second_channels, first_channels), second_channels
            self.idx[param_name + '.norm.weight'], self.idx[param_name + '.norm.bias'] = \
                second_channels, second_channels
            first_channels = second_channels
        self.idx['linear.weight'] = (torch.arange(self.classes_size), first_channels)
        self.idx['linear.bias'] = torch.arange(self.classes_size)

    def clear_idx(self):
        self.idx.clear()

    def get_attar(self, attar_name):
        names = attar_name.split('.')
        obj = self
        for x in names:
            obj = getattr(obj, x)
        return obj


def get_diff_channels_sum(input, blc, k, device):
    res = []
    for i in range(input.shape[1] + 1):
        temp = input.clone()
        if i != input.shape[1]:
            temp[:, i, :, :] = 0
        i_sum = blc(temp)
        i_sum = torch.sum(i_sum)
        res.append(i_sum)
    pre_sum = res.pop()
    pre_sum = pre_sum.to(device)
    diff_channels_sum = torch.tensor(res).to(device)
    diff_channels_sum = torch.abs(diff_channels_sum - pre_sum)
    return get_top_index(diff_channels_sum, k, topmode)


def get_top_index(x, k, topmode):
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


class Model(Conv):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        # 适配当前框架的参数
        data_shape = [in_channels, 32, 32]  # CIFAR-10图像大小为32x32
        hidden_size = [64, 128, 256, 512]  # 使用标准的通道数量
        super(Model, self).__init__(data_shape, hidden_size, out_dim)


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