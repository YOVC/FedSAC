import errno
import torch
import random
import numpy as np
import os
import torch.optim as optim
import copy
from collections import OrderedDict
import torch.nn.functional as F

from config import cfg
from globals import get_base_params
from models.conv import conv, get_model_params
from models.resnet import resnet18, get_model_params
from models.vgg import vgg16


# train_classify begin
def set_control(cfg):
    args = get_base_params()
    for k, v in args.items():
        if v is not None:
            cfg[k] = v
        # cfg[k] = v

def set_seed(seed):
    torch.manual_seed(seed)  # 设置cpu随机种子， 方便复现
    torch.cuda.manual_seed(seed)  # 设置GPU种子， 方便复现
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.set_deterministic_debug_mode('default')
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_model(model_name, dataset, mode_rate):
    if model_name == 'resnet18':
        hidden_size = [64, 128, 256, 512]
        num_blocks = [2, 2, 2, 2]
        shape = dataset['train'][0]['img'].shape
        num_class = len(dataset['train'].classes)
        model = resnet18(shape, hidden_size, num_blocks=num_blocks, num_classes=num_class, model_rate=mode_rate)
    elif model_name == 'resnet34':
        hidden_size = [64, 128, 256, 512]
        num_blocks = [3, 4, 6, 3]
        shape = dataset['train'][0]['img'].shape
        num_class = len(dataset['train'].classes)
        model = resnet18(shape, hidden_size, num_blocks=num_blocks, num_classes=num_class, model_rate=mode_rate)
    elif model_name == 'conv':
        hidden_size = [64, 128, 256, 512]
        shape = dataset['train'][0]['img'].shape
        num_class = len(dataset['train'].classes)
        model = conv(shape, hidden_size, num_class, model_rate=mode_rate)
    elif model_name == 'vgg16':
        # hidden_size = [l1, l2, l3, l4, l5, fc_hidden]
        hidden_size = [32, 32, 64, 64, 128, 512]
        num_classes = len(dataset['train'].classes)
        model = vgg16(hidden_size, num_classes, model_rate=mode_rate)
    else:
        raise ValueError('Dont have this Model: {}'.format(model_name))
    return model


def make_optimizer(model, lr):
    if cfg['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=cfg['momentum'],
                                  weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if cfg['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs']['global'],
                                                         eta_min=cfg['min_lr'])
    elif cfg['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['factor'],
                                                         patience=cfg['patience'], verbose=True,
                                                         threshold=cfg['threshold'], threshold_mode='rel',
                                                         min_lr=cfg['min_lr'])
    elif cfg['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg['lr'], max_lr=10 * cfg['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def generate_fix_submodel_rate_list():
    model_rate_list = cfg['submodel_rate']
    model_rate_list_probs = torch.tensor(cfg['probs'])
    m = torch.distributions.Categorical(probs=model_rate_list_probs)
    samples = m.sample(sample_shape=torch.Size([cfg['numbers']]))
    model_rate_list_fixed = torch.tensor(model_rate_list)[samples]
    model_rate_list_fixed = model_rate_list_fixed.tolist()
    return model_rate_list_fixed


def generate_submodel_rate(mode, sub_model_rate_list, user_idx):
    if mode == 'dynamic':
        m = torch.distributions.Categorical(probs=torch.tensor(cfg['probs']))
        return cfg['submodel_rate'][m.sample()]
    elif mode == 'fix':
        return sub_model_rate_list[user_idx]
    else:
        raise ValueError('Can not match mode {}'.format(mode))


def load_params_to_client_model(global_model, client_model, inference_data, mode, device, rate, select_mode,
                                label_mask, clients_models_shape, user_idx):
    if mode == 'aware':
        global_model.to(device)
        with torch.no_grad():
            inference_data = inference_data['img'].to(device)
            global_model.get_idx_aware(inference_data, rate, select_mode)
            global_model.cpu()
        client_model_idx = OrderedDict()
        for k, v in global_model.idx.items():
            temp = []
            if isinstance(global_model.idx[k], tuple):
                for index in range(len(global_model.idx[k])):
                    temp.append(global_model.idx[k][index].to('cpu'))
            else:
                temp.append(global_model.idx[k].to('cpu'))
            client_model_idx[k] = (copy.deepcopy(temp[0]), copy.deepcopy(temp[1])) if len(temp) > 1 else temp[0]
    elif mode == 'awareGrad':
        # gradients = get_model_gradient(global_model, inference_data, label_mask, device)
        # correct_seq_gradients = get_correct_seq_gradients(gradients)
        # global_model.get_idx_aware_grad(rate, correct_seq_gradients, select_mode)
        # client_model_idx = copy.deepcopy(global_model.idx)
        gradient = get_model_gradient(global_model, inference_data, label_mask, device)
        global_model.get_idx_aware_grad(rate, select_mode, gradient)
        # global_model.cpu()
        client_model_idx = copy.deepcopy(global_model.idx)
    elif mode == 'roll':
        global_model.get_idx_roll(rate)
        client_model_idx = copy.deepcopy(global_model.idx)
        global_model.roll += 1
    elif mode == 'rand':
        global_model.get_idx_rand(rate)
        client_model_idx = copy.deepcopy(global_model.idx)
    elif mode == 'hetero':
        global_model.get_idx_hetero(rate)
        client_model_idx = copy.deepcopy(global_model.idx)
    elif mode == 'fedavg':
        global_model.get_idx_hetero(1)
        client_model_idx = copy.deepcopy(global_model.idx)
    else:
        raise ValueError('Not valid model name')
    try:
        client_model_params = get_model_params(global_model, client_model_idx)
    except NameError as e:
        print(e)
    global_model.clear_idx()
    client_model.load_state_dict(client_model_params)
    clients_models_shape[int(user_idx)] = copy.deepcopy(client_model_idx)


# train_classify end

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


# get_gradient from global model using inference data
def get_model_gradient(model, inference_data, label_mask, device):
    # children_model_name = get_children_model_name(model)
    model.zero_grad()
    model.to(device)
    input = inference_data['img'].to(device)
    target = inference_data['label'].to(device)

    gradients = OrderedDict()
    handles = []

    def save_gradient(module, grad_in, grad_out, layer_name):
        '''
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            # if self.reshape_transform is not None:
            #     grad = self.reshape_transform(grad)
            gradients[layer_name] = grad.cpu().detach()
        '''
        gradients[layer_name] = torch.abs(grad_out[0].cpu())

    for name, module in model.named_modules():
        if 'conv' in name or 'linear' in name:
            handles.append(module.register_full_backward_hook(
                lambda module, grad_in, grad_out, layer_name=name: save_gradient(module, grad_in, grad_out, layer_name)))

    output = model(input)
    output = output.masked_fill(label_mask == 0, 0)
    loss = F.cross_entropy(output, target)
    loss.backward()
    for handle in handles:
        handle.remove()
    model.cpu()
    return gradients


def get_correct_seq_gradients(gradients):
    temp = list(gradients.items())[::-1]
    temp_seq, shortcut = [], []
    for k, v in temp:
        param_name = k.split('.')[-1]
        if param_name == 'shortcut':
            shortcut.append((k, v))
        elif param_name == 'conv2':
            temp_seq.append((k, v))
            temp_seq.append(shortcut[-1])
        else:
            temp_seq.append((k, v))
    return OrderedDict(temp_seq)


def check_exist(path):
    return os.path.exists(path)


def makedir_exits_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exits_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'numpy':
        np.save(path, input, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'numpy':
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return
