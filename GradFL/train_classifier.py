import copy

import torch
import random
import os
import yaml
import wandb
import torch.nn.functional as F
from collections import OrderedDict

from config import cfg
from utils import set_control, set_seed, get_model, make_scheduler, make_optimizer, generate_submodel_rate, \
    generate_fix_submodel_rate_list
from utils import load_params_to_client_model
from data import get_transform, get_dataset, SplitDataset, non_iid, make_dataloader, get_inferen_data, \
    dataset_class_list


def main():
    set_control(cfg)
    set_seed(cfg['seed'])
    print('Config is OK!\n {}'.format(cfg))
    device = torch.device(cfg['device'])

    # use wandb to record
    if cfg['use_wandb']:
        wandb.init(
            project=cfg['project_name'],
            group=cfg['group_name'],
            config=cfg
        )
        print("Wandb is inited!")

    # global_dataset and non-iid progress
    transform = get_transform(cfg['dataset'], cfg['model_name'])
    dataset = get_dataset(cfg['dataset'], transform)
    print('Dataset {} is OK!'.format(cfg['dataset']))
    data_train_split, label_train_split = non_iid(dataset['train'], cfg['numbers'], cfg['shardperuser'])

    # global_model
    global_model = get_model(cfg['model_name'], dataset, mode_rate=1)
    # global_model = global_model.to(device)

    # optimizer
    optimizer = make_optimizer(global_model, cfg['lr'])

    # scheduler
    scheduler = make_scheduler(optimizer)

    # fix mode sub_rate list
    sub_model_rate_list = generate_fix_submodel_rate_list()

    # for grad-base mode we need konw class-list
    class_list = dataset_class_list(dataset['train'])

    # for grad-base pre-train the global model
    pre_run_train(cfg, global_model, class_list, dataset['train'], optimizer, device)

    # record last round users params and rate
    # last_client_info = {i: [1, global_model.state_dict()] for i in range(cfg['numbers'])}
    last_client_info = {i: [] for i in range(cfg['numbers'])}
    print('Experiment is beginning!')
    for i in range(cfg['rounds']):
        print(' ROUNDS {}'.format(i + 1), end=' ')
        num_activate_users = int(cfg['numbers'] * cfg['frc'])
        user_idxs = torch.randperm(cfg['numbers'])[:num_activate_users]
        clients_models_params = {}
        clients_models_shape = {}
        for user_idx in user_idxs:
            # sub_dataset
            client_dataset = SplitDataset(dataset['train'], data_train_split[int(user_idx)])
            client_dataloader = make_dataloader(client_dataset, batch_size=cfg['batchsize'])
            label_mask = torch.zeros(len(dataset['train'].classes), device=device)
            label_mask[label_train_split[user_idx]] = 1

            # sub_model
            rate = generate_submodel_rate(cfg['client_state'], sub_model_rate_list, user_idx)
            client_model = get_model(cfg['model_name'], dataset, mode_rate=rate)

            # for GradFL sever don't know the client labels, so we should inferen its labels
            user_idx_label = get_client_dataset_label(user_idx, cfg, class_list, last_client_info, dataset,
                                                      label_train_split, device)

            inferen_data = get_inferen_data(cfg['mode'], cfg['inferen_batch'], class_list, user_idx_label,
                                            client_dataset, dataset['train'])
            load_params_to_client_model(global_model, client_model, inferen_data, cfg['mode'], device, rate,
                                        cfg['select_mode'], label_mask, clients_models_shape, user_idx)
            # train model
            run_train(client_model, client_dataloader, label_mask, device, optimizer, clients_models_params, user_idx)
            last_client_info[int(user_idx)].clear()
            last_client_info[int(user_idx)].extend([rate, copy.deepcopy(client_model.state_dict())])
        combine_global_model_from_clients(global_model, user_idxs, clients_models_shape, clients_models_params)
        run_global_test(global_model, dataset, device)
        scheduler.step()
    if cfg['user_wandb']:
        wandb.finish()


def run_train(client_model, client_dataloader, label_mask, device, optimizer, clients_models_params, user_idx):
    client_model.to(device)
    lr = optimizer.param_groups[0]['lr']
    local_optimizer = make_optimizer(client_model, lr)
    client_model.train()
    local_loss = 0
    for local_epoch in range(cfg['epoch']):
        for batch in client_dataloader:
            img, target = batch['img'], batch['label']
            img, target = img.to(device), target.to(device)
            client_model.zero_grad()
            output = client_model(img)
            output = output.masked_fill(label_mask == 0, 0)
            loss = F.cross_entropy(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1)
            local_optimizer.step()
            # 将loss相加
            local_loss += loss
    # print('     user_idx {}: Local Loss {}'.format(user_idx, local_loss))
    client_model.cpu()
    clients_models_params[int(user_idx)] = copy.deepcopy(client_model.state_dict())


def pre_run_train(cfg, model, class_list, dataset, optimizer, deivce):
    if cfg['mode'] == 'awareGrad' and cfg['is_pre_train']:
        model.to(deivce)
        lr = optimizer.param_groups[0]['lr']
        model_optimizer = make_optimizer(model, lr)

        shardperuser = cfg['shardperuser']
        inferen_batch = cfg['inferen_batch']
        public_per_class_num = inferen_batch // shardperuser

        public_dataset_label = []
        for i in range(len(class_list)):
            public_dataset_label.extend(class_list[i][: public_per_class_num])
        public_dataset = SplitDataset(dataset, public_dataset_label)
        public_dataset_dataloader = make_dataloader(public_dataset, public_per_class_num)

        pre_loss = []
        model.train()
        for local_epoch in range(cfg['pre_train_epoch']):
            local_loss = 0
            for batch in public_dataset_dataloader:
                img, target = batch['img'].to(deivce), batch['label'].to(deivce)
                model.zero_grad()
                output = model(img)
                loss = F.cross_entropy(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()
                local_loss += loss.data
            pre_loss.append(local_loss)
        print('Pre-train loss: {}'.format(pre_loss))
        model.cpu()
    else:
        print('mode {} don\'t need pre-train!'.format(cfg['mode']))


def combine_global_model_from_clients(global_model, user_idxs, clients_models_shape, clients_models_params):
    # 将clients集合赋值给global model
    client_num_model_param = OrderedDict()
    global_temp_params = OrderedDict()
    for k, v in global_model.state_dict().items():
        client_num_model_param[k] = torch.ones_like(v)
        global_temp_params[k] = copy.deepcopy(v)
    for user_idx in user_idxs:
        for k, v in clients_models_params[int(user_idx)].items():
            temp_shape = clients_models_shape[int(user_idx)][k]
            if k in global_temp_params:
                if v.dim() > 1:
                    global_temp_params[k][torch.meshgrid(temp_shape, indexing='ij')] += v
                    client_num_model_param[k][torch.meshgrid(temp_shape, indexing='ij')] += 1
                else:
                    global_temp_params[k][temp_shape] += v
                    client_num_model_param[k][temp_shape] += 1
            else:
                raise NameError('{} not in global_temp_params'.format(k))
    # 获得取平均值之后的global_params
    for k in global_temp_params:
        global_temp_params[k] /= client_num_model_param[k]
    global_model.load_state_dict(global_temp_params)


def run_global_test(global_model, dataset, device):
    # 测试global model
    test_dataloader = make_dataloader(dataset['test'], batch_size=32)
    correct, total = 0, 0
    global_model.to(device)
    global_model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            img, target = batch['img'], batch['label']
            img, target = img.to(device), target.to(device)
            output = global_model(img)
            loss = F.cross_entropy(output, target)
            _, predicted = torch.max(output, dim=1)
            test_loss += loss.data
            total += img.shape[0]
            correct += (predicted == target).sum().item()
    print(' Global Model Acc: {}, Test Loss: {}'.format(correct / total, test_loss))
    if cfg['use_wandb']:
        wandb.log({'acc': correct / total, 'loss': test_loss})
    global_model.cpu()


def get_grad_client_dataset_label(user_id, cfg, class_list, last_client_info, dataset, device):
    model_name = cfg['model_name']
    numbers = cfg['numbers']
    shardperuser = cfg['shardperuser']
    inferen_batch = cfg['inferen_batch']
    public_per_class_num = inferen_batch // shardperuser
    if last_client_info[int(user_id)]:
        model_rate = last_client_info[int(user_id)][0]
        model_params = last_client_info[int(user_id)][1]
        model = get_model(model_name, dataset, model_rate)
        model.load_state_dict(model_params)
        model = model.to(device)
        user_class_acc = []
        '''
        for i in range(len(class_list)):
            i_label_list = class_list[i][:public_per_class_num]
            i_dataset = SplitDataset(dataset['train'], i_label_list)
            i_dataloader = make_dataloader(i_dataset, public_per_class_num)
            correct_num = 0
            with torch.no_grad():
                for batch in i_dataloader:
                    img, target = batch['img'].to(device), batch['label'].to(device)
                    output = model(img)
                    _, predicted = torch.max(output, dim=1)
                    correct_num += (predicted == target).sum().item()
                user_class_acc.append(correct_num)
        '''
        label_list = []
        for i in range(len(class_list)):
            label_list.extend(class_list[i][:public_per_class_num])
        i_dataset = SplitDataset(dataset['train'], label_list)
        i_dataloader = make_dataloader(i_dataset, public_per_class_num, shuffle=False)
        with torch.no_grad():
            for batch in i_dataloader:
                correct_num = 0
                img, target = batch['img'].to(device), batch['label'].to(device)
                output = model(img)
                _, predicted = torch.max(output, dim=1)
                correct_num += (predicted == target).sum().item()
                user_class_acc.append(correct_num)
        user_idx_label = [i for i, _ in
                          sorted(enumerate(user_class_acc), key=lambda x: x[1], reverse=True)[: shardperuser]]
        # return user_idx_label
    else:
        user_idx_label = torch.randperm(len(class_list))[: shardperuser].tolist()
    return user_idx_label


def get_client_dataset_label(user_id, cfg, class_list, last_client_info, dataset, label_train_split, device):
    if cfg['mode'] == 'awareGrad' and (not cfg['client_send_label']):
        return get_grad_client_dataset_label(user_id, cfg, class_list, last_client_info, dataset, device)
    elif cfg['mode'] == 'aware' or cfg['client_send_label']:
        return label_train_split[user_id]
    else:
        return []


if __name__ == "__main__":
    main()
