import argparse
import importlib
import json
import os
from collections import Counter
import numpy as np

def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of dataset;', type=str, default='cifar10')
    parser.add_argument('--dist', help='type of distribution;', type=int, default=18)
    parser.add_argument('--skew', help='the degree of niid;', type=float, default=0)
    parser.add_argument('--num_clients', help='the number of clients;', type=int, default=10)

    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

def analyze_data_distribution(generator):
    """分析并记录数据分布统计信息"""
    print('-----------------------------------------------------')
    print('Analyzing data distribution...')
    
    # 读取生成的联邦数据
    taskpath = generator.taskpath
    data_file = os.path.join(taskpath, 'data.json')
    
    if not os.path.exists(data_file):
        print("Data file not found. Cannot analyze distribution.")
        return
    
    with open(data_file, 'r') as f:
        feddata = json.load(f)
    
    # 统计信息字典
    stats = {
        'server_validation_size': len(feddata['dvalid']['y']),
        'server_test_size': len(feddata['dtest']['y']),
        'clients_stats': {},
        'total_train_samples': 0,
        'total_valid_samples': 0,
        'class_distribution_summary': {}
    }
    
    # 获取类别数量
    all_labels = set(feddata['dvalid']['y'] + feddata['dtest']['y'])
    for client_name in feddata['client_names']:
        all_labels.update(feddata[client_name]['dtrain']['y'])
        all_labels.update(feddata[client_name]['dvalid']['y'])
    
    num_classes = len(all_labels)
    class_labels = sorted(list(all_labels))
    
    print(f"Dataset: {generator.benchmark}")
    print(f"Number of classes: {num_classes}")
    print(f"Class labels: {class_labels}")
    print(f"Number of clients: {generator.num_clients}")
    print(f"Distribution type: {generator.dist_name} (ID: {generator.dist_id})")
    print(f"Skewness: {generator.skewness}")
    print(f"Server validation set size: {stats['server_validation_size']}")
    print(f"Server test set size: {stats['server_test_size']}")
    print()
    
    # 分析每个客户端的数据分布
    for client_name in feddata['client_names']:
        client_data = feddata[client_name]
        
        # 训练集统计
        train_labels = client_data['dtrain']['y']
        train_class_count = Counter(train_labels)
        train_total = len(train_labels)
        
        # 验证集统计
        valid_labels = client_data['dvalid']['y']
        valid_class_count = Counter(valid_labels)
        valid_total = len(valid_labels)
        
        # 总计
        total_samples = train_total + valid_total
        
        # 合并训练和验证集的类别统计
        combined_class_count = Counter()
        combined_class_count.update(train_class_count)
        combined_class_count.update(valid_class_count)
        
        stats['clients_stats'][client_name] = {
            'train_samples': train_total,
            'valid_samples': valid_total,
            'total_samples': total_samples,
            'train_class_distribution': dict(train_class_count),
            'valid_class_distribution': dict(valid_class_count),
            'combined_class_distribution': dict(combined_class_count)
        }
        
        stats['total_train_samples'] += train_total
        stats['total_valid_samples'] += valid_total
        
        # 打印客户端信息
        print(f"{client_name}:")
        print(f"  训练样本数: {train_total}")
        print(f"  验证样本数: {valid_total}")
        print(f"  总样本数: {total_samples}")
        print(f"  训练集类别分布: {dict(train_class_count)}")
        print(f"  验证集类别分布: {dict(valid_class_count)}")
        print(f"  总体类别分布: {dict(combined_class_count)}")
        print()
    
    # 计算全局统计
    print("全局统计:")
    print(f"  所有客户端训练样本总数: {stats['total_train_samples']}")
    print(f"  所有客户端验证样本总数: {stats['total_valid_samples']}")
    print(f"  服务器验证集大小: {stats['server_validation_size']}")
    print(f"  服务器测试集大小: {stats['server_test_size']}")
    
    # 计算类别分布汇总
    global_class_count = Counter()
    for client_name in feddata['client_names']:
        client_combined = stats['clients_stats'][client_name]['combined_class_distribution']
        global_class_count.update(client_combined)
    
    stats['class_distribution_summary'] = dict(global_class_count)
    print(f"  客户端总体类别分布: {dict(global_class_count)}")
    
    # 计算数据分布的不平衡程度
    client_sample_counts = [stats['clients_stats'][name]['total_samples'] for name in feddata['client_names']]
    mean_samples = np.mean(client_sample_counts)
    std_samples = np.std(client_sample_counts)
    cv_samples = std_samples / mean_samples if mean_samples > 0 else 0
    
    print(f"  客户端样本数统计: 平均={mean_samples:.2f}, 标准差={std_samples:.2f}, 变异系数={cv_samples:.4f}")
    print()
    
    # 保存统计信息到文件
    stats_file = os.path.join(taskpath, 'data_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"数据分布统计信息已保存到: {stats_file}")
    print('Done.')

if __name__ == '__main__':
    option = read_option()
    TaskGen = getattr(importlib.import_module('.'.join(['benchmark', option['dataset'], 'core'])), 'TaskGen')
    generator = TaskGen(dist_id = option['dist'], skewness = option['skew'], num_clients=option['num_clients'])
    generator.run()
    
    # 分析数据分布
    analyze_data_distribution(generator)

