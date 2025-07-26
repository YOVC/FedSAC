#!/bin/bash

# 生成使用Dirichlet分布的MNIST数据集
echo "生成Dirichlet分布的cifar10数据集..."
python generate_fedtask.py --dataset cifar10 --dist 2 --skew 0.5 --num_clients 10
# 运行Dirichlet分布的cifar10数据集 standalone模式
echo "运行Dirichlet分布的cifar10数据集 standalone模式..."
python main.py \
    --task cifar10_cnum10_dist2_skew0.5_seed0 \
    --algorithm standalone \
    --model resnet18 \
    --num_epochs 25 \
    --learning_rate 0.01 \
    --batch_size 32 \
    --eval_interval 1 \
    --seed 0 \
    --gpu 0 \
    --neuron_eval_interval 1

# 运行新的固定裁剪算法
echo "运行WeightValue_FixedPruning算法..."
python main.py \
    --task cifar10_cnum10_dist2_skew0.5_seed0 \
    --algorithm WeightValue_FixedPruning \
    --model resnet18 \
    --num_rounds 50 \
    --num_epochs 15 \
    --learning_rate 0.01 \
    --batch_size 32 \
    --eval_interval 1 \
    --seed 0 \
    --gpu 0 \
    --neuron_eval_interval 1


echo "运行HessianValue_FixedPruning算法..."
python main.py \
    --task cifar10_cnum10_dist2_skew0.5_seed0 \
    --algorithm HessianValue_FixedPruning \
    --model resnet18 \
    --num_rounds 50 \
    --num_epochs 15 \
    --learning_rate 0.01 \
    --batch_size 32 \
    --eval_interval 1 \
    --seed 0 \
    --gpu 0 \
    --neuron_eval_interval 1
echo "实验完成！"