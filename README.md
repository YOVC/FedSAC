# FedAVE: Adaptive Data Value Evaluation Framework for Collaborative Fairness in Federated Learning 
This repository is an implementation of the collaborative fairness of federated learning algorithm (under review).

## First
Run the command below to get the splited dataset MNIST:
```
python generate_fedtask.py --dataset cifar10 --dist 18 --skew 0 --num_clients 10
```

## Second
Run the command below to qucikly get a result of the basic algorithm FedAVE on MNIST with a simple MLP:
```
python main.py --task mnist_cnum10_dist18_skew0.0_seed0 --model mlp --algorithm FedAVE --num_rounds 20
--num_epochs 3 --learning_rate 0.15 --batch_size 32 --eval_interval 1 --Beta 5 --gpu 0
```


[0.1585, 0.2029, 0.2158, 0.2176, 0.3111, 0.3101, 0.3936, 0.3008, 0.3752, 0.3683]
```bash
python main.py --task cifar10_cnum10_dist18_skew0_seed0 --model cnn --algorithm FedSAC --num_rounds 200 --num_epochs 15 --learning_rate 0.03 --batch_size 32 --eval_interval 1 --beta 3 --gpu 0
```

```bash
python main.py --task cifar10_cnum10_dist18_skew0_seed0 --model cnn --algorithm standalone --num_rounds 200 --num_epochs 200 --learning_rate 0.03 --batch_size 32 --eval_interval 1 --beta 3 --gpu 0
```