## INSTRUCTION
This code is the implement of the Grad-based extraction sub-model Federated Learning.

## DEMO
python train_classifier.py --rounds 100 --lr 0.01 --mode roll --model_name conv --dataset emnist --shardperuser 2 --group_name xxx --client_send_label --device 0

### resnet34 + tinyimagenet
#### awareGrad
python train_classifier.py --rounds 100 --lr 0.01 --mode awareGrad --model_name resnet34 --dataset tinyimagenet --shardperuser 20 --group_name xxx --client_send_label --device 0
#### roll
python train_classifier.py --rounds 100 --lr 0.01 --mode roll --model_name resnet34 --dataset tinyimagenet --shardperuser 20 --group_name xxx --device 0

### parameters mode
awareGrad -> FedGSE

aware -> FedDSE

roll -> FedRolex

rand -> Federated Dropout

hetero -> HeteroFL

fedavg -> FedAVG

