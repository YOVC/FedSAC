import torch


# 自定义钩子函数
def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad

    return hook


grads = []


# 定义一个简单的神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        # 在第一个卷积层上注册一个钩子函数
        self.conv1.register_backward_hook(save_grad(grads, 'conv1'))

        # 在第二个卷积层上注册一个钩子函数
        self.conv2.register_backward_hook(save_grad(grads, 'conv2'))

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义一个输入张量，并设置requires_grad=True
x = torch.randn(1, 3, 32, 32, requires_grad=True)

# 定义一个模型实例
model = Net()

# 对输入张量进行前向传播
y = model(x)

# 定义一个输出张量，作为损失函数的输入
loss = y.sum()

# 对损失函数进行反向传播
loss.backward()

# 保存第一个卷积层输出数据的梯度
conv1_grad = model.conv1.weight.grad

# 保存第二个卷积层输出数据的梯度
conv2_grad = model.conv2.weight.grad
