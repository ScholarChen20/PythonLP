from abc import ABC

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.nn import functional as F

# 修正后的LinearLayer
class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.W = nn.Parameter(
            torch.randn(input_dim, output_dim, device=device) * np.sqrt(2. / input_dim)
        )
        self.b = nn.Parameter(torch.zeros(output_dim, device=device))

    def forward(self, x):
        return x @ self.W + self.b


# 修正后的BatchNorm1d
class BatchNorm1d:
    def __init__(self, num_features, momentum=0.1):
        self.momentum = momentum
        self.gamma = torch.ones(num_features,device='cuda')
        self.beta = torch.zeros(num_features,device='cuda')
        self.running_mean = torch.zeros(num_features,device='cuda')
        self.running_var = torch.ones(num_features,device='cuda')

    def forward(self, x, training=True):
        if training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return self.gamma * x_norm + self.beta

# 激活函数实现
def relu(x):
    return torch.maximum(torch.zeros_like(x), x)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Softmax实现（数值稳定版）
def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return exp_x / exp_x.sum(dim=1, keepdim=True)

# 交叉熵损失实现
def cross_entropy(y_pred, y_true, smoothing=0.1):
    num_classes = y_pred.shape[1]
    y_true_smooth = y_true * (1 - smoothing) + smoothing / num_classes
    return -torch.mean(torch.sum(y_true_smooth * torch.log(y_pred + 1e-10), dim=1))

# MLP网络定义
class MLP(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.fc1 = LinearLayer(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = LinearLayer(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = LinearLayer(256, 10)
        self.act = relu if activation == 'relu' else sigmoid
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, training=True):
        x = self.fc1.forward(x)
        x = self.bn1.forward(x)
        x = self.act(x)
        x = self.dropout(x) if training else x

        x = self.fc2.forward(x)
        x = self.bn2.forward(x)
        x = self.act(x)
        x = self.dropout(x) if training else x

        x = self.fc3.forward(x)
        return softmax(x)

#改进后网络
class LeNet5(nn.Module, ABC):
    """
    略微修改过的 LeNet5 模型
    Attributes:
        need_dropout (bool): 是否需要增加随机失活层
        conv1 (nn.Conv2d): 卷积核1，默认维度 (6, 5, 5)
        pool1 (nn.MaxPool2d): 下采样函数1，维度 (2, 2)
        conv2 (nn.Conv2d): 卷积核2，默认维度 (16, 5, 5)
        pool2 (nn.MaxPool2d): 下采样函数2，维度 (2, 2)
        conv3 (nn.Conv2d): 卷积核3，默认维度 (120, 5, 5)
        fc1 (nn.Linear): 全连接函数1，维度 (120, 84)
        fc2 (nn.Linear): 全连接函数2，维度 (84, 10)
        dropout (nn.Dropout): 随机失活函数
    """
    def __init__(self, dropout_prob=0., halve_conv_kernels=False):
        """
        初始化模型各层函数
        :param dropout_prob: 随机失活参数
        :param halve_conv_kernels: 是否将卷积核数量减半
        """
        super(LeNet5, self).__init__()
        kernel_nums = [6, 16]
        if halve_conv_kernels:
            kernel_nums = [num // 2 for num in kernel_nums]
        self.need_dropout = dropout_prob > 0

        # 卷积层 1，6个 5*5 的卷积核
        # 由于输入图像是 28*28，所以增加 padding=2，扩充到 32*32
        self.conv1 = nn.Conv2d(1, kernel_nums[0], (5, 5), padding=2)
        # 下采样层 1，采样区为 2*2
        self.pool1 = nn.MaxPool2d((2, 2))
        # 卷积层 2，16个 5*5 的卷积核
        self.conv2 = nn.Conv2d(kernel_nums[0], kernel_nums[1], (5, 5))
        # 下采样层 2，采样区为 2*2
        self.pool2 = nn.MaxPool2d((2, 2))
        # 卷积层 3，120个 5*5 的卷积核
        self.conv3 = nn.Conv2d(kernel_nums[1], 120, (5, 5))
        # 全连接层 1，120*84 的全连接矩阵
        self.fc1 = nn.Linear(120, 84)
        # 全连接层 2，84*10 的全连接矩阵
        self.fc2 = nn.Linear(84, 10)
        # 随机失活层，失活率为 dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        前向传播函数，返回给定输入数据的预测标签数组
        :param x: 维度为 (batch_size, 28, 28) 的图像数据
        :return: 维度为 (batch_size, 10) 的预测标签
        """
        x = x.view(-1, 1, 28, 28) # (batch_size, 1, 28, 28)
        feature_map = self.conv1(x)             # (batch_size, 6, 28, 28)
        feature_map = self.pool1(feature_map)   # (batch_size, 6, 14, 14)
        feature_map = self.conv2(feature_map)   # (batch_size, 16, 10, 10)
        feature_map = self.pool2(feature_map)   # (batch_size, 16, 5, 5)
        feature_map = self.conv3(feature_map).squeeze()     # (batch_size, 120)
        out = self.fc1(feature_map)             # (batch_size, 84)
        if self.need_dropout:
            out = self.dropout(out)             # (batch_size, 10)
        out = self.fc2(out)                     # (batch_size, 10)
        return out

# 数据加载
# 训练集数据增强
transform_train = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # 减小旋转角度
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # Random affine transformation of the image keeping center invariant,
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

# 测试集变换
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])
train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=128, shuffle=False)

# 训练函数
def train(model, train_loader, test_loader, epochs=20, lr=0.1):
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        lr = 0.01 * (0.5 ** (epoch // 10))  # 学习率衰减
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        for X, y in train_loader:
            X, y = X.cuda(), y.cuda()  # 移至GPU
            # 前向传播
            outputs = model(X)
            # loss = cross_entropy(outputs, torch.nn.functional.one_hot(y, 10).float())

            # total_loss += loss.item()
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()

        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0
        all_test_preds, all_test_labels = [], []
        all_train_preds, all_train_labels = [], []
        with torch.no_grad():
            # 测试集预测
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.cuda(), y_test.cuda()
                outputs = model(X_test)
                pred = outputs.argmax(dim=1)
                test_correct += (pred == y_test).sum().item()

                # 修正这里 ↓↓↓
                all_test_preds.extend(pred.cpu().numpy())
                all_test_labels.extend(y_test.cpu().numpy())

            # 训练集预测
            for X_train, y_train in train_loader:
                X_train, y_train = X_train.cuda(), y_train.cuda()
                outputs = model(X_train)
                pred = outputs.argmax(dim=1)

                # 修正这里 ↓↓↓
                all_train_preds.extend(pred.cpu().numpy())
                all_train_labels.extend(y_train.cpu().numpy())

        # 记录指标
        train_acc.append(correct / len(train_set))
        test_acc.append(test_correct / len(test_set))
        train_losses.append(total_loss / len(train_loader))

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Acc: {train_acc[-1]:.4f} | Test Acc: {test_acc[-1]:.4f}")

    # 绘制混淆矩阵
    cm_test = confusion_matrix(all_test_labels, all_test_preds)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp_test.plot()
    plt.title("Test Confusion Matrix")
    plt.show()

    # 绘制训练集混淆矩阵
    cm_train = confusion_matrix(all_train_labels, all_train_preds)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_train.plot()
    plt.title("Train Confusion Matrix")
    plt.show()

    return train_acc, test_acc, cm_train, cm_test

if __name__ == '__main__':
    ## 初始化模型并移动到GPU
    model = LeNet5(dropout_prob=0.5, halve_conv_kernels=False)
    model = model.to('cuda')

    # 检查设备一致性
    # print(f"Model parameters on: {next(model.parameters()).device}")

    # 开始训练
    train_acc, test_acc, cm_train, cm_test = train(model, train_loader, test_loader, epochs=20)

    # 绘制训练曲线
    plt.plot(train_acc, label='Train Acc')
    plt.plot(test_acc, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
