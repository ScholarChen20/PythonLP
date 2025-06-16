import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from chap2.CNN.CNN import CNN

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
train_loader = DataLoader(training_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=True)

train_feature , train_label = next(iter(train_loader))
print(train_feature.size())
# print(train_label.size())
# img = train_feature[0].squeeze()
# # print(train_feature[0].squeeze())
# label=train_label[0].shape
# plt.imshow(img,cmap="gray")
# plt.show()

cnn=CNN()
input = train_feature
output = cnn(input)
print(output.shape)