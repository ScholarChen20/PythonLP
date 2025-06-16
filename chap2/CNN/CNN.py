import torch
from torch import nn
from torch.nn import Conv2d, Flatten, MaxPool2d, Sequential, Linear

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# print(f"Using {device} device")

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.model1=Sequential(
            Conv2d(1,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(576,64),
            Linear(64,10)
        )
        self.softmax = nn.Softmax(1)

    def forward(self,x):
        x = self.model1(x)
        x = self.softmax(x)
        return x

# cnn = CNN()
# print(cnn)
# # input = torch.ones((64,3,32,32))
# output = cnn(input)
# print(output.shape)