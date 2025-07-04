import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=nn.Conv2d(1,20,5)
        self.conv2=nn.Conv2d(20,20,5)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


cnn=Model()
print(cnn)