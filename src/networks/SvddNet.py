import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_net import BaseNet


class SvddNet(BaseNet):
    """  SvddNet 代表 deep-svdd 的网络结构 """

    def __init__(self):
        super().__init__()

        self.rep_dim = 32

        self.fc1 = nn.Linear(8,  16, bias=False)
        self.fc2 = nn.Linear(16, 32, bias=False)
        self.fc3 = nn.Linear(32, 64, bias=False)
        self.fc4 = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        return x


class SvddNet_Autoencoder(BaseNet):
    """  SvddNet_Autoencoder 代表 svdd-autoencoder 的网络结构 """

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        # encoder
        self.fe1 = nn.Linear(8, 16, bias=False)
        self.fe2 = nn.Linear(16, 32, bias=False)
        self.fe3 = nn.Linear(32, 64, bias=False)
        self.fe4 = nn.Linear(64, 32, bias=False)

        # decoder
        self.fd1 = nn.Linear(32, 64, bias=False)
        self.fd2 = nn.Linear(64, 32, bias=False)
        self.fd3 = nn.Linear(32, 16, bias=False)
        self.fd4 = nn.Linear(16, 8, bias=False)

    def forward(self, x):

        # encoder
        x = F.leaky_relu(self.fe1(x))
        x = F.leaky_relu(self.fe2(x))
        x = F.leaky_relu(self.fe3(x))
        x = self.fe4(x)

        # decoder
        x = self.fd1(x)
        x = F.leaky_relu(self.fd2(x))
        x = F.leaky_relu(self.fd3(x))
        x = self.fd4(x)
        return x
