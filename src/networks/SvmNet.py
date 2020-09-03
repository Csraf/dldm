import torch
import torch.nn as nn
from base.base_net import BaseNet

"""
    rbm + svm
    网络的输入和输出的维度都是8，代表8个特征。

"""


class SVMNet(BaseNet):
    def __init__(self, n_features=9):
        super(SVMNet, self).__init__()
        self.l1 = nn.Linear(n_features, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = torch.sigmoid(self.l3(x))
        return x

