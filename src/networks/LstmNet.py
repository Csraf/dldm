import torch
import torch.nn as nn
from src.base.base_net import BaseNet


class LstmNet(BaseNet):
    """
    LstmNet 代表 lstm-autoencoder 的网络结构

    结构：
        encoder
            输入层：[128, 1, 9]
            隐藏层：[128, 1, 64]
            折叠层：[128, 64]
            连接层：[128, 8]

        decoder
            输入层：[128, 1, 8]
            隐藏层：[128, 1, 64]
            折叠层：[128, 64]
            连接层：[128, 9]

    注意：
        1. 在网络中解码层的输出部分加入了 sigmoid 模块，限制输出范围[0,1]，
           这是因为后面的 svdd-autoencoder 要求网络的输入范围为[0,1].
        2. 下面只是简单地展示了单层的 lstm 结构，后面根据需求决定是否更改.

    改动：
        1. 特征修改为12个，改变输入维度

    """

    def __init__(self, n_features=9):
        super().__init__()
        self.n_features = n_features
        self.lstm_encoder = nn.LSTM(
            input_size=self.n_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fe = nn.Linear(in_features=64, out_features=8)

        self.lstm_decoder = nn.LSTM(
            input_size=8,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fd = nn.Linear(in_features=64, out_features=self.n_features)

    def forward(self, x):
        """ x: code, y: out """
        x, _ = self.lstm_encoder(x)
        x = x[:, -1, :]
        x = self.fe(x)

        y = x.view(-1, 1, 8)

        y, _ = self.lstm_decoder(y)
        y = y[:, -1, :]
        y = self.fd(y)
        return x, y
