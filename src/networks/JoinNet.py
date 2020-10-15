from base.base_net import BaseNet


class JoinNet(BaseNet):
    """
    JoinNet的网络结构
                            svddNet
        lstmNet Encoder ---|
                            lstmNet Decoder

    要想实现联合训练，必须获取训练好的 lstmNet 的 svddNet
    """
    def __init__(self, lstm_net, svdd_net):
        super().__init__()
        self.lstm_net = lstm_net
        self.svdd_net = svdd_net

    def forward(self, x):
        """ 联合训练 """
        x = self.lstm_net.encode(x)
        y0 = self.lstm_net.decode(x)
        y1 = self.svdd_net.compute(x)
        return y0, y1
