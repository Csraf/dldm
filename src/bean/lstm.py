import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network
from optim.lstm_trainer import LstmTrainer


class Lstm(object):
    """
        dldm-lstm 部分的逻辑代码

    """

    def __init__(self, n_features=8):
        """初始化lstm参数."""
        self.n_features = n_features

        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

        self.train_code = None
        self.train_label = None

        self.test_code = None
        self.test_label = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def set_network(self, net_name):
        """Builds the neural networks ."""

        self.net_name = net_name
        self.net = build_network(net_name, self.n_features)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'RMSprop', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Lstm model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = LstmTrainer(optimizer_name,
                                   lr=lr,
                                   n_epochs=n_epochs,
                                   lr_milestones=lr_milestones,
                                   batch_size=batch_size,
                                   weight_decay=weight_decay,
                                   device=device,
                                   n_jobs_dataloader=n_jobs_dataloader,
                                   n_features=self.n_features)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Lstm model on the test data."""

        if self.trainer is None:
            self.trainer = LstmTrainer(device=device, n_jobs_dataloader=n_jobs_dataloader)

        # 必须测试两次，得到训练集和测试集的码
        self.trainer.test(dataset, self.net, is_test=1)  # 训练集
        self.trainer.test(dataset, self.net, is_test=0)  # 测试集

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def load_code(self):
        """ 加载 lstm 网络输出的 code 和 label """
        return self.trainer.load_code()
