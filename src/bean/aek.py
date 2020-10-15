import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.aek_trainer import AEKTrainer


class AEK(object):
    """
        对比实验 ae + kmeans
    """

    def __init__(self):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        self.net_name = None
        self.net = None  # neural networks \phi

        self.ae_net = None  # autoencoder networks for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
            'test_score': None,
            'test_f_score': None,
            'test_mcc': None,
            'test_ftr': None,
            'test_tpr': None,
        }

    def set_network(self, net_name):
        """Builds the neural networks ."""

        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        self.ae_net = build_autoencoder(self.net_name)
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AEKTrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                     batch_size=batch_size, weight_decay=weight_decay, device=device,
                                     n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.ae_trainer.test(dataset=dataset, ae_net=self.ae_net, flg=1)  # 测试训练集

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """
            测试两次
            第一次，使用 Kmeans 算法对训练集进行分类，得到正常数据的簇结构（中心点，阈值），簇个数为 4 。
            第二次，使用 Kmeans 算法对测试集进行分类，并根据阈值等特征贴标签，计算准确率。
        """
        if self.ae_trainer is None:
            self.ae_trainer = AEKTrainer(device=device, n_jobs_dataloader=n_jobs_dataloader)
        self.ae_trainer.test(dataset=dataset, ae_net=self.ae_net, flg=0)  # 测试测试集

        self.results['test_auc'] = self.ae_trainer.test_auc
        self.results['test_time'] = self.ae_trainer.test_time
        self.results['test_f_score'] = self.ae_trainer.test_f_score
        self.results['test_mcc'] = self.ae_trainer.test_mcc
        self.results['test_ftr'] = self.ae_trainer.test_ftr
        self.results['test_tpr'] = self.ae_trainer.test_tpr

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
