from base.base_dataset import BaseADDataset
from lstm import Lstm
from bean.deepSVDD import DeepSVDD

from optim.join_trainer import JoinTrainer


class Join(object):
    """
        dldm-join(联合训练)部分的逻辑代码

    """

    def __init__(self, lstm: Lstm, deepsvdd: DeepSVDD, alpha=0.3, n_features=8):
        self.lstm_net = lstm.net
        self.svdd_net = deepsvdd.net

        self.alpha = alpha
        self.n_features = n_features
        self.trainer = None

        self.nu = deepsvdd.nu
        self.R = deepsvdd.R_tensor  # hypersphere radius R
        self.c = deepsvdd.c_tensor  # hypersphere center c
        self.objective = deepsvdd.objective

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

    def train(self, dataset: BaseADDataset,
              lr_1: float = 0.0001,
              lr_2: float = 0.0001,
              lr_milestones_1: tuple = (),
              lr_milestones_2: tuple = (),
              n_epochs: int = 50,
              batch_size: int = 128,
              weight_decay_1: float = 0.0001,
              weight_decay_2: float = 1e-6,
              epsilon=0.0001,
              momentum=0.9,
              device: str = 'cuda',
              n_jobs_dataloader: int = 0):

        """训练联合模型."""
        print('R', self.R)
        print('c', self.c)
        self.trainer = JoinTrainer(objective=self.objective,
                                   R=self.R,
                                   c=self.c,
                                   nu=self.nu,
                                   lr_1=lr_1,
                                   lr_2=lr_2,
                                   lr_milestones_1=lr_milestones_1,
                                   lr_milestones_2=lr_milestones_2,
                                   n_epochs=n_epochs,
                                   batch_size=batch_size,
                                   n_jobs_dataloader=n_jobs_dataloader,
                                   weight_decay_1=weight_decay_1,
                                   weight_decay_2=weight_decay_2,
                                   epsilon=epsilon,
                                   momentum=momentum,
                                   device=device,
                                   n_features=self.n_features,
                                   alpha=self.alpha)
        self.lstm_net, self.svdd_net = self.trainer.train(dataset=dataset, net1=self.lstm_net, net2=self.svdd_net)
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """测试联合模型."""

        print('R', self.R)
        print('c', self.c)
        if self.trainer is None:
            self.trainer = JoinTrainer(self.objective, self.R, self.c, self.nu,
                                       device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset=dataset, net1=self.lstm_net, net2=self.svdd_net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_score'] = self.trainer.test_score
        self.results['test_f_score'] = self.trainer.test_f_score
        self.results['test_mcc'] = self.trainer.test_mcc
        self.results['test_ftr'] = self.trainer.test_ftr
        self.results['test_tpr'] = self.trainer.test_tpr
