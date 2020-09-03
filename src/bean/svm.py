from base.base_dataset import BaseADDataset
from networks.main import build_network
from optim.svm_trainer import SVMTrainer


class SVM(object):
    """
        对比实验：rbm + svm

    """

    def __init__(self):
        self.net_name = None
        self.net = None  # neural networks \phi

        self.svm_trainer = None

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

    def set_network(self, net_name, n_features=9):
        """Builds the neural networks ."""

        self.net_name = net_name
        self.net = build_network(net_name, n_features=9)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'SGD', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cpu',
              n_jobs_dataloader: int = 0):
        """Trains the Lstm model on the training data."""

        self.trainer = SVMTrainer(optimizer_name=optimizer_name,
                                  lr=lr,
                                  n_epochs=n_epochs,
                                  lr_milestones=lr_milestones,
                                  batch_size=batch_size,
                                  weight_decay=weight_decay,
                                  device=device,
                                  n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset=dataset, svm_net=self.net)
        self.trainer.test(dataset=dataset, svm_net=self.net)
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_score'] = self.trainer.test_score
        self.results['test_f_score'] = self.trainer.test_f_score
        self.results['test_mcc'] = self.trainer.test_mcc
        self.results['test_ftr'] = self.trainer.test_ftr
        self.results['test_tpr'] = self.trainer.test_tpr
