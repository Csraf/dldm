from JoinNet import JoinNet
from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve

import torch
import torch.optim as optim
import logging
import time
import numpy as np


class JoinTrainer(BaseTrainer):
    """
    联合训练 具体实现
        net1 ： lstmNet
        net2 ： svddNet

        优化器选择 Adam

    """

    def __init__(self, objective, R, c, nu: float,
                 lr_1: float = 0.0001,
                 lr_2: float = 0.0001,
                 lr_milestones_1: tuple = (),
                 lr_milestones_2: tuple = (),
                 n_epochs: int = 50,
                 batch_size: int = 128,
                 weight_decay_1: float = 1e-6,
                 weight_decay_2: float = 1e-6,
                 epsilon=0.0001,
                 momentum=0.9,
                 device: str = 'cuda',
                 n_jobs_dataloader: int = 0,
                 n_features=8,
                 alpha=0.3):
        super().__init__(n_epochs=n_epochs, batch_size=batch_size, device=device, n_jobs_dataloader=n_jobs_dataloader,
                         lr=lr_2, lr_milestones=lr_milestones_2, weight_decay=weight_decay_1, optimizer_name='Adam')

        # join parameters
        self.alpha = alpha
        self.n_features = n_features

        # optim parameters
        self.lr_1 = lr_1
        self.lr_2 = lr_2
        self.lr_milestones_1 = lr_milestones_1
        self.lr_milestones_2 = lr_milestones_2

        self.weight_decay_1 = weight_decay_1
        self.weight_decay_2 = weight_decay_2

        # Deep SVDD parameters
        self.R = R  # radius R initialized with 0 by default.
        self.c = c if c is not None else None
        self.nu = nu
        self.objective = objective

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated
        self.epsilon = epsilon
        self.momentum = momentum

        # join time
        self.test_auc = None
        self.test_time = None
        self.test_score = None
        self.test_f_score = None
        self.test_mcc = None
        self.test_ftr = None
        self.test_tpr = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        """
            net1：lstmNet
            net2：svddNet
        """
        logger = logging.getLogger()

        # Set device for networks
        join_net =net.to(self.device)
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        optimizer = optim.Adam(join_net.parameters(), lr=self.lr_2, weight_decay=self.weight_decay_2)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones_2, gamma=0.1)

        # Training
        logger.info('Starting train join ...')
        start_time = time.time()
        join_net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:  # 这里加载的是kdd99
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the networks parameter gradients
                optimizer.zero_grad()

                # join train two network
                lstm_out, svdd_out = join_net(inputs.view(-1, 1, self.n_features))

                dist = torch.sum((svdd_out - self.c) ** 2, dim=1)

                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    svdd_loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    svdd_loss = torch.mean(dist)

                lstm_loss = torch.mean(torch.sum((lstm_out - inputs) ** 2, dim=tuple(range(1, lstm_out.dim()))))
                loss = svdd_loss + self.alpha * lstm_loss

                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

                # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)
        logger.info('Finished training.')

        return join_net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for networks
        join_net = net.to(self.device)
        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        join_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)

                # Update networks parameters via back propagation: forward + backward + optimize
                _, svdd_out = join_net(inputs.view(-1, 1, self.n_features))

                dist = torch.sum((svdd_out - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        # compute auc
        self.test_auc = roc_auc_score(labels, scores)
        self.test_score = scores

        self.test_ftr, self.test_tpr, _ = roc_curve(labels, scores)

        optimal_threshold, _ = find_optimal_cutoff(labels, scores)
        pred_labels = np.array(list(map(lambda x: 0.0 if x <= optimal_threshold else 1.0, scores)))

        # get f-score mcc
        _, _, f_score, _ = precision_recall_fscore_support(labels, pred_labels, labels=[0, 1])
        self.test_f_score = f_score[1]
        self.test_mcc = matthews_corrcoef(labels, pred_labels)

        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Finished testing.')


def find_optimal_cutoff(label, y_prob):
    """ 寻找最优阀值 - - 阿登指数  """
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    y = tpr - fpr
    youden_index = np.argmax(y)
    optimal_threshold = thresholds[youden_index]
    point = [fpr[youden_index], tpr[youden_index]]
    return optimal_threshold, point


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
