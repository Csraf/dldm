from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve, precision_recall_fscore_support
import time
import torch
import torch.optim as optim
import logging
import numpy as np


class SVMTrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'SGD', lr: float = 1e-3, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cpu', n_jobs_dataloader: int = 8):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        self.device = device
        self.n_epochs = n_epochs
        self.momentum = 0.9
        self.step_size = 7
        self.gamma = 0.1
        self.lr = lr

        self.test_auc = None
        self.test_time = None
        self.test_score = None
        self.test_f_score = None
        self.test_mcc = None
        self.test_ftr = None
        self.test_tpr = None

    def hinge_loss(self, outputs, labels):
        """
        折页损失计算
        :param outputs: 大小为(N, num_classes)
        :param labels: 大小为(N)
        :return: 损失值
        """
        num_labels = len(labels)
        corrects = outputs[range(num_labels), labels.long()].unsqueeze(0).t().float()

        # 最大间隔
        margin = 1.0
        margins = outputs - corrects + margin
        loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

        # # 正则化强度
        # reg = 1e-3
        # loss += reg * torch.sum(weight ** 2)

        return loss

    def train(self, dataset: BaseADDataset, svm_net: BaseNet):
        """ 训练 svm 模型 """
        logger = logging.getLogger()

        # Set device for networks
        svm_net = svm_net.to(self.device)

        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        optimizer = optim.SGD(svm_net.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        # Training
        logger.info('Starting train svm_trainer ...')
        start_time = time.time()
        svm_net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, labels, _ = data
                inputs = inputs.to(self.device)

                # Zero the networks parameter gradients
                optimizer.zero_grad()

                # Update networks parameters via back propagation: forward + backward + optimize
                outputs = svm_net(inputs)

                # get loss
                loss = self.hinge_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        logger.info('svm_trainer train time: %.3f' % pretrain_time)
        logger.info('Finished train svm_trainer.')

        return svm_net

    def test(self, dataset: BaseADDataset, svm_net: BaseNet):
        """ 测试 svm 模型 """
        logger = logging.getLogger()

        # Set device for networks
        svm_net = svm_net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        svm_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = svm_net(inputs)
                _, scores = torch.max(outputs, 1)
                loss = self.hinge_loss(outputs, labels)

                scores = scores.float()
                labels = labels.float()
                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_score = scores

        print(len(labels))
        print(len(scores))
        self.test_auc = roc_auc_score(labels, scores)
        self.test_ftr, self.test_tpr, _ = roc_curve(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        self.test_mcc = matthews_corrcoef(labels, scores)
        _, _, f_score, _ = precision_recall_fscore_support(labels, scores, labels=[0, 1])
        self.test_f_score = f_score[1]
        self.test_time = time.time() - start_time
        logger.info('svm_trainer testing time: %.3f' % self.test_time)
        logger.info('Finished testing svm_trainer.')
