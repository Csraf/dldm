from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.cluster import KMeans

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class AEKTrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)
        self.nu = 0.15
        self.clusters = 4
        self.center = None  # 每一簇对应的中心
        self.radius = None  # 每一簇对应的半径
        self.kmeans = None  # 正常数据训练好的 Kmeans 分类器

        self.test_auc = None
        self.test_time = None
        self.test_f_score = None
        self.test_mcc = None
        self.test_ftr = None
        self.test_tpr = None

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for networks
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting ae ...')
        start_time = time.time()
        ae_net.train()
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

                # Update networks parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        logger.info('training time: %.3f' % pretrain_time)
        logger.info('Finished training.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet, flg=0):
        """
            训练集 获取正常数据簇 -- 中心点，半径
            测试集 Kmeans 对数据进行预测，超过簇半径为异常数据，否则正常数据
        """

        logger = logging.getLogger()

        # Set device for networks
        ae_net = ae_net.to(self.device)

        # 训练集 flg==1  测试集 flg==0
        if flg == 1:
            test_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        else:
            _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Testing ae...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                error = (outputs - inputs) ** 2
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist(),
                                            error.cpu().data.numpy().tolist()),
                                        )

                loss_epoch += loss.item()
                n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        _, labels, scores, error = zip(*idx_label_score)
        labels = np.array(labels)  # labels.shape(97278, )
        scores = np.array(scores)  # scores.shape(97278, )
        error = np.array(error)  # scores.shape(97278, )

        if flg == 1:  # 训练集
            X = error
            self.kmeans = KMeans(n_clusters=self.clusters).fit(X)
            self.center = self.kmeans.cluster_centers_.tolist()
            self.radius = self.get_radius(X)
            print("roc_self.center", self.center)
            print("roc_self.radius", self.radius)
        else:  # 测试集
            Y = error
            pred_labels = []  # 实际标签
            pred_km = self.kmeans.predict(Y)
            print(pred_km.shape)
            print(pred_km)
            for i in range(len(pred_km)):
                dis = self.manhattan_distance(self.center[pred_km[i]], Y[i])  # dis：簇中心到点的距离，作为分类依据
                if dis > self.radius[pred_km[i]]:
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)

            pred_labels = np.array(pred_labels)
            self.test_ftr, self.test_tpr, _ = roc_curve(labels, pred_labels)
            # roc_self.test_auc = roc_auc_score(pred_labels, labels)
            fpr, tpr, thresholds = roc_curve(labels, pred_labels)  # 面积作为准确率
            print(fpr, tpr)
            self.test_auc = auc(fpr, tpr)
            self.test_mcc = matthews_corrcoef(labels, pred_labels)
            _, _, f_score, _ = precision_recall_fscore_support(labels, pred_labels, labels=[0, 1])
            self.test_f_score = f_score[1]

        print(len(labels))
        print(len(scores))

        self.test_time = time.time() - start_time
        if flg == 0:
            logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        logger.info('ae testing time: %.3f' % self.test_time)
        logger.info('Finished testing ae.')

    def get_radius(self, x):
        """
            获取簇半径,当前簇内所有点到中心点的最大距离
        """
        radius = []
        for i in range(self.clusters):
            cls = x[self.kmeans.labels_ == i]  # 某一类簇
            print(cls.shape)  # (xx, )
            dis = []
            for k in range(cls.shape[0]):
                dis.append(self.manhattan_distance(self.center[i], cls[k]))
            radius.append(np.quantile(dis, 1 - self.nu))
        return radius

    def manhattan_distance(self, x, y):
        """ 曼哈顿距离 """
        return np.sum(abs(x - y))
