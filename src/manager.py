from base_dataset import BaseADDataset
from datasets.code import Code_Dataset
from datasets.rbmKdd99 import RBMDataset

from bean.svdd import SDeepSVDD
from bean.deepSVDD import DeepSVDD
from bean.lstm import Lstm
from bean.join import Join
from bean.aek import AEK
from bean.rbm import RBM
from bean.svm import SVM


class Manager(object):
    """  对象管理器：管理算法对象，执行训练测试等任务。"""
    def __init__(self, device='cpu'):
        self.device = device

    """ 自对比实验 """

    def lstm_manager(self, dataset: BaseADDataset, n_features=9, n_epoch=10):
        """ lstm """
        lstm = Lstm(n_features=n_features)
        lstm.set_network('LstmNet')
        lstm.train(dataset=dataset, device=self.device, optimizer_name='RMSprop', n_epochs=n_epoch)
        lstm.test(dataset=dataset, device=self.device)
        return lstm

    def lstm_svdd_manager(self, dataset: BaseADDataset, lstm: Lstm, pre_epoch=10, n_epochs=4):
        """ svdd """
        code_dataset = Code_Dataset(lstm)
        svdd = DeepSVDD(lstm=lstm, objective='one-class', n_code=8)
        svdd.set_network('SvddNet')
        svdd.pretrain(dataset=code_dataset, device=self.device, n_epochs=pre_epoch)
        svdd.train(dataset=code_dataset, device=self.device, n_epochs=n_epochs)
        svdd.test(dataset=dataset, device=self.device)
        return svdd

    def join_manager(self, dataset: BaseADDataset, lstm: Lstm, svdd: DeepSVDD, alpha=0.15, n_features=9, n_epochs=4):
        """ dldm """
        join = Join(lstm=lstm, deepsvdd=svdd, alpha=alpha, n_features=n_features)
        join.train(dataset=dataset, device=self.device, n_epochs=n_epochs)
        join.test(dataset=dataset, device=self.device)
        return join

    """ 对比实验 """

    def ae_kmeans_manager(self, dataset: BaseADDataset, n_epochs=30):
        """ join """
        aek = AEK()
        aek.set_network('KddNet')
        aek.pretrain(dataset=dataset, device=self.device, n_epochs=n_epochs)
        return aek

    def rbm_svm_manager(self, n_features=9, rbm_epochs=35, svm_epochs=20, lr=0.05, rbm_data=[]):
        rbm = RBM(n_visible=n_features, n_hidden=n_features, max_epoch=rbm_epochs, learning_rate=lr)
        svm_train, svm_train_label, svm_test, svm_test_label = rbm.compute(rbm_data[0], rbm_data[1], rbm_data[2],
                                                                           rbm_data[3])
        svm_dataset = RBMDataset(svm_train, svm_train_label, svm_test, svm_test_label)
        print("svm_train", svm_train.shape)
        print("svm_test", svm_test.shape)
        svm = SVM()
        svm.set_network('SVMNet', n_features=n_features)
        svm.train(dataset=svm_dataset, n_epochs=svm_epochs)
        return rbm, svm

    def dsvdd_manager(self, dataset: BaseADDataset, pre_epoch=10, n_epochs=4):
        dsvdd = SDeepSVDD(objective='one-class')
        dsvdd.set_network('KddNet')
        dsvdd.pretrain(dataset=dataset, device=self.device, n_epochs=pre_epoch)
        dsvdd.train(dataset=dataset, device=self.device, n_epochs=n_epochs)
        dsvdd.test(dataset=dataset, device=self.device)
        return dsvdd
