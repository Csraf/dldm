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

""" 
    对象管理器：管理算法对象，执行训练测试等任务。

        _manager：表示包含训练和测试逻辑
        _train：仅包含训练逻辑
        _test：仅包含测试逻辑
"""


class Manager(object):
    def __init__(self, device='cpu'):
        self.device = device

    def lstm_manager(self, dataset: BaseADDataset, n_features=9, n_epoch=10):
        """ lstm 训练 + 测试逻辑 """
        lstm = Lstm(n_features=n_features)
        lstm.set_network('LstmNet')
        lstm.train(dataset=dataset, device=self.device, optimizer_name='RMSprop', n_epochs=n_epoch)
        lstm.test(dataset=dataset, device=self.device)
        return lstm

    def lstm_svdd_manager(self, dataset: BaseADDataset, lstm: Lstm, pre_epoch=10, n_epochs=4):
        """ svdd 训练 + 测试逻辑 """
        code_dataset = Code_Dataset(lstm)
        svdd = DeepSVDD(lstm=lstm, objective='one-class', n_code=8)
        svdd.set_network('SvddNet')
        svdd.pretrain(dataset=code_dataset, device=self.device, n_epochs=pre_epoch)
        svdd.train(dataset=code_dataset, device=self.device, n_epochs=n_epochs)
        svdd.test(dataset=dataset, device=self.device)
        return svdd

    def join_manager(self, dataset: BaseADDataset, lstm: Lstm, svdd: DeepSVDD, alpha=0.15, n_features=9, n_epochs=4):
        """ dldm 训练 + 测试逻辑 """
        join = Join(lstm=lstm, deepsvdd=svdd, alpha=alpha, n_features=n_features)
        join.train(dataset=dataset, device=self.device, n_epochs=n_epochs)
        join.test(dataset=dataset, device=self.device)
        return join

    def ae_kmeans_manager(self, dataset: BaseADDataset, n_epochs=30):
        """ ae_kmeans 训练 + 测试逻辑 """
        aek = AEK()
        aek.set_network('KddNet')
        aek.train(dataset=dataset, device=self.device, n_epochs=n_epochs)
        return aek

    def rbm_svm_manager(self, n_features=9, rbm_epochs=35, svm_epochs=20, lr=0.05, rbm_data=[]):
        """ rbm_svm 训练 + 测试逻辑 """
        rbm = RBM(n_visible=n_features, n_hidden=n_features, max_epoch=rbm_epochs, learning_rate=lr)
        rbm.train(rbm_data[0])
        svm_train, svm_train_label, svm_test, svm_test_label = rbm.test(rbm_data[0], rbm_data[1], rbm_data[2],
                                                                        rbm_data[3])
        svm_dataset = RBMDataset(svm_train, svm_train_label, svm_test, svm_test_label)
        print("svm_train", svm_train.shape)
        print("svm_test", svm_test.shape)
        svm = SVM()
        svm.set_network('SVMNet', n_features=n_features)
        svm.train(dataset=svm_dataset, n_epochs=svm_epochs)
        return rbm, svm

    def dsvdd_manager(self, dataset: BaseADDataset, pre_epoch=10, n_epochs=4):
        """ dsvdd 训练 + 测试逻辑 """
        dsvdd = SDeepSVDD(objective='one-class')
        dsvdd.set_network('KddNet')
        dsvdd.pretrain(dataset=dataset, device=self.device, n_epochs=pre_epoch)
        dsvdd.train(dataset=dataset, device=self.device, n_epochs=n_epochs)
        dsvdd.test(dataset=dataset, device=self.device)
        return dsvdd

    def train_aek(self, dataset: BaseADDataset, n_epochs=30):
        """ aek 训练逻辑 """
        aek = AEK()
        aek.set_network('KddNet')
        aek.train(dataset=dataset, device=self.device, n_epochs=n_epochs)
        return aek

    def train_rbm_svm(self, rbm_data=[], n_features=9, rbm_epochs=35, svm_epochs=20, lr=0.05):
        """ rbm_svm 训练逻辑 """
        rbm = RBM(n_visible=n_features, n_hidden=n_features, max_epoch=rbm_epochs, learning_rate=lr)
        rbm.train(rbm_data[0])
        svm_train, svm_train_label, svm_test, svm_test_label = rbm.test(rbm_data[0], rbm_data[1], rbm_data[2],
                                                                        rbm_data[3])
        svm_dataset = RBMDataset(svm_train, svm_train_label, svm_test, svm_test_label)
        svm = SVM()
        svm.set_network('SVMNet', n_features=n_features)
        svm.train(dataset=svm_dataset, n_epochs=svm_epochs)
        return rbm, svm

    def train_dsvdd(self, dataset: BaseADDataset, pre_epoch=10, n_epochs=4):
        """ dsvdd 训练逻辑 """
        dsvdd = SDeepSVDD(objective='one-class')
        dsvdd.set_network('KddNet')
        dsvdd.pretrain(dataset=dataset, device=self.device, n_epochs=pre_epoch)
        dsvdd.train(dataset=dataset, device=self.device, n_epochs=n_epochs)
        return dsvdd

    def train_dldm(self, dataset: BaseADDataset, n_features=9, lstm_epoch=10, n_code=8, svdd_pre_epoch=10,
                   svdd_epochs=20,
                   join_alpha=0.1, join_epochs=4):
        """ dldm 训练逻辑 """
        lstm = Lstm(n_features=n_features)
        lstm.set_network('LstmNet')
        lstm.train(dataset=dataset, device=self.device, optimizer_name='RMSprop', n_epochs=lstm_epoch)
        lstm.test(dataset=dataset, device=self.device)
        code_dataset = Code_Dataset(lstm)
        svdd = DeepSVDD(lstm=lstm, objective='one-class', n_code=8)
        svdd.set_network('SvddNet')
        svdd.pretrain(dataset=code_dataset, device=self.device, n_epochs=svdd_pre_epoch)
        svdd.train(dataset=code_dataset, device=self.device, n_epochs=svdd_epochs)

        join = Join(lstm=lstm, deepsvdd=svdd, alpha=join_alpha, n_features=n_features)
        join.train(dataset=dataset, device=self.device, n_epochs=join_epochs)
        return lstm, svdd, join

    def test_rbm_svm(self, rbm_data, rbm, svm):
        """ 测试 rbm svm, 返回模型对象 """
        svm_train, svm_train_label, svm_test, svm_test_label = rbm.test(rbm_data[0], rbm_data[1], rbm_data[2],
                                                                        rbm_data[3])
        svm_dataset = RBMDataset(svm_train, svm_train_label, svm_test, svm_test_label)
        svm.test(svm_dataset)
        svm.results['test_time'] = rbm.results['test_time'] + svm.results['test_time']
        return svm

    def test_aek(self, dataset: BaseADDataset, aek):
        """ 测试 ae_kmeans, 返回模型对象 """
        aek.test(dataset=dataset, device=self.device)
        return aek

    def test_dsvdd(self, dataset: BaseADDataset, dsvdd):
        """ 测试 dsvdd, 返回模型对象 """
        dsvdd.test(dataset=dataset, device=self.device)
        return dsvdd

    def test_dldm(self, dataset: BaseADDataset, dldm):
        """ 测试 dldm, 返回模型对象 """
        dldm.test(dataset=dataset, device=self.device)
        return dldm
