import numpy as np
import time
from optim.rbm_trainer import RBMTrainer


class RBM(object):
    """
        对比实验： rbm + svm
    """

    def __init__(self, n_visible=9, n_hidden=9, momentum=0.5, learning_rate=0.1, max_epoch=50,
                 batch_size=128, penalty=0, weight=None, v_bias=None, h_bias=None):

        self.rbm_trainer = None
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.momentum = momentum
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.penalty = penalty

        self.rbm_train = []
        self.rbm_test = []
        self.rbm_train_label = []
        self.rbm_test_label = []

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def get_rbm(self, datas):
        """ 获取rbm的重构数据 """
        res = []
        for i in range(datas.shape[0]):
            data = datas[i].reshape(1, -1)
            res.append(self.rbm_trainer.predict(data))
        res = np.array(res)
        return res

    def train(self, train_data):
        """ 训练 rbm 模型 """
        self.rbm_trainer = RBMTrainer(n_visible=self.n_visible, n_hidden=self.n_hidden,
                                      max_epoch=self.max_epoch, learning_rate=self.learning_rate)
        self.rbm_trainer.fit(train_data)
        self.rbm_train = self.rbm_trainer.predict(train_data)

    def test(self, train_data, train_label, test_data, test_label):
        """ 测试 rbm 模型 -- 获取测试时间 """
        self.rbm_train_label = train_label
        self.rbm_test_label = test_label
        start = time.time()
        self.rbm_test = self.rbm_trainer.predict(test_data)
        self.results['test_time'] = time.time() - start
        return self.rbm_train, self.rbm_train_label, self.rbm_test, self.rbm_test_label
