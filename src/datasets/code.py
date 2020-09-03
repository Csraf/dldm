from base.torchvision_dataset import TorchvisionDataset
from lstm import Lstm

import numpy as np
import torch
import torch.utils.data.dataset as Dataset


class Code_Dataset(TorchvisionDataset):
    """
        数据集：
            来自于 lstm-autoencoder 输出的 code
            该数据集作为 svdd-autoencoder 和 deep-svdd的输入

        属性：
            train_code：训练集的数据  shape = (97278, 8)
            train_label：训练集的标签 shape = (97278,)
            test_code：测试集的数据   shape = (283913, 8)
            test_label：测试集的标签  shape = (283913,)
    """

    def __init__(self, lstm: Lstm):
        super().__init__()

        self.n_classes = 2

        train_code, train_label, test_code, test_label = lstm.load_code()

        self.train_set = Code(np.array(train_code), np.array(train_label))
        self.test_set = Code(np.array(test_code), np.array(test_label))

        print("train_code", np.array(train_code).shape)
        print("train_label", np.array(train_label).shape)
        print("test_code", np.array(test_code).shape)
        print("test_label", np.array(test_label).shape)

        print('train_size', len(self.train_set))
        print(self.train_set.__getitem__(0))
        # print(roc_self.train_set.__getitem__(2000))
        # print(roc_self.train_set.__getitem__(7000))

        print('test_size', len(self.test_set))
        print(self.test_set.__getitem__(0))
        # print(roc_self.test_set.__getitem__(2000))
#        print(roc_self.test_set.__getitem__(7000))


class Code(Dataset.Dataset):

    def __init__(self, Data, Label):
        self.Data = torch.Tensor(Data)
        self.Label = torch.Tensor(Label)

    def __getitem__(self, index):
        return self.Data[index], self.Label[index], index

    def __len__(self):
        return len(self.Data)




