import torch.utils.data.dataset as Dataset
import torch
from datasets.pre_file import pre_file
from datasets.pre_data import load_data_kdd99
from filePaths import src_train
from filePaths import src_test

from filePaths import handle_train
from filePaths import handle_test

from filePaths import final_train
from filePaths import final_test

from base.torchvision_dataset import TorchvisionDataset


class Kdd99_Dataset(TorchvisionDataset):
    """
        数据集：
            来自于 Kdd99 数据集的九个特征
            该数据集作为 lstm-autoencoder 的输入

        属性：
            train：训练集的数据,       shape = (xxxxx, 9)
            train_label：训练集的标签  shape = (xxxxx,)
            test：测试集的数据         shape = (yyyyy, 9)
            test_label：测试集的标签   shape = (yyyyy,)

            test：测试集的类型，kdd99测试集(0), sdn测试集(1)  ---  已废弃
            n_features：特征数目
            dos_type：dos 攻击种类数

            exper_type：代表实验类型
                0：基础实验（join，ae_kmeans），训练集获取正常数据，测试集获取所有数据
                1：对比实验（rbm）：训练集获取所有数据，测试集获取所有数据
                2：对比实验（join，ae_kmeans，dos_types）：训练集获取正常数据，测试集获取正常数据 + 指定攻击
                3：对比实验（rbm，dos_types）：训练集获取所有数据，测试集获取正常数据 + 指定攻击

    """

    def __init__(self, n_features=8, exper_type=0, dos_types=0):
        """  """
        super().__init__()
        self.n_features = n_features
        self.exper_type = exper_type
        self.dos_types = dos_types
        self.train = None
        self.test = None
        self.train_labels = None
        self.test_labels = None

        pre_file(src_train, handle_train, train=1, exper_type=self.exper_type, dos_types=self.dos_types)
        pre_file(src_test, handle_test, train=0, exper_type=self.exper_type, dos_types=self.dos_types)

        train, train_label = load_data_kdd99(handle_train, final_train, self.n_features)

        test, test_label = load_data_kdd99(handle_test, final_test, self.n_features)  # kdd99 测试集

        self.train = train
        self.test = test
        self.train_labels = train_label
        self.test_labels = test_label

        self.train_set = Kdd99(train, train_label)
        self.test_set = Kdd99(test, test_label)

        print("train", train.shape)
        print("train_label", train_label.shape)
        print("test", test.shape)
        print("test_label", test_label.shape)

        print(self.train_set.__getitem__(0))
        print(self.test_set.__getitem__(0))

    def update_test(self, exper_type=0, dos_types=0):
        """ 多次 dos 攻击，更新测试集 """
        pre_file(src_test, handle_test, 0, exper_type=exper_type, dos_types=dos_types)
        test, test_label = load_data_kdd99(handle_test, final_test, self.n_features)  # kdd99 测试集
        self.test_set = Kdd99(test, test_label)
        self.test = test
        self.test_labels = test_label

        print("test", test.shape)
        print("test_label", test_label.shape)


class Kdd99(Dataset.Dataset):

    def __init__(self, Data, Label):
        self.Data = torch.Tensor(Data)
        self.Label = torch.Tensor(Label)

    def __getitem__(self, index):
        return self.Data[index], self.Label[index], index

    def __len__(self):
        return len(self.Data)
