import torch.utils.data.dataset as Dataset
import torch
from base.torchvision_dataset import TorchvisionDataset


class RBMDataset(TorchvisionDataset):
    """
        数据集：
            来自于 rbm(受限玻尔兹曼机) 输出的二进制数据
            该数据集作为 svm 模型的输入

        属性：
            rbm_train：训练集的数据,       shape = (xxxxx, 9)
            rbm_train_label：训练集的标签  shape = (xxxxx,)
            rbm_test：测试集的数据         shape = (xxxxx, 9)
            rbm_test_label：测试集的标签   shape = (xxxxx,)

    """
    def __init__(self, rbm_train, rbm_train_label, rbm_test, rbm_test_label):
        super().__init__()

        self.train_set = RbmKdd99(rbm_train, rbm_train_label)
        self.test_set = RbmKdd99(rbm_test, rbm_test_label)


class RbmKdd99(Dataset.Dataset):

    def __init__(self, Data, Label):
        self.Data = torch.Tensor(Data)
        self.Label = torch.Tensor(Label)

    def __getitem__(self, index):
        return self.Data[index], self.Label[index], index

    def __len__(self):
        return len(self.Data)
