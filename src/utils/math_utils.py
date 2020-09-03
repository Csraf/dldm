import numpy as np
from filePaths import self_join_fpr
from filePaths import self_join_tpr
from filePaths import self_no_join_fpr
from filePaths import self_no_join_tpr

from filePaths import ae_kmeans_fpr
from filePaths import ae_kmeans_tpr
from filePaths import rbm_fpr
from filePaths import rbm_tpr
from filePaths import rbm_one_fpr
from filePaths import rbm_one_tpr
from filePaths import dsvdd_fpr
from filePaths import dsvdd_tpr
from filePaths import join_fpr
from filePaths import join_tpr

from utils.roc_utils import roc_mean


def init_list(s=4, type=0):
    """ 初始化 list """
    if type == 0:
        return [0.0 for i in range(s)]
    elif type == 1:
        return [None for i in range(s)]
    elif type == 2:
        return [[] for i in range(s)]
    return list


def add_list(s=4, datas=None, models=None, key='test_auc'):
    """ 添加 list """
    for i in range(s):
        if key == 'test_ftr' or key == 'test_tpr':
            datas[i].append(models[i].results[key])
        else:
            datas[i] += models[i].results[key]
    return datas


def print_list(s=4, datas=None, model_name=[], label_name=''):
    """ 打印 list """
    for i in range(s):
        print(model_name[i], label_name, datas[i])


def get_mean_fpr_tpr(s, fprs, tprs):
    """ 获取fprs, tprs平均值 """
    for i in range(s):
        fprs[i], tprs[i] = roc_mean(fpr=fprs[i], tpr=tprs[i])
    return fprs, tprs


def save_fpr_tpr(fprs, tprs, type=0):
    """
        保存中间结果（fprs tprs）为 npy 格式文件
            type = 0：自对比实验
            type = 1：对比实验

    """
    if type == 0:
        np.save(self_no_join_fpr, np.array(fprs[0]))
        np.save(self_no_join_tpr, np.array(tprs[0]))
        np.save(self_join_fpr, np.array(fprs[1]))
        np.save(self_join_tpr, np.array(tprs[1]))
    elif type == 1:
        np.save(ae_kmeans_fpr, np.array(fprs[0]))
        np.save(ae_kmeans_tpr, np.array(tprs[0]))
        np.save(rbm_fpr, np.array(fprs[1]))
        np.save(rbm_tpr, np.array(tprs[1]))
        np.save(rbm_one_fpr, np.array(fprs[2]))
        np.save(rbm_one_tpr, np.array(tprs[2]))
        np.save(dsvdd_fpr, np.array(fprs[3]))
        np.save(dsvdd_tpr, np.array(tprs[3]))
        np.save(join_fpr, np.array(fprs[4]))
        np.save(join_tpr, np.array(tprs[4]))


def load_fpr_tpr(type=0):
    """
        加载中间结果（fpr，tpr）
            type = 0：自对比实验
            type = 1：对比实验

    """
    fprs = []
    tprs = []
    if type == 0:
        fprs.append(np.load(self_no_join_fpr + '.npy'))
        fprs.append(np.load(self_join_fpr + '.npy'))
        tprs.append(np.load(self_no_join_tpr + '.npy'))
        tprs.append(np.load(self_join_tpr + '.npy'))
    elif type == 1:
        fpr_path = [ae_kmeans_fpr, rbm_fpr, rbm_one_fpr, dsvdd_fpr, join_fpr]
        tpr_path = [ae_kmeans_tpr, rbm_tpr, rbm_one_tpr, dsvdd_tpr, join_tpr]
        for i in range(len(fpr_path)):
            fprs.append(np.load(fpr_path[i] + '.npy'))

        for i in range(len(tpr_path)):
            tprs.append(np.load(tpr_path[i] + '.npy'))
    return fprs, tprs
