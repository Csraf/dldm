import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import filePaths
from utils.math_utils import init_list

""" 绘制折线图工具类 """


def plot_broken_line(x, y, x_name="dos", y_name="dos", labels=[], y_scale=()):
    """ 画折线 """
    colors = ['indianred', 'green', 'darkviolet', 'dodgerblue']
    for i in range(len(labels)):
        plt.plot(x, y[i], marker='.', ms=8, label=labels[i], color=colors[i])
    # plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.ylim(y_scale)
    plt.legend(loc="lower right")
    plt.savefig(filePaths.img_broken_line + x_name + " -- " + y_name)
    plt.show()


def get_results(datas, model_name):
    """ 读取csv文件 """
    aucs = init_list(type=2, s=4)  # [[],[],[],[]]
    fscores = init_list(type=2, s=4)  # [[],[],[],[]]
    mccs = init_list(type=2, s=4)  # [[],[],[],[]]
    times = init_list(type=2, s=4)  # [[],[],[],[]]

    results = [aucs, fscores, mccs, times]
    for i in range(len(results)):
        for j in range(len(model_name)):
            for k in range(len(datas)):
                results[i][j].append(datas[k][model_name[j]][i])

    return aucs, fscores, mccs, times


def plot_broken_line_N():
    """ 针对三种非dos攻击，绘制四个折线图 """
    no_dos_index = ['N1', 'N2', 'N3']
    model_name = ['ae_kmeans', 'rbm_svm', 'dsvdd', 'join']
    urls = [filePaths.N1, filePaths.N2, filePaths.N3]
    labels = ["AE+K-Means", "RBM+SVM", "DSVDD", "DLDM"]
    datas = []
    for i in range(len(no_dos_index)):
        datas.append(pd.read_csv(urls[i]))
    aucs, fscores, mccs, times = get_results(datas=datas, model_name=model_name)

    plot_broken_line(x=no_dos_index, y=aucs, y_name="AUC", x_name="other attack", labels=labels, y_scale=(0.8, 1))
    plot_broken_line(x=no_dos_index, y=times, y_name="time(T)", x_name="other attack", labels=labels, y_scale=(1, 10))
    plot_broken_line(x=no_dos_index, y=fscores, y_name="F1-score", x_name="other attack", labels=labels,
                     y_scale=(0.8, 1))
    plot_broken_line(x=no_dos_index, y=mccs, y_name="MCC", x_name="other attack", labels=labels, y_scale=(0.8, 1))


def plot_broken_line_D():
    """ 针对六种dos攻击，绘制四个折线图 """
    dos_index = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    model_name = ['ae_kmeans', 'rbm_svm', 'dsvdd', 'join']
    urls = [filePaths.D1, filePaths.D2, filePaths.D3, filePaths.D4, filePaths.D5, filePaths.D6]
    labels = ["AE+K-Means", "RBM+SVM", "DSVDD", "DLDM"]
    datas = []
    for i in range(len(dos_index)):
        datas.append(pd.read_csv(urls[i]))

    aucs, fscores, mccs, times = get_results(datas=datas, model_name=model_name)

    plot_broken_line(x=dos_index, y=aucs, x_name="dos", y_name="AUC", labels=labels, y_scale=(0.8, 1))
    plot_broken_line(x=dos_index, y=fscores, y_name="F1-score", x_name="dos", labels=labels, y_scale=(0.8, 1))
    plot_broken_line(x=dos_index, y=mccs, y_name="MCC", x_name="dos", labels=labels, y_scale=(0.8, 1))
    plot_broken_line(x=dos_index, y=times, y_name="time(T)", x_name="dos", labels=labels, y_scale=(1, 10))
