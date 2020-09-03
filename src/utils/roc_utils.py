import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn import metrics


def plot_roc(labels, pred, roc_auc):
    """ 绘制 roc 曲线 """
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(labels, pred)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()


def plot_dict_rocs(labels, scores, aucs, names, colors):
    """ 绘制多条 roc 曲线  --  dict 类型"""

    k = 0
    # 显示不同颜色，不同名字
    plt.figure(0).clf()
    for i, j in zip(scores, aucs):
        pred = scores[i]
        auc = aucs[j]
        fpr, tpr, ts = metrics.roc_curve(labels, pred)
        plt.title('ROC')
        plt.plot(fpr, tpr, 'b', color=colors[k], label=names[k] + ' = %0.4f' % auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        k += 1
    plt.show()


def plot_list_rocs(labels, scores, aucs, names, colors):
    """ 绘制多条 roc 曲线 -- list 类型 """
    k = 0
    # 显示不同颜色，不同名字
    plt.figure(0).clf()
    for i in range(len(aucs)):
        pred = scores[i]
        auc = aucs[i]
        fpr, tpr, ts = metrics.roc_curve(labels, pred)
        plt.title('ROC')
        plt.plot(fpr, tpr, 'b', color=colors[k], label=names[k] + ' = %0.4f' % auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        k += 1
    plt.show()


def roc_mean(fpr=[], tpr=[]):
    """ 平均 roc 曲线 """
    n = len(fpr)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n
    return all_fpr, mean_tpr


def plot_mean_rocs(img_name=None, img_path=None, fprs=[], tprs=[], aucs=[], names=[], colors=[]):
    """ 绘制多条平均 roc 曲线 """
    k = 0
    # 显示不同颜色，不同名字
    plt.figure(0).clf()
    for i in range(len(aucs)):
        auc = aucs[i]
        fpr = fprs[i].tolist()
        tpr = tprs[i].tolist()
        fpr.insert(0, 0)
        tpr.insert(0, 0)
        fpr = np.array(fpr)
        tpr = np.array(tpr)
        plt.title('ROC')
        plt.plot(fpr, tpr, 'b', color=colors[k], label=names[k] + ' = %0.4f' % auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        k += 1
    plt.savefig(img_path + img_name)
    plt.show()

