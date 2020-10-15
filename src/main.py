import numpy as np
import pandas as pd
import filePaths
from datasets.kdd99 import Kdd99_Dataset
from manager import Manager
from utils.roc_utils import plot_mean_rocs, roc_mean
from utils.math_utils import init_list, add_list, print_list, init_array2, mean_array2
from utils.math_utils import get_mean_fpr_tpr, save_fpr_tpr, load_fpr_tpr
from utils.log_utils import init_device, init_logger
from utils.line_chart_utils import plot_broken_line_D, plot_broken_line_N


def self_compare_exper(device: str = 'cuda', n_features=9, alpha=0.1, n=1, s=2):
    """
        自对比实验
            对比对象：lstm_svdd， lstm_svdd_join
            对比指标：roc曲线，roc_auc，test_time

        属性：
            s :实验中对比算法的数量
            n：实验迭代次数
    """
    """ 初始化列表 """
    times = init_list(s=s, type=0)  # 0.0
    aucs = init_list(s=s, type=0)  # 0.0
    fprs = init_list(s=s, type=2)  # [[],[]]
    tprs = init_list(s=s, type=2)  # [[],[]]

    kdd99_dataset = Kdd99_Dataset(n_features=n_features, exper_type=0, dos_types=0)

    for i in range(n):
        print("---------- 第", i + 1, "轮迭代 ----------")
        """ 训练 + 测试模型 """
        manager = Manager(device)
        lstm = manager.lstm_manager(dataset=kdd99_dataset, n_features=n_features, n_epoch=10)
        svdd = manager.lstm_svdd_manager(dataset=kdd99_dataset, lstm=lstm, pre_epoch=10, n_epochs=4)
        join = manager.join_manager(dataset=kdd99_dataset, lstm=lstm, svdd=svdd, alpha=alpha, n_features=n_features,
                                    n_epochs=4)

        models = [svdd, join]
        aucs = add_list(s=s, datas=aucs, models=models, key='test_auc')
        times = add_list(s=s, datas=times, models=models, key='test_time')
        fprs = add_list(s=s, datas=fprs, models=models, key='test_ftr')
        tprs = add_list(s=s, datas=tprs, models=models, key='test_tpr')

    times = [times[i] / n for i in range(s)]
    aucs = [aucs[i] / n for i in range(s)]
    fprs[0], tprs[0] = roc_mean(fpr=fprs[0], tpr=tprs[0])
    fprs[1], tprs[1] = roc_mean(fpr=fprs[1], tpr=tprs[1])

    model_name = ['dsvdd', 'join']
    print_list(s=s, model_name=model_name, label_name='aucs', datas=aucs)
    print_list(s=s, model_name=model_name, label_name='times', datas=times)

    plot_mean_rocs(img_name='s1', img_path=filePaths.img_roc_self, fprs=fprs, tprs=tprs, aucs=aucs,
                   names=['non joint training', 'joint training'], colors=['green', 'dodgerblue'])

    """ 保存文件 """
    save_fpr_tpr(fprs, tprs, type=0)


def compare_exper_dos_types(device: str = 'cuda', n_features=9, alpha=0.1, n=1, s=4):
    """
        对比实验 -- 多种 dos 攻击
            对比对象：ae_kmeans， rbm_svm， dsvdd， join(dldm)
            对比指标：roc曲线，roc_auc，f-score，mcc，test_time

        属性：
            s :实验中对比算法的数量
            n：实验迭代次数

        注意：不再绘制每种攻击的roc曲线
    """

    m = 9  # 攻击种类数
    kdd99_dataset = Kdd99_Dataset(n_features=n_features, exper_type=2, dos_types=1)
    rbm_dataset = Kdd99_Dataset(n_features=n_features, exper_type=3, dos_types=1)
    rbm_data = [rbm_dataset.train, rbm_dataset.train_labels, rbm_dataset.test, rbm_dataset.test_labels]
    urls = [filePaths.D1, filePaths.D2, filePaths.D3, filePaths.D4, filePaths.D5, filePaths.D6,
            filePaths.N1, filePaths.N2, filePaths.N3]

    aucs = init_array2(s=s, m=m, type=0)  # 4*9 0
    times = init_array2(s=s, m=m, type=0)
    mccs = init_array2(s=s, m=m, type=0)
    f_scores = init_array2(s=s, m=m, type=0)

    # fprs = init_array2(s=s, m=m, type=0)  # 4*9 []
    # tprs = init_array2(s=s, m=m, type=0)

    for i in range(n):
        """ 训练模型 """
        manager = Manager(device)
        aek = manager.train_aek(dataset=kdd99_dataset, n_epochs=30)
        rbm, svm = manager.train_rbm_svm(rbm_data=rbm_data, n_features=n_features, rbm_epochs=35, svm_epochs=20, lr=0.05)
        dsvdd = manager.train_dsvdd(dataset=kdd99_dataset, pre_epoch=10, n_epochs=4)
        _, _, join = manager.train_dldm(dataset=kdd99_dataset, n_features=9, lstm_epoch=10, n_code=8, svdd_pre_epoch=10,
                                        svdd_epochs=4, join_alpha=alpha, join_epochs=4)

        for j in range(m):
            print("---------- 第", i + 1, "轮迭代 ----------")
            print("---------- 第", j + 1, "种攻击 ----------")

            """ 更新测试数据集 """
            kdd99_dataset.update_test(dos_types=j + 1, exper_type=2)
            rbm_dataset.update_test(dos_types=j + 1, exper_type=3)
            rbm_data = [rbm_dataset.train, rbm_dataset.train_labels, rbm_dataset.test, rbm_dataset.test_labels]

            """ 测试模型 """
            aek = manager.test_aek(dataset=kdd99_dataset, aek=aek)
            svm = manager.test_rbm_svm(rbm_data=rbm_data, rbm=rbm, svm=svm)
            dsvdd = manager.test_dsvdd(dataset=kdd99_dataset, dsvdd=dsvdd)
            join = manager.test_dldm(dataset=kdd99_dataset, dldm=join)

            """ 获取返回结果 """
            models = [aek, svm, dsvdd, join]

            for k in range(s):
                aucs[k][j] = aucs[k][j] + models[k].results['test_auc']
                mccs[k][j] = mccs[k][j] + models[k].results['test_mcc']
                f_scores[k][j] = f_scores[k][j] + models[k].results['test_f_score']
                times[k][j] = times[k][j] + models[k].results['test_time']

            # fprs = add_list(s=s, datas=fprs, models=models, key='test_ftr')
            # tprs = add_list(s=s, datas=tprs, models=models, key='test_tpr')

    """ 多次结果取均值 """
    aucs = mean_array2(s=s, m=m, n=n, datas=aucs)
    mccs = mean_array2(s=s, m=m, n=n, datas=mccs)
    f_scores = mean_array2(s=s, m=m, n=n, datas=f_scores)
    times = mean_array2(s=s, m=m, n=n, datas=times)

    # fprs, tprs = get_mean_fpr_tpr(s, fprs, tprs)

    """ 绘制 roc 曲线 """
    # plot_mean_rocs(img_name=str(j + 1), img_path=filePaths.img_roc_comp_dos, fprs=fprs, tprs=tprs, aucs=aucs,
    #                names=model_name, colors=['indianred', 'green', 'darkviolet', 'dodgerblue'])

    """ 保存到文件中  用于绘制折线图 """
    model_name = ['ae_kmeans', 'rbm_svm', 'dsvdd', 'join']
    for j in range(9):
        list = init_list(s=4, type=2)
        for i in range(4):
            list[0].append(aucs[i][j])
            list[1].append(f_scores[i][j])
            list[2].append(mccs[i][j])
            list[3].append(times[i][j])
        index = ['AUC', 'F1_SCORE', 'MCC', 'TIME']
        data = pd.DataFrame(columns=model_name, index=index, data=list)
        data.to_csv(urls[j], encoding='gbk')


def compare_exper(device: str = 'cuda', n_features=9, alpha=0.1, n=1, s=5):
    """
        对比实验
            对比对象：ae_kmeans， rbm_svm， rbm_svm（单类），dsvdd， join
            对比指标：roc曲线，roc_auc，f-score，mcc，test_time

        注意：
            1. roc曲线要画出五个对比对象
            2. 将 fpr，tpr 保存文件中，记录实验中期数据
    """

    kdd99_dataset = Kdd99_Dataset(n_features=n_features, exper_type=0, dos_types=0)
    rbm_dataset = Kdd99_Dataset(n_features=n_features, exper_type=1, dos_types=0)
    rbm_one_dataset = Kdd99_Dataset(n_features=n_features, exper_type=4, dos_types=0)

    times = init_list(s=s, type=0)  # 0.0
    aucs = init_list(s=s, type=0)  # 0.0
    mccs = init_list(s=s, type=0)  # 0.0
    f_scores = init_list(s=s, type=0)  # 0.0
    fprs = init_list(s=s, type=2)  # []
    tprs = init_list(s=s, type=2)  # []

    rbm_data = [rbm_dataset.train, rbm_dataset.train_labels, rbm_dataset.test, rbm_dataset.test_labels]
    rbm_one_data = [rbm_one_dataset.train, rbm_one_dataset.train_labels, rbm_one_dataset.test,
                    rbm_one_dataset.test_labels]

    for i in range(n):
        """ 训练+测试模型 """
        print("---------- 第", i + 1, "轮迭代 ----------")
        manager = Manager(device)
        aek = manager.ae_kmeans_manager(dataset=kdd99_dataset, n_epochs=50)
        rbm, svm = manager.rbm_svm_manager(n_features=n_features, rbm_epochs=35, svm_epochs=20, lr=0.05,
                                           rbm_data=rbm_data)
        rbm_one, svm_one = manager.rbm_svm_manager(n_features=n_features, rbm_epochs=35, svm_epochs=20, lr=0.05,
                                                   rbm_data=rbm_one_data)
        dsvdd = manager.dsvdd_manager(dataset=kdd99_dataset, pre_epoch=10, n_epochs=4)
        lstm = manager.lstm_manager(dataset=kdd99_dataset, n_features=n_features, n_epoch=10)
        svdd = manager.lstm_svdd_manager(dataset=kdd99_dataset, lstm=lstm, pre_epoch=10, n_epochs=4)
        join = manager.join_manager(dataset=kdd99_dataset, lstm=lstm, svdd=svdd, alpha=alpha, n_features=n_features,
                                    n_epochs=4)

        """ 获取返回结果 """
        models = [aek, svm, svm_one, dsvdd, join]
        svm.results['test_time'] = svm.results['test_time'] + rbm.results['test_time']
        svm_one.results['test_time'] = svm_one.results['test_time'] + rbm_one.results['test_time']

        aucs = add_list(s=s, datas=aucs, models=models, key='test_auc')
        mccs = add_list(s=s, datas=mccs, models=models, key='test_mcc')
        times = add_list(s=s, datas=times, models=models, key='test_time')
        f_scores = add_list(s=s, datas=f_scores, models=models, key='test_f_score')
        fprs = add_list(s=s, datas=fprs, models=models, key='test_ftr')
        tprs = add_list(s=s, datas=tprs, models=models, key='test_tpr')

    """ 多次结果取均值 """
    times = [times[i] / n for i in range(s)]
    aucs = [aucs[i] / n for i in range(s)]
    mccs = [mccs[i] / n for i in range(s)]
    f_scores = [f_scores[i] / n for i in range(s)]
    fprs, tprs = get_mean_fpr_tpr(s, fprs, tprs)

    """ 打印结果 """
    model_name = ['ae_kmeans', 'rbm_svm', 'rbm_svm_one', 'dsvdd', 'join']
    print_list(s=s, model_name=model_name, label_name='aucs', datas=aucs)
    print_list(s=s, model_name=model_name, label_name='times', datas=times)
    print_list(s=s, model_name=model_name, label_name='mccs', datas=mccs)
    print_list(s=s, model_name=model_name, label_name='f_scores', datas=f_scores)

    """ 绘制 roc 曲线 """
    plot_mean_rocs(img_name='c1', img_path=filePaths.img_roc_comp, fprs=fprs, tprs=tprs, aucs=aucs,
                   names=model_name, colors=['indianred', 'green', 'yellow', 'darkviolet', 'dodgerblue'])

    """ 保存文件 -- fprs tprs 用于绘制多次实验的平均 roc 曲线"""
    save_fpr_tpr(fprs, tprs, type=1)


if __name__ == '__main__':
    """
        属性
            n_features:   选取数据集中特征数，现在是9，最好不要改动
            alpha:        dldm联合训练时，两个损失回传的权重参数
            s:            对比算法的数量
            n:            实验的执行次数

        实验
            self_compare_exper：自对比实验
            compare_exper：对比实验
            compare_exper_dos_types：对比实验 -- 多种dos攻击实验
    """
    n = 10
    s = [2, 5, 4]
    alpha = 0.1
    n_features = 9

    device = init_device()
    init_logger()

    """ 自对比实验 """
    self_compare_exper(device=device, n_features=n_features, alpha=alpha, n=n, s=s[0])

    """ 自对比实验 -- 根据中间结果绘制roc曲线 """
    # fprs, tprs = load_fpr_tpr(type=0)
    # aucs = [0.9907, 0.9961]  # 该数据需要手动保存
    # plot_mean_rocs(img_name='s1', img_path=filePaths.img_roc_self, fprs=fprs, tprs=tprs, aucs=aucs,
    #                names=['non joint training', 'joint training'], colors=['green', 'dodgerblue'])

    """ 对比实验 """
    # compare_exper(device=device, n_features=n_features, alpha=alpha, n=n, s=s[1])

    """ 对比实验 -- 根据中间结果绘制roc曲线 """
    # fprs, tprs = load_fpr_tpr(type=1)
    # aucs = [0.9520, 0.9471, 0.8007, 0.9760, 0.9962]
    # plot_mean_rocs(img_name='c1', img_path=filePaths.img_roc_comp, fprs=fprs, tprs=tprs, aucs=aucs,
    #                names=['ae_kmeans', 'rbm_svm', 'rbm_svm_one', 'dsvdd', 'join'],
    #                colors=['indianred', 'green', 'yellow', 'darkviolet', 'dodgerblue'])

    """ 对比实验 -- 多种攻击 """
    # compare_exper_dos_types(device=device, n_features=n_features, alpha=alpha, n=n, s=s[2])

    """ 对比实验 -- 多种攻击 -- 根据中间结果绘制折线图 """
    # plot_broken_line_D()  # 绘制 dos 折线图
    # plot_broken_line_N()  # 绘制非 dos 折线图
