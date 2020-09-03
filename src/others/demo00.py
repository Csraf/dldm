""" 绘制折线图 """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd


def plot_broken_line(x, y, x_name="dos", y_name="dos", labels=[], y_scale=()):
    """ 画折线 """
    colors = ['indianred', 'green', 'darkviolet', 'dodgerblue']
    for i in range(len(labels)):
        plt.plot(x, y[i], marker='.', ms=8, label=labels[i], color=colors[i])
    # plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.ylim(y_scale)
    plt.legend(loc="lower right")
    plt.savefig(x_name + " -- " + y_name)
    plt.show()


# 针对六种dos攻击，绘制四个折线图
dos_index = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']

aek_auc = [0.9433856415740012, 0.947414868040155, 0.9559424296941412,
           0.9523469121104213, 0.9351729384275247, 0.9540967294153553]

rbm_auc = [0.9645909784272464, 0.9467386144287848, 0.964805393993899,
           0.954114209995109, 0.9653718570967542, 0.9696089646566612]

dsvdd_auc = [0.9604261462711691, 0.9587020894703094, 0.9580434959473048,
             0.9833540713241353, 0.9681854043861937, 0.977735133243371]

join_auc = [0.972635257988095, 0.9991124710559437, 0.9904970104534608,
            0.9902760029919869, 0.9971007063761737, 0.9962943515012669]

aek_time = [2.637500047683716, 6.263399839401245, 6.403800129890442,
            6.513000130653381, 6.528600096702576, 6.442799925804138]

rbm_time = [1.39199960231781, 3.244800090789795, 3.3384000062942505,
            3.385200023651123, 3.3618000745773315, 3.4911998510360718]

dsvdd_time = [1.0099999904632568, 2.363399863243103, 2.371200203895569,
              2.3711999654769897, 2.3477998971939087, 2.468600034713745]

join_time = [2.227499842643738, 5.038800001144409, 5.148000240325928,
             5.335199952125549, 5.187000155448914, 5.196699857711792]

aek_mcc = [0.8882669139974387, 0.9296727953217344, 0.9411769013077743,
           0.9374452858895601, 0.9105928344580348, 0.9395250643237447]

rbm_mcc = [0.9306028440762373, 0.8431158644986797, 0.9432778460447753,
           0.9184695922319657, 0.8942982503683293, 0.943798755470686]

dsvdd_mcc = [0.9239342616422246, 0.8968443452649877, 0.8211469323798648,
             0.9502432586533233, 0.8348942259576488, 0.8869837828752056]

join_mcc = [0.9215742414313424, 0.9786419815847632, 0.8935914743696063,
            0.9322541038013639, 0.9707623950482906, 0.9609966842571283]

aek_f_score = [0.943510986999412, 0.9852624345307903, 0.9876707597051181,
               0.9869016722951777, 0.9815075983734398, 0.9873340521328722]

rbm_f_score = [0.9646481519913761, 0.9598864095326706, 0.9880612326561149,
               0.982682690548382, 0.9742177220581247, 0.9880620141847912]

dsvdd_f_score = [0.9612387699226923, 0.9781411900576302, 0.9556762518288628,
                 0.9894617472136031, 0.9656971012063635, 0.9758423163178479]

join_f_score = [0.9603065676928841, 0.9953477616600654, 0.9758289682514735,
                0.9856964259504974, 0.993592421610767, 0.9914784431186321]

plot_broken_line(x=dos_index, y=[aek_auc, rbm_auc, dsvdd_auc, join_auc], x_name="dos", y_name="AUC",
                 labels=["AE+K-Means", "RBM+SVM", "DSVDD", "DLDM"], y_scale=(0.8, 1))
plot_broken_line(x=dos_index, y=[aek_time, rbm_time, dsvdd_time, join_time], y_name="time(T)", x_name="dos",
                 labels=["AE+K-Means", "RBM+SVM", "DSVDD", "DLDM"], y_scale=(1, 10))
plot_broken_line(x=dos_index, y=[aek_mcc, rbm_mcc, dsvdd_mcc, join_mcc], y_name="MCC", x_name="dos",
                 labels=["AE+K-Means", "RBM+SVM", "DSVDD", "DLDM"], y_scale=(0.8, 1))
plot_broken_line(x=dos_index, y=[aek_f_score, rbm_f_score, dsvdd_f_score, join_f_score], y_name="F1-score", x_name="dos",
                 labels=["AE+K-Means", "RBM+SVM", "DSVDD", "DLDM"], y_scale=(0.8, 1))

# 针对三种非dos攻击，绘制四个折线图
no_dos_index = ['N1', 'N2', 'N3']

aek_no_auc = [0.9533200204348498, 0.943706545038636, 0.9436128857995237]
rbm_no_auc = [0.9440091351759785, 0.964887568787155, 0.9344018099705296]
dsvdd_no_auc = [0.9769113777401621, 0.9641404391922703, 0.9655532097904226]
join_no_auc = [0.9887887729045266, 0.9820514861745548, 0.9838431751218855]

aek_no_time = [7.0345001220703125, 7.298500061035156, 7.342800164222717]
rbm_no_time = [3.5749998092651367, 3.788999915122986, 3.8773999214172363]
dsvdd_no_time = [2.5230000019073486, 2.571500062942505, 2.5243998289108276]
join_no_time = [5.888000011444092, 5.693999886512756, 5.658599901199341]

aek_no_mcc = [0.9382349742065633, 0.8946928074955529, 0.8980291680376296]
rbm_no_mcc = [0.8885091373098875, 0.8962696751796724, 0.8784388796789666]
dsvdd_no_mcc = [0.9363786678083795, 0.817668629600136, 0.8390182446178847]
join_no_mcc = [0.9273492851353774, 0.8934831228810469, 0.9040627632141891]

aek_no_f_score = [0.9870710822203559, 0.9780884056033377, 0.9788921055772994]
rbm_no_f_score = [0.9756852908818276, 0.9764350766583919, 0.9748186713881314]
dsvdd_no_f_score = [0.9865485288095759, 0.9596866843605187, 0.9668908269381703]
join_no_f_score = [0.98470418055504, 0.9762018332331466, 0.9789757223770312]


plot_broken_line(x=no_dos_index, y=[aek_no_auc, rbm_no_auc, dsvdd_no_auc, join_no_auc], y_name="AUC",
                 x_name="other attack",
                 labels=["AE+K-Means", "RBM+SVM", "DSVDD", "DLDM"], y_scale=(0.8, 1))
plot_broken_line(x=no_dos_index, y=[aek_no_time, rbm_no_time, dsvdd_no_time, join_no_time], y_name="time(T)",
                 x_name="other attack",
                 labels=["AE+K-Means", "RBM+SVM", "DSVDD", "DLDM"], y_scale=(1, 10))
plot_broken_line(x=no_dos_index, y=[aek_no_mcc, rbm_no_mcc, dsvdd_no_mcc, join_no_mcc], y_name="MCC",
                 x_name="other attack",
                 labels=["AE+K-Means", "RBM+SVM", "DSVDD", "DLDM"], y_scale=(0.8, 1))
plot_broken_line(x=no_dos_index, y=[aek_no_f_score, rbm_no_f_score, dsvdd_no_f_score, join_no_f_score],
                 y_name="F1-score",
                 x_name="other attack", labels=["AE+K-Means", "RBM+SVM", "DSVDD", "DLDM"], y_scale=(0.8, 1))
