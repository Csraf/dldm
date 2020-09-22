""" 数据集 """

src_train = '../data/dataset/kddcup.data_10_percent_corrected' # 原始kdd99训练集
handle_train = '../data/dataset/kddcup.data_10_percent_corrected.csv' # 指定kdd99测试集
final_train = '../data/dataset/kddcup.data_10_percent_final.csv' # 归一化，标准化后的kdd99训练集
src_test = '../data/dataset/corrected' # 原始kdd99测试集
handle_test = '../data/dataset/corrected.csv' # 指定kdd99测试集
final_test = '../data/dataset/corrected_final.csv' # 归一化，标准化后的kdd99测试集


""" 中间数据 fpr + tpr 来绘制roc曲线  """

self_no_join_fpr = '../data/numpy_self/fpr_tpr/no_join_fpr'
self_no_join_tpr = '../data/numpy_self/fpr_tpr/no_join_tpr'
self_join_fpr = '../data/numpy_self/fpr_tpr/join_fpr'
self_join_tpr = '../data/numpy_self/fpr_tpr/join_tpr'

ae_kmeans_fpr = '../data/numpy_comp/fpr_tpr/ae_kmeans_fpr'
ae_kmeans_tpr = '../data/numpy_comp/fpr_tpr/ae_kmeans_tpr'
rbm_fpr = '../data/numpy_comp/fpr_tpr/rbm_fpr'
rbm_tpr = '../data/numpy_comp/fpr_tpr/rbm_tpr'
rbm_one_fpr = '../data/numpy_comp/fpr_tpr/rbm_one_fpr'
rbm_one_tpr = '../data/numpy_comp/fpr_tpr/rbm_one_tpr'
dsvdd_fpr = '../data/numpy_comp/fpr_tpr/dsvdd_fpr'
dsvdd_tpr = '../data/numpy_comp/fpr_tpr/dsvdd_tpr'
join_fpr = '../data/numpy_comp/fpr_tpr/join_fpr'
join_tpr = '../data/numpy_comp/fpr_tpr/join_tpr'

""" 中间数据 对比实验中多种攻击输出的csv文件，用于绘制折线图 """

D1 = '../data/csv_comp_dos/d1.csv'
D2 = '../data/csv_comp_dos/d2.csv'
D3 = '../data/csv_comp_dos/d3.csv'
D4 = '../data/csv_comp_dos/d4.csv'
D5 = '../data/csv_comp_dos/d5.csv'
D6 = '../data/csv_comp_dos/d6.csv'
N1 = '../data/csv_comp_dos/n1.csv'
N2 = '../data/csv_comp_dos/n2.csv'
N3 = '../data/csv_comp_dos/n3.csv'

""" 图像保存位置 """

img_broken_line = "../img/broken_line/"  # 折线图
img_roc_self = "../img/roc_self/"  # 自对比实验roc曲线
img_roc_comp = "../img/roc_comp/"  # 对比实验roc曲线
img_roc_comp_dos = "../img/roc_comp_dos/"  # 对比实验(多种攻击)roc曲线








