import torch
import numpy as np
import csv
import pandas


def z_score_normalization(x):
    x = (x - np.mean(x)) / np.std(x)
    return x


def z_score_normalizations(x_mat):
    """ 标准化数据 -- 只考虑连续型数据 -- 第一个proto为离散型 """
    for i in range(1, x_mat.shape[1]):
        x_mat[:, i] = z_score_normalization(x_mat[:, i])

    return x_mat


def min_max_normalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def min_max_normalizations(x_mat):
    """ 归一化数据 """
    for i in range(x_mat.shape[1]):
        x_mat[:, i] = min_max_normalization(x_mat[:, i])

    return x_mat


def load_data_kdd99(handled_file, final_file,  features):
    """
        数据预处理：
            输入文件名 xx.csv
            输出文件名 xx_final.csv

        选取实验需要的特征，就是下面八个特征。
            1  protocol_type 离散型
            3  src_bytes
            22 count
            23 srv_count
            30 srv_diff_host_rate
            31 dst_host_count
            32 dst_host_srv_count
            37 dst_host_src_diff_host_rate
            xx duration

        标准化处理
            连续型：l2 norm
            离散型：不作处理

        归一化处理
            离散型：最大值最小值归一化
            连续型：最大值最小值归一化

    """
    fr = open(handled_file)
    lines = fr.readlines()
    rows = len(lines)

    x_mat = np.zeros((rows, features))
    y_label = np.zeros(rows)

    for i in range(rows):
        line = lines[i].strip()
        item_mat = line.split(',')

        # 考虑src_bytes
        if features == 9:
            x_mat[i][0] = item_mat[1]   # protocol_type -- 离散型
            x_mat[i][1] = item_mat[4]   # src_bytes
            x_mat[i][2] = item_mat[22]  # count
            x_mat[i][3] = item_mat[23]  # srv_count
            x_mat[i][4] = item_mat[30]  # srv_diff_host_rate
            x_mat[i][5] = item_mat[31]  # dst_host_count
            x_mat[i][6] = item_mat[32]  # dst_host_srv_count
            x_mat[i][7] = item_mat[36]  # dst_host_src_diff_host_rate
            x_mat[i][8] = item_mat[0]   # duration

        # 不考虑src_bytes
        elif features == 8:
            x_mat[i][0] = item_mat[1]   # protocol_type -- 离散型
            x_mat[i][1] = item_mat[22]  # count
            x_mat[i][2] = item_mat[23]  # srv_count
            x_mat[i][3] = item_mat[30]  # srv_diff_host_rate
            x_mat[i][4] = item_mat[31]  # dst_host_count
            x_mat[i][5] = item_mat[32]  # dst_host_srv_count
            x_mat[i][6] = item_mat[36]  # dst_host_src_diff_host_rate
            x_mat[i][7] = item_mat[0]   # duration

        y_label[i] = item_mat[41]   # label
        # print(i, "-", x_mat[i][0], x_mat[i][1], x_mat[i][2], x_mat[i][3], x_mat[i][4], x_mat[i][5], x_mat[i][6], x_mat[i][7])

    fr.close()
    x_mat = z_score_normalizations(x_mat)
    x_mat = min_max_normalizations(x_mat)

    write_file(x_mat, final_file)
    return x_mat, y_label


def write_file(data, data_file):
    data_file = open(data_file, 'w', newline='')
    csv_writer = csv.writer(data_file)
    count = 0  # 行数
    for row in data:  # 循环读取文件数据
        temp_line = np.array(row)
        csv_writer.writerow(temp_line)
        count += 1
        # print(count, 'final:', temp_line[0], temp_line[1], temp_line[2], temp_line[3])
    data_file.close()
