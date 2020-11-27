# coding:utf-8
import numpy as np
import pandas as pd
import csv

global label_list


def find_index(x, y):
    """ 将相应的非数字类型转换为数字标识即符号型数据转化为数值型数据 """
    return [i for i in range(len(y)) if y[i] == x]


def handle_protocol(inputs):
    """ 数值化协议类型 """
    protocol_list = ['tcp', 'udp', 'icmp']
    if inputs[1] in protocol_list:
        return find_index(inputs[1], protocol_list)[0]


def handle_service(inputs):
    """ 数值化70种网络服务类型 """
    service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
                    'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
                    'hostnames',
                    'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell',
                    'ldap',
                    'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp',
                    'nntp',
                    'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
                    'shell',
                    'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time',
                    'urh_i', 'urp_i',
                    'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
    if inputs[2] in service_list:
        return find_index(inputs[2], service_list)[0]


def handle_flag(inputs):
    """ 数值化网络连接状态 """
    flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    if inputs[3] in flag_list:
        return find_index(inputs[3], flag_list)[0]


def handle_label(inputs):
    """
        标签类型：正常数据，dos攻击，其他攻击
        index   dos type
        4       neptune
        5       smurf
        7       pod
        8       teardrop
        11      land
        13      back
        1       buffer_overflow  -- U2R
        6       guess_passwd  -- R2L
        10      ipsweep  -- Probing
    """
    label_list = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
                  'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
                  'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
                  'spy.', 'rootkit.']
    # global label_list
    if inputs[41] in label_list:
        return find_index(inputs[41], label_list)[0]
    else:
        label_list.append(inputs[41])
        return find_index(inputs[41], label_list)[0]


def get_type_dos(dos_type=0, data=0):
    """ 获取指定类型的 dos """
    if dos_type == 1:
        return int(data) == 4  # neptune
    elif dos_type == 2:
        return int(data) == 5  # smurf
    elif dos_type == 3:
        return int(data) == 13  # back
    else:
        return False


def get_types_dos(dos_types=0, data=0):
    """ 获取指定数量的 dos """
    if dos_types == 1:
        return int(data) == 4  # neptune
    elif dos_types == 2:
        return int(data) == 4 or int(data) == 5  # + smurf
    elif dos_types == 3:
        return int(data) == 13 or int(data) == 4 or int(data) == 5  # + back
    elif dos_types == 4:
        return int(data) == 13 or int(data) == 4 or int(data) == 5 or int(data) == 8  # + teardrop
    elif dos_types == 5:
        return int(data) == 13 or int(data) == 4 or int(data) == 5 or int(data) == 7 or int(data) == 8  # + pod
    elif dos_types == 6:
        return int(data) == 13 or int(data) == 4 or int(data) == 5 or int(data) == 7 or int(data) == 8 or int(
            data) == 11  # + land
    elif dos_types == 7:
        return int(data) == 13 or int(data) == 4 or int(data) == 5 or int(data) == 7 or int(data) == 8 or int(
            data) == 11 or int(data) == 1  # + buffer_overflow
    elif dos_types == 8:
        return int(data) == 13 or int(data) == 4 or int(data) == 5 or int(data) == 7 or int(data) == 8 or int(
            data) == 11 or int(data) == 1 or int(data) == 6  # + guess_passwd
    elif dos_types == 9:
        return int(data) == 13 or int(data) == 4 or int(data) == 5 or int(data) == 7 or int(data) == 8 or int(
            data) == 11 or int(data) == 1 or int(data) == 6 or int(data) == 10  # + ipsweep
    else:
        return False


def get_all_dos(data=0):
    """ 获取所有标签为 dos 的攻击 """
    return int(data) == 4 or int(data) == 5 or int(data) == 7 or int(data) == 8 or int(
        data) == 11 or int(data) == 13


def selectdatas(label, train=0, exper_type=0, dos_types=0):
    """
        按照实验需求，按行选取 kdd99 数据集，并处理数据集的标签
        label：数据的原始标签（0-20）
        train：数据集类型的标志位
            0：训练集
            1：测试集
        exper_type：代表实验类型
            0：基础实验 / 基础对比实验（join，ae_kmeans，dsvdd），训练集：所有正常数据，测试集：所有正常数据+所有dos攻击
            1：基础对比实验 （rbm）：训练集：所有正常数据+所有dos攻击，测试集：所有正常数据+所有dos攻击
            2：攻击对比实验 （join，ae_kmeans，dsvdd）：训练集：所有正常数据，测试集：所有正常数据 + 指定攻击
            3：攻击对比实验 （rbm）：训练集：所有正常数据+所有dos攻击，测试集：所有正常数据 + 指定攻击
            4：基础对比实验(单类 rbm)：训练集：所有正常数据，100:1的异常数据，测试集：所有正常数据+所有dos攻击
        dos_types：攻击种类
            0-9：分别代表九种攻击，前六种是dos攻击，后三种是其他攻击。
        return: 数据对应的标签
    """
    if exper_type == 0:  # 训练集：所有正常数据，测试集：所有正常数据+所有dos攻击
        if train == 0:  # 训练集
            if label == 0:  # 正常数据
                return 0
        if train == 1:  # 测试集
            if label == 0:
                return 0
            if get_all_dos(label):
                return 1

    elif exper_type == 1:  # 训练集：所有正常数据+所有dos攻击，测试集：所有正常数据+所有dos攻击
        if train == 0:
            if label == 0:
                return 0
            elif get_all_dos(label):
                return 1
        if train == 1:
            if label == 0:
                return 0
            elif get_all_dos(label):
                return 1

    elif exper_type == 2:  # 训练集：所有正常数据，测试集：所有正常数据 + 指定攻击
        if train == 0:
            if label == 0:
                return 0
        if train == 1:
            if label == 0:
                return 0
            elif get_types_dos(dos_types, label):
                return 1

    elif exper_type == 3:  # 训练集：所有正常数据+所有dos攻击，测试集：所有正常数据 + 指定攻击
        if train == 0:
            if label == 0:  # 正常数据
                return 0
            elif get_all_dos(label):
                return 1
        if train == 1:  # 测试集
            if label == 0:
                return 0
            elif get_types_dos(dos_types, label):
                return 1

    elif exper_type == 4:  # 训练集：所有正常数据，100:1的异常数据，测试集：所有正常数据+所有dos攻击
        if train == 0:  # 训练集
            if label == 0:  # 正常数据
                return 0
            elif get_all_dos(label):
                return 1
        if train == 1:  # 测试集
            if label == 0:
                return 0
            elif get_all_dos(label):
                return 1
    return -1  # 不需要提取该数据


def pre_file(source_file, handled_file, train, exper_type=0, dos_types=0):
    """
        根据不同实验需求：
            1. 将字符型 转换为 数值型
            2. 获取指定训练数据和测试数据
            3. 输入文件名 xx，输出文件名 xx.csv

        属性：
            train：数据集类型，训练集或者测试集

            dos_types：dos 攻击种类数 [1-9]

            exper_type：代表实验类型
                0：基础实验 / 基础对比实验（join，ae_kmeans，dsvdd），训练集获取正常数据，测试集获取所有数据
                1：基础对比实验 （rbm）：训练集获取所有数据，测试集获取所有数据
                2：攻击对比实验 （join，ae_kmeans，dsvdd）：训练集获取正常数据，测试集获取正常数据 + 指定攻击
                3：攻击对比实验 （rbm）：训练集获取所有数据，测试集获取正常数据 + 指定攻击
                4：基础对比实验(单类 rbm)：训练集获取所有正常数据，正常数据和异常数据比例为100:1，测试集获取所有数据
    """
    data_file = open(handled_file, 'w', newline='')
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        csv_writer = csv.writer(data_file)
        count = 0  # 行数
        if exper_type == 4:
            err = 0  # 训练集中异常数据条目数
        for row in csv_reader:
            temp_line = np.array(row)
            temp_line[1] = handle_protocol(row)
            temp_line[2] = handle_service(row)
            temp_line[3] = handle_flag(row)
            temp_line[41] = handle_label(row)
            # Ajoy 获取所有的正常数据，并写入相关文件中（分别将训练集和测试集中的正常数据放入不同的文件中）
            label = selectdatas(int(temp_line[41]), train, exper_type, dos_types)
            if label == -1:
                continue

            temp_line[41] = label
            count += 1
            csv_writer.writerow(temp_line)

            if exper_type == 4 and label == 1 and train == 0:  # 训练集中寻找比例数据100:1的异常值
                err += 0
                if err >= 972:
                    break

        data_file.close()
