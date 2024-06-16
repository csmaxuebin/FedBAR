import os

import numpy as np
import torch

dataset_path_cifar10 = r'../dataset/cifar10/10c_batch64_noniid_dir'
dataset_path_cifar100 = r'../dataset/cifar100/10c_batch64_noniid_dir/'

def read_data(dataset, idx, is_train=True):
    """

    Args:
        dataset: 数据集名字，String
        idx: 数据集文件序号
        is_train: 是否是训练集

    Returns:数据集本身
    TODO 要加数据集在这里加
    """
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data

# def read_data(dataset, idx, is_train=True, usedata=None):
#     """
#
#     Args:
#         dataset: 数据集名字，String
#         idx: 数据集文件序号
#         is_train: 是否是训练集
#
#     Returns:数据集本身
#     TODO 要加数据集在这里加
#     """
#
#     # dataset_path = r'../dataset/cifar10/50c_least2c_batch64_noniid'
#     if usedata is not None:
#         path = usedata
#     if is_train:
#         train_data_dir = os.path.join(path, 'train/')
#         # train_data_dir = dataset_path + 'train/'
#
#         train_file = train_data_dir + str(idx) + '.npz'
#         with open(train_file, 'rb') as f:
#             train_data = np.load(f, allow_pickle=True)['data'].tolist()
#
#         return train_data
#
#     else:
#         test_data_dir = os.path.join(path, 'test/')
#         # test_data_dir = dataset_path + 'test/'
#
#         test_file = test_data_dir + str(idx) + '.npz'
#         with open(test_file, 'rb') as f:
#             test_data = np.load(f, allow_pickle=True)['data'].tolist()
#
#         return test_data


def read_client_data(dataset, idx, is_train=True, bound=-1, usedata=None):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx, is_train)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data if bound < 0 else train_data[:bound]
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data if bound < 0 else test_data[:bound]


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
