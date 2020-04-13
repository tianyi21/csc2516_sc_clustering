#   -*- coding: utf-8 -*-
#
#   utils.py
#
#   Developed by Tianyi Liu on 2020-03-05 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""

import os
import pickle as pkl
from scipy.io import mmread
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
import numpy as np

from cfgs import NUMPY_SEED, TORCH_SEED


def load_data(mm, np, cache, file_path, w_cache, skip_row, skip_col, seps, transpose, label, label_path, cache_name, col_name):
    if mm:
        data_dict, dim = load_mm(file_path, w_cache, skip_row, skip_col, seps, transpose, label, label_path, cache_name, col_name)
    elif np:
        data_dict, dim = load_mat(file_path, w_cache, skip_row, skip_col, seps, transpose, label, label_path, cache_name, col_name)
    elif cache:
        data_dict, dim = load_cache(file_path)
    else:
        raise Exception
    return data_dict, dim


def write_cache(data_dict, cache_name):
    cache_path = os.path.join("./cache", cache_name)
    print("Writing cache to {}".format(cache_path))
    if not os.path.exists("./cache"):
        os.mkdir("./cache")
    if os.path.isfile(cache_path):
        while True:
            ans = input("File {} exists. Confirm overwrite ? [Y/N] ".format(cache_path))
            if ans.lower() == 'y':
                os.remove(cache_path)
                break
            elif ans.lower() == 'n':
                raise FileExistsError("File with default cache name exists.")
            else:
                print("Invalid input.")

    with open(cache_path, 'wb') as f:
        pkl.dump(data_dict, f)


def load_mm(file_path, cache, skip_row, skip_col, seps, transpose, label, label_path, cache_name, col_name="Group"):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File {} not found.".format(file_path))
    print("Loading data from: {}".format(file_path))
    exp_mat_mm = mmread(file_path)
    exp_mat_dense = exp_mat_mm.todense()
    data = pd.DataFrame(exp_mat_mm,
                      range(skip_row, exp_mat_dense.shape[0] + 1), range(skip_col, exp_mat_dense.shape[1] + 1))
    data = data.to_numpy() if not transpose else data.to_numpy().T
    print("Expression Matrix Shape: {}".format(data.shape))

    if label:
        label = load_label(label_path, seps, col_name)
        data_dict = {'data': data, 'label': label}
    else:
        data_dict = {'data': data}
    if cache:
        write_cache(data_dict, cache_name)
    return data_dict, data_dict['data'].shape[1]


def load_mat(file_path, cache, skip_row, skip_col, seps, transpose, label, label_path, cache_name, col_name="Group"):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File {} not found.".format(file_path))
    print("Loading data from: {}".format(file_path))
    data = pd.read_csv(file_path, skiprows=skip_row, sep=seps, header=None)
    data = data.iloc[:, skip_col:].astype(float)
    data = data.to_numpy() if not transpose else data.to_numpy().T
    print("Expression Matrix Shape: {}".format(data.shape))

    if label:
        label = load_label(label_path, seps, col_name)
        data_dict = {'data': data, 'label': label}
    else:
        data_dict = {'data': data}
    if cache:
        write_cache(data_dict, cache_name)
    return data_dict, data_dict['data'].shape[1]


def load_cache(cache_path):
    if not os.path.isfile(cache_path):
        raise FileNotFoundError("Cache {} not found.".format(cache_path))
    print("Loading cache from: {}".format(cache_path))
    with open(cache_path, 'rb') as f:
        data_dict = pkl.load(f)
    print("Cached Shape: {}".format(data_dict['data'].shape))
    return data_dict, data_dict['data'].shape[1]


def load_label(label_path, seps, col_name):
    print("Loading label from: {}".format(label_path))
    label = pd.read_csv(label_path, sep=seps, usecols=[col_name])
    label = np.squeeze(label.to_numpy().astype(np.int))
    print("Label Shape: {} with {} clusters".format(label.shape, np.unique(label)))
    return label


def split_data(data_dict, label, device, batch_size, tr_ratio, vl_ratio, ts_ratio=.0, sub=1):
    torch.manual_seed(TORCH_SEED)
    sample_length = int(sub * len(data_dict['data']))
    print("Subsampled dataset with {}/{} ({}%) data points.".format(sample_length, len(data_dict['data']), np.round(
        100 * sample_length / len(data_dict['data']), 2)))
    data = torch.tensor(data_dict['data'][:sample_length]).to(device).float()
    if label:
        labels = torch.tensor(data_dict['label'][:sample_length]).to(device).long()
        assert len(data) == len(labels)
        dataset = TensorDataset(data, labels)
    else:
        dataset = TensorDataset(data)

    tr_size, vl_size, ts_size = int(np.round(len(data) * tr_ratio)), int(np.round(len(data) * vl_ratio)), int(
        len(data) - np.round(len(data) * tr_ratio) - np.round(len(data) * vl_ratio))

    tr_set, vl_set, ts_set = random_split(dataset, [tr_size, vl_size, ts_size])

    tr_loader, vl_loader = DataLoader(tr_set, batch_size, shuffle=False), DataLoader(vl_set, batch_size, shuffle=False)
    ts_loader = DataLoader(ts_set, batch_size, shuffle=False) if ts_ratio != 0 else None
    print("Dataset split as: Train {} ({}%); Validate {} ({}%); Test {} ({}%)\n".format(
        tr_size, 100 * np.round(tr_size / len(dataset), 2),
        vl_size, 100 * np.round(vl_size / len(dataset), 2),
        ts_size, 100 * np.round(ts_size / len(dataset), 2)))
    return tr_loader, vl_loader, ts_loader


def add_noise(data_dict, noise_type, prob=0.2, sig=0.5):
    def _add_dropout(data, ops):
        print("Simulating dropout noise with prob={}".format(ops))
        mask = np.random.binomial(1, 1 - ops, size=data.shape)
        data *= mask
        return data

    def _add_gaussian(data, sig):
        print("Simulating Gaussian noise with relative sigma={}".format(sig))
        mask = np.random.normal(loc=0, scale=sig, size=data.shape)
        mask = np.where(mask < -1, -1, mask)
        mask = np.where(mask > 1, 1, mask)
        mask += 1
        data *= mask
        return data.astype(np.int)

    np.random.seed(NUMPY_SEED)
    data = data_dict['data']
    if noise_type.lower() == "dropout":
        data = _add_dropout(data, prob)
    elif noise_type.lower() == "gaussian":
        data = _add_gaussian(data, sig)
    elif noise_type.lower() == 'd+g':
        data = _add_dropout(data, prob)
        data = _add_gaussian(data, sig)
    elif noise_type.lower() == 'none':
        pass
    else:
        raise NotImplementedError
    data_dict['data'] = data
    return data_dict


class Logger():
    def __init__(self, path, filename, loss_logger=False, benchmark_logger=False):
        self.filename = filename
        self.path = path
        if not os.path.exists(self.path):
            os.mkdir(self.path)
            print("Directory {} created for logging.".format(self.path))
        self.f = open(os.path.join(self.path, self.filename), 'w')
        if not loss_logger and not benchmark_logger:
            self.f.write("EPOCH\t\t# Clusters\t\tARI\t\tSilhouette\t\tTime\n")
        elif loss_logger:
            self.f.write("EPOCH\t\tStep\t\tTrain Loss\t\tValidation Loss\n")
        elif benchmark_logger:
            self.f.write("Dimension\t\tMethod\t\t# Cluster\t\tARI\t\tSilhouette\t\tTime\n")

    def log(self, line):
        self.f.write(line + '\n')

    def close(self):
        self.f.close()
