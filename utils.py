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


def load_data(mm, np, cache, path, w_cache, skip_row, skip_col, seps, transpose, label, path_label, col_name):
    if mm:
        data_dict, dim = load_mm(path, w_cache, skip_row, skip_col, seps, transpose, label, path_label, col_name)
    elif np:
        data_dict, dim = load_mat(path, w_cache, skip_row, skip_col, seps, transpose, label, path_label, col_name)
    elif cache:
        data_dict, dim = load_cache(path)
    else:
        raise Exception
    return data_dict, dim


def write_cache(data_dict):
    print("Writing cache to {}".format("./cache/cache.pkl"))
    if not os.path.exists("./cache"):
        os.mkdir("./cache")
    if os.path.isfile("./cache/cache.pkl"):
        while True:
            ans = input("File ./cache/cache.pkl exists. Confirm overwrite ? [Y/N] ")
            if ans.lower() == 'y':
                os.remove("./cache/cache.pkl")
                break
            elif ans.lower() == 'n':
                raise FileExistsError("File with default cache name exists.")
            else:
                print("Invalid input.")

    with open("./cache/cache.pkl", 'wb') as f:
        pkl.dump(data_dict, f)


def load_mm(path_to_file, cache, skip_row, skip_col, seps, transpose, label, path_to_label, col_name="Group"):
    if not os.path.isfile(path_to_file):
        raise FileNotFoundError("File {} not found.".format(path_to_file))
    print("Loading data from: {}".format(path_to_file))
    exp_mat_mm = mmread(path_to_file)
    exp_mat_dense = exp_mat_mm.todense()
    data = pd.DataFrame(exp_mat_mm,
                      range(skip_row, exp_mat_dense.shape[0] + 1), range(skip_col, exp_mat_dense.shape[1] + 1))
    data = data.to_numpy() if not transpose else data.to_numpy().T
    print("Expression Matrix Shape: {}".format(data.shape))

    if label:
        label = load_label(path_to_label, seps, col_name)
        data_dict = {'data': data, 'label': label}
    else:
        data_dict = {'data': data}
    if cache:
        write_cache(data_dict)
    return data_dict, data_dict['data'].shape[1]


def load_mat(path_to_file, cache, skip_row, skip_col, seps, transpose, label, path_to_label, col_name="Group"):
    if not os.path.isfile(path_to_file):
        raise FileNotFoundError("File {} not found.".format(path_to_file))
    print("Loading data from: {}".format(path_to_file))
    data = pd.read_csv(path_to_file, skiprows=skip_row, sep=seps, header=None)
    data = data.iloc[:, skip_col:].astype(float)
    data = data.to_numpy() if not transpose else data.to_numpy().T
    print("Expression Matrix Shape: {}".format(data.shape))

    if label:
        label = load_label(path_to_label, seps, col_name)
        data_dict = {'data': data, 'label': label}
    else:
        data_dict = {'data': data}
    if cache:
        write_cache(data_dict)
    return data_dict, data_dict['data'].shape[1]


def load_cache(path_to_cache):
    if not os.path.isfile(path_to_cache):
        raise FileNotFoundError("Cache {} not found.".format(path_to_cache))
    print("Loading cache from: {}".format(path_to_cache))
    with open(path_to_cache, 'rb') as f:
        data_dict = pkl.load(f)
    print("Cached Shape: {}".format(data_dict['data'].shape))
    return data_dict, data_dict['data'].shape[1]


def load_label(path_to_label, seps, col_name):
    print("Loading label from: {}".format(path_to_label))
    label = pd.read_csv(path_to_label, sep=seps, usecols=[col_name])
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
