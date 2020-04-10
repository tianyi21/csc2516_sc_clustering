#   -*- coding: utf-8 -*-
#
#   eval.py
#
#   Developed by Tianyi Liu on 2020-03-09 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, silhouette_score


def compute_loss(model, x, y, q=None, p=None, mu=None, logvar=None):
    def _loss_weighted_se(x, y):
        diff = y - x
        weighted_diff = torch.where(diff > 0, 0.5 * diff ** 2, 4 * diff ** 2)
        # weighted_diff = diff ** 2
        weighted_diff_sum = torch.sum(weighted_diff) / (y.size(0) * y.size(1))
        return weighted_diff_sum

    """
    def _loss_cluster_dist(x, cls, metric="Euclidean"):
        if metric != "Euclidean" and metric != "Absolute":
            raise NotImplementedError
        loss_intra, loss_inter = 0, 0
        for item, item_cls in zip(x, cls):
            dist = (x - item) ** 2 if metric == "Euclidean" else torch.abs(x - item)
            dist = torch.sum(dist, dim=1)
            loss_intra += torch.sum(dist[cls == item_cls])
            loss_inter += torch.sum(dist[cls != item_cls])
        return loss_intra / x.size(0), loss_inter / x.size(0)

    def _loss_cluster_num(count, softmax_prob=False):
        count = count.float()
        if softmax_prob:
            sm = nn.Softmax()
            count = sm(count)
        else:
            count /= count.sum()
        loss_entropy = - torch.sum(count * torch.log2(count))
        return loss_entropy
    """

    if model == "ae":
        loss_weighted_se = _loss_weighted_se(x, y)
        return loss_weighted_se
    elif model == "vae":
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss_weighted_se = _loss_weighted_se(x, y)
        return loss_weighted_se, loss_kl


def cal_ari(pred, truth):
    return adjusted_rand_score(pred, truth.cpu().numpy())


def cal_silhouette(data, pred):
    return silhouette_score(data.cpu().numpy(), pred)
