#   -*- coding: utf-8 -*-
#
#   analyze.py
#   
#   Developed by Tianyi Liu on 2020-04-01 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""

from utils import load_data, random_split
from eval import adjusted_rand_score, silhouette_score
from cfgs import *

import os
import argparse
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import time


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m',
                       dest="mm",
                       action="store_true",
                       help="Set -> Read Matrix Marker format")
    group.add_argument('-n',
                       dest="np",
                       action="store_true",
                       help="Set -> Read general csv format")
    group.add_argument('-c',
                       dest="cache",
                       action="store_true",
                       help="Set -> Read cached data")
    parser.add_argument('-l',
                        dest="label",
                        action="store_false",
                        help="Set -> DO NOT read label during training")
    parser.add_argument('--path',
                        dest="path",
                        default="./cache/cache.pkl",
                        help="Specify the path of data/cache")
    parser.add_argument('--path_label',
                        dest="path_label",
                        default="./gt.csv",
                        help="Specify the path of label")
    parser.add_argument('-t',
                        dest="transpose",
                        action="store_true",
                        help="Set -> Transpose the data read")
    parser.add_argument('-w',
                        dest="write_cache",
                        action="store_false",
                        help="Set -> Write to cache if read from data")
    parser.add_argument('--seps',
                        dest="seps",
                        default=',',
                        help="Data separator, e.g., \\t, ,")
    parser.add_argument('--skiprow',
                        dest="skip_row",
                        type=int,
                        default=1,
                        help="Skip row")
    parser.add_argument('--skipcol',
                        dest="skip_col",
                        type=int,
                        default=1,
                        help="Skip column")
    parser.add_argument('--col_name',
                        dest="col_name",
                        default="Group",
                        help="Label column name")
    parser.add_argument('--tsne',
                        dest="tsne",
                        default="./cache/",
                        help="TSNE embedding cache")
    parser.add_argument('--misc',
                        dest='misc',
                        default='./misc/',
                        help='Path to store miscs')
    args = parser.parse_args()
    return args


def linear_correlation(embedding, cluster):
    clf = LogisticRegression()
    clf.fit(embedding, cluster)
    print("Linear classification score: {}".format(clf.score))


def t_sne_visualize(data, labels, misc, plot=False, epoch=None, model=None, sets=None):
    print("Running T-SNE embedding.\n")
    if not os.path.exists(misc):
        print("Directory {} created.".format(misc))
        os.mkdir(misc)
    t_sne_embedding = TSNE(n_components=2).fit_transform(data)
    if labels is None:
        plt.scatter(t_sne_embedding[:, 0], t_sne_embedding[:, 1], s=3)
    else:
        unique_label = np.unique(labels)
        for label in unique_label:
            plt.scatter(t_sne_embedding[labels == label, 0], t_sne_embedding[labels == label, 1], s=3, label=label)
        plt.legend(loc="upper right")
    if sets is None:
        if epoch is None:
            plt.title("T-SNE")
            plt.savefig(os.path.join(misc, "tsne.pdf"), dpi=400)
        else:
            plt.title("T-SNE with {} Epoch {}".format(model.upper(), epoch))
            plt.savefig(os.path.join(misc, "tsne_{}_{}.pdf".format(model, epoch)), dpi=400)
    else:
        plt.title("T-SNE of {} set".format(sets))
        plt.savefig(os.path.join(misc, "tsne_{}.pdf".format(sets)), dpi=400)
    if plot:
        plt.show()
    plt.clf()
    return t_sne_embedding


def k_means_cls(data, label, k):
    print("Running K-means with k={}".format(k))
    tic = time.time()
    cls = KMeans(n_clusters=k, random_state=2516).fit(data)
    toc = time.time()
    t = np.round(toc - tic)
    ari = adjusted_rand_score(cls.labels_, label)
    sil = silhouette_score(data, cls.labels_)
    print("ARI: {}\tSil: {}".format(np.round(ari, 4), np.round(sil, 4)))
    print("Elapsed time: {}s".format(t))
    return cls.labels_, ari, sil


def run_k_means(data_dict, t_sne_embedding, misc, model=None, epoch=None):
    if not os.path.exists(misc):
        print("Directory {} created.".format(misc))
        os.mkdir(misc)
    k_means_cls_result = {}
    for k in K_MEANS_DIM:
        k_means_labels, ari, sil = k_means_cls(data_dict['data'], data_dict['label'], k)
        k_means_unique_label = np.unique(k_means_labels)
        for label in k_means_unique_label:
            plt.scatter(t_sne_embedding[k_means_labels == label, 0], t_sne_embedding[k_means_labels == label, 1], s=3,
                        label=label)
        plt.legend(loc="upper right")
        plt.title("K-means: k={} ARI={} Sil={}".format(k, np.round(ari, 4), np.round(sil, 4)))
        if model is None:
            plt.savefig(os.path.join(misc, "kmeans_{}.pdf".format(k)), dpi=400)
        else:
            plt.savefig(os.path.join(misc, "kmeans_{}_{}_{}.pdf".format(model, epoch, k)), dpi=400)
        plt.clf()
        k_means_cls_result.update({k: k_means_labels, 'ARI': ari, 'Sil': sil})

    if model is not None:
        with open(os.path.join(misc, 'k_means_{}_{}.pkl'.format(model, epoch)), 'wb') as f:
            pkl.dump(k_means_cls_result, f)
    else:
        with open(os.path.join(misc, 'k_means.pkl'), 'wb') as f:
            pkl.dump(k_means_cls_result, f)

    return k_means_cls_result


def run_decan(data, embedding, label, t_sne_embedding, misc, visualization, model, epoch, eps=6, min_samples=50):
    cls = DBSCAN(eps=eps, min_samples=min_samples)
    cls.fit(embedding)
    unique_label = np.unique(cls.labels_)
    ari = adjusted_rand_score(cls.labels_, label)
    sil = silhouette_score(data, cls.labels_)

    # On original T-SNE
    for lb in unique_label:
        plt.scatter(t_sne_embedding[cls.labels_ == lb, 0], t_sne_embedding[cls.labels_ == lb, 1], s=3, label=lb)
    plt.legend(loc="upper right")
    plt.title("DBSCAN: k={} ARI={} Sil={}".format(len(unique_label), np.round(ari, 4), np.round(sil, 4)))
    plt.savefig(os.path.join(misc, "DBSCAN_{}_{}.pdf".format(model.lower(), epoch)), dpi=400)
    plt.clf()
    print("DBSCAN finds {} clusters. ARI={}, Silhouette={}.".format(len(unique_label), np.round(ari, 4), np.round(sil, 4)))

    # On reduced T-SNE
    for lb in unique_label:
        plt.scatter(embedding[cls.labels_ == lb, 0], embedding[cls.labels_ == lb, 1], s=3, label=lb)
    plt.legend(loc="upper right")
    plt.title("DBSCAN: k={} ARI={} Sil={}".format(len(unique_label), np.round(ari, 4), np.round(sil, 4)))
    plt.savefig(os.path.join(visualization, "tsne_{}_{}_DBSCAN.pdf".format(model.lower(), epoch)), dpi=400)
    plt.clf()

    # Save
    if model is not None:
        with open(os.path.join(misc, 'DBSCAN_{}_{}.pkl'.format(model, epoch)), 'wb') as f:
            pkl.dump(cls.labels_, f)

    return cls.labels_


def run_t_sne(data_dict, tsne, misc=None, cache_name="tsne1.pkl", epoch=None, model=None, sets=None):
    if not os.path.exists(tsne):
        print("Directory {} created.".format(tsne))
        os.mkdir(tsne)

    if not os.path.isfile(os.path.join(tsne, cache_name)):
        t_sne_embedding = t_sne_visualize(data_dict['data'], data_dict['label'], misc, epoch=epoch, model=model, sets=sets)
        with open(os.path.join(tsne, cache_name), 'wb') as f:
            pkl.dump(t_sne_embedding, f)
    else:
        print("T-SNE cache found at {}".format(os.path.join(tsne, cache_name)))
        with open(os.path.join(tsne, cache_name), 'rb') as f:
            t_sne_embedding = pkl.load(f)

    return t_sne_embedding


if __name__ == "__main__":
    arg = parse_args()
    data_dict, _ = load_data(arg.mm, arg.np, arg.cache, arg.path, arg.write_cache, arg.skip_row, arg.skip_col, arg.seps,
                             arg.transpose, arg.label, arg.path_label, arg.col_name)

    t_sne_embedding = run_t_sne(data_dict, arg.tsne, arg.misc)
    k_means_cls_result = run_k_means(data_dict, t_sne_embedding, arg.misc)

