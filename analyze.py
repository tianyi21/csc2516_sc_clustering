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
    parser.add_argument('--label_path',
                        dest="label_path",
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
    parser.add_argument('--cache_path',
                        dest="cache_path",
                        default="./cache/",
                        help="TSNE embedding cache")
    args = parser.parse_args()
    return args


def linear_correlation(embedding, cluster):
    clf = LogisticRegression()
    clf.fit(embedding, cluster)
    return clf.score


def t_sne_visualize(data, labels, cls_path, plot=False, epoch=None, model=None, sets=None):
    print(">>> Running T-SNE embedding")
    if not os.path.exists(cls_path):
        print(">>> Directory {} created.".format(cls_path))
        os.mkdir(cls_path)
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
            plt.savefig(os.path.join(cls_path, "TSNE.pdf"), dpi=400)
        else:
            plt.title("T-SNE with {} Epoch {}".format(model.upper(), epoch))
            plt.savefig(os.path.join(cls_path, "TSNE_{}_{}.pdf".format(model, epoch)), dpi=400)
    else:
        plt.title("T-SNE of {} set".format(sets))
        plt.savefig(os.path.join(cls_path, "TSNE_{}.pdf".format(sets)), dpi=400)
    if plot:
        plt.show()
    plt.clf()
    return t_sne_embedding


def k_means_cls(data, label, k, original_data=None):
    print(">>> Running KMeans with k={}".format(k))
    tic = time.time()
    cls = KMeans(n_clusters=k, random_state=2516).fit(data)
    toc = time.time()
    t = np.round(toc - tic, 4)
    ari = adjusted_rand_score(cls.labels_, label)
    try:
        sil = silhouette_score(data, cls.labels_) if original_data is None else silhouette_score(original_data, cls.labels_)
    except ValueError:
        sil = -2 # Dummy
    print("\tStats: ARI: {}\tSilhouette: {}".format(np.round(ari, 4), np.round(sil, 4)))
    print("\tElapsed time: {}s".format(t))
    return cls.labels_, ari, sil, t


def run_k_means(data_dict, data, t_sne_embedding, visualization, cls_path, model=None, epoch=None, emb_type="tsne", logger=None):
    if not os.path.exists(visualization):
        print(">>> Directory {} created.".format(visualization))
        os.mkdir(visualization)
    k_means_cls_result = {}
    for k in K_MEANS_DIM:
        k_means_labels, ari, sil, t = k_means_cls(data_dict['data'], data_dict['label'], k, original_data=data)
        k_means_unique_label = np.unique(k_means_labels)
        for label in k_means_unique_label:
            plt.scatter(t_sne_embedding[k_means_labels == label, 0], t_sne_embedding[k_means_labels == label, 1], s=3,
                        label=label)
        plt.legend(loc="upper right")
        plt.title("KMeans: k={} ARI={} Sil={}".format(k, np.round(ari, 4), np.round(sil, 4)))
        if model is None:
            plt.savefig(os.path.join(visualization, "KMeans_{}.pdf".format(k)), dpi=400)
        else:
            plt.savefig(os.path.join(visualization, "KMeans_{}_{}_{}_{}.pdf".format(model, epoch, k, emb_type)), dpi=400)
        plt.clf()
        k_means_cls_result.update({k: k_means_labels, 'ARI': ari, 'Sil': sil})
        if logger is not None:
            logger.log("{}\t\t{}\t\t{}\t\t{}\t\t{}".format(epoch, k, np.round(ari, 8), np.round(sil, 8), t))

    if model is not None:
        with open(os.path.join(cls_path, 'KMeans_{}_{}.pkl'.format(model, epoch)), 'wb') as f:
            pkl.dump(k_means_cls_result, f)
    else:
        with open(os.path.join(cls_path, 'KMeans.pkl'), 'wb') as f:
            pkl.dump(k_means_cls_result, f)
    return k_means_cls_result


def run_dbscan(data, label, output_embedding, t_sne_embedding, cls_path, visualization, model, epoch, emb_type="tsne", eps=5, min_samples=20, logger=None):
    print(">>> Running DBSCAN")
    if not os.path.exists(cls_path):
        print(">>> Directory {} created.".format(cls_path))
        os.mkdir(cls_path)

    cls = DBSCAN(eps=eps, min_samples=min_samples)
    tic = time.time()
    cls.fit(output_embedding)
    toc = time.time()
    t = np.round(toc - tic, 4)
    unique_label = np.unique(cls.labels_)
    ari = adjusted_rand_score(cls.labels_, label)
    try:
        sil = silhouette_score(data, cls.labels_)
    except ValueError:
        sil = -2 # Dummy

    print(
        "\tDBSCAN finds {} clusters with embedding provided.\n\t Stats: ARI={}, Silhouette={}.".format(
            len(unique_label),
            np.round(ari, 4),
            np.round(sil, 4)))
    print("\tElapsed time: {}s".format(t))

    if logger is not None:
        logger.log("{}\t\t{}\t\t{}\t\t{}\t\t{}".format(epoch, len(unique_label), np.round(ari, 8), np.round(sil, 8), t))

    # On original T-SNE
    for lb in unique_label:
        plt.scatter(t_sne_embedding[cls.labels_ == lb, 0], t_sne_embedding[cls.labels_ == lb, 1], s=3, label=lb)
    plt.legend(loc="upper right")
    plt.title("DBSCAN: k={} ARI={} Sil={}".format(len(unique_label), np.round(ari, 4), np.round(sil, 4)))
    plt.savefig(os.path.join(visualization, "DBSCAN_{}_{}_{}.pdf".format(model.lower(), epoch, emb_type)), dpi=400)
    plt.clf()

    # On current embedding if current embedding is 2-D
    if output_embedding.shape[1] == 2:
        for lb in unique_label:
            plt.scatter(output_embedding[cls.labels_ == lb, 0], output_embedding[cls.labels_ == lb, 1], s=3, label=lb)
        plt.legend(loc="upper right")
        plt.title("DBSCAN: k={} ARI={} Sil={}".format(len(unique_label), np.round(ari, 4), np.round(sil, 4)))
        plt.savefig(os.path.join(visualization, "DBSCAN_{}_{}_{}_Direct.pdf".format(model.lower(), epoch, emb_type)), dpi=400)
        plt.clf()

    # Save
    if model is not None:
        with open(os.path.join(cls_path, "DBSCAN_{}_{}.pkl".format(model, epoch)), 'wb') as f:
            pkl.dump(cls.labels_, f)

    return cls.labels_


def run_t_sne(data_dict, cache_path, cls_path=None, cache_name="tsne.pkl", epoch=None, model=None, sets=None):
    if not os.path.exists(cache_path):
        print(">>> Directory {} created.".format(cache_path))
        os.mkdir(cache_path)

    if not os.path.isfile(os.path.join(cache_path, cache_name)):
        t_sne_embedding = t_sne_visualize(data_dict['data'], data_dict['label'], cls_path, epoch=epoch, model=model, sets=sets)
        with open(os.path.join(cache_path, cache_name), 'wb') as f:
            pkl.dump(t_sne_embedding, f)
    else:
        print("T-SNE cache found at {}".format(os.path.join(cache_path, cache_name)))
        with open(os.path.join(cache_path, cache_name), 'rb') as f:
            t_sne_embedding = pkl.load(f)

    return t_sne_embedding


if __name__ == "__main__":
    arg = parse_args()
    data_dict, _ = load_data(arg.mm, arg.np, arg.cache, arg.path, arg.write_cache, arg.skip_row, arg.skip_col, arg.seps,
                             arg.transpose, arg.label, arg.label_path, arg.col_name)

    t_sne_embedding = run_t_sne(data_dict, arg.cache_path, CLS_PATH)
    # k_means_cls_result = run_k_means(data_dict, t_sne_embedding, CLS_PATH)

