#   -*- coding: utf-8 -*-
#
#   analyze.py
#   
#   Developed by Tianyi Liu on 2020-04-01 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""

from utils import load_mat, load_label, Logger
from eval import adjusted_rand_score, silhouette_score
from cfgs import *


import os
import argparse
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        dest="data_path",
                        help="Path to data")
    parser.add_argument('--label_path',
                        dest="label_path",
                        help="Path to label")
    args = parser.parse_args()
    return args


def linear_correlation(embedding, cluster):
    clf = LogisticRegression()
    clf.fit(embedding, cluster)
    return clf.score


def t_sne_visualize(data, labels, cls_path, plot=False, epoch=None, model=None, sets=None, n_component=2):
    print(">>> Running T-SNE embedding")
    if not os.path.exists(cls_path):
        print(">>> Directory {} created.".format(cls_path))
        os.mkdir(cls_path)
    tic = time.time()
    t_sne_embedding = TSNE(n_components=n_component, n_jobs=T_SNE_N_JOB).fit_transform(data)
    toc = time.time()
    print("T-SNE takes {} s".format(toc - tic, 4))
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
    if label is not None:
        ari = adjusted_rand_score(cls.labels_, label)
        try:
            sil = silhouette_score(data, cls.labels_) if original_data is None else silhouette_score(original_data, cls.labels_)
        except ValueError: # Dummy
            sil = -2
        print("\tStats: ARI: {}\tSilhouette: {}".format(np.round(ari, 4), np.round(sil, 4)))
    else: # Dummy
        ari = -2
        sil = -2
    print("\tElapsed time: {}s".format(t))
    return cls.labels_, ari, sil, t


def run_k_means(output_embedding, label, origin_data, t_sne_embedding, visualization, cls_path, model=None,
                epoch=None, emb_type="tsne", logger=None):
    if not os.path.exists(visualization):
        print(">>> Directory {} created.".format(visualization))
        os.mkdir(visualization)
    k_means_cls_result = {}
    for k in K_MEANS_DIM:
        k_means_labels, ari, sil, t = k_means_cls(output_embedding, label, k, original_data=origin_data)
        k_means_unique_label = np.unique(k_means_labels)
        for lb in k_means_unique_label:
            plt.scatter(t_sne_embedding[k_means_labels == lb, 0], t_sne_embedding[k_means_labels == lb, 1], s=3,
                        label=lb)
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


def run_dbscan(output_embedding, label, origin_data, t_sne_embedding, cls_path, visualization, model, epoch,
               emb_type="tsne", eps=5, min_samples=20, logger=None):
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
    print("\tDBSCAN finds {} clusters with embedding provided.".format(len(unique_label)))

    if label is not None:
        ari = adjusted_rand_score(cls.labels_, label)
        try:
            sil = silhouette_score(origin_data, cls.labels_)
        except ValueError: # Dummy
            sil = -2
        print("\tStats: ARI={}, Silhouette={}.".format(np.round(ari, 4), np.round(sil, 4)))
    else: # Dummy
        ari = -2
        sil = -2

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
        plt.savefig(os.path.join(visualization, "DBSCAN_{}_{}_{}_Direct.pdf".format(model.lower(), epoch, emb_type)),
                    dpi=400)
        plt.clf()

    # Save
    if model is not None:
        with open(os.path.join(cls_path, "DBSCAN_{}_{}.pkl".format(model, epoch)), 'wb') as f:
            pkl.dump(cls.labels_, f)

    return cls.labels_


def run_t_sne(data, label, cache_path, cls_path=None, cache_name="tsne.pkl", epoch=None, model=None, sets=None, n_component=2, rtn_time=False):
    if not os.path.exists(cache_path):
        print(">>> Directory {} created.".format(cache_path))
        os.mkdir(cache_path)

    if not os.path.isfile(os.path.join(cache_path, cache_name)):
        t_sne_embedding = t_sne_visualize(data, label, cls_path, epoch=epoch, model=model, sets=sets, n_component=n_component)
        with open(os.path.join(cache_path, cache_name), 'wb') as f:
            pkl.dump(t_sne_embedding, f)
    else:
        print("T-SNE cache found at {}".format(os.path.join(cache_path, cache_name)))
        with open(os.path.join(cache_path, cache_name), 'rb') as f:
            t_sne_embedding = pkl.load(f)

    return t_sne_embedding


def run_dr(data, method, cache_path, cache_name, n_component):
    if not os.path.exists(cache_path):
        print(">>> Directory {} created.".format(cache_path))
        os.mkdir(cache_path)

    if not os.path.isfile(os.path.join(cache_path, cache_name)):
        tic = time.time()
        if method == "PCA":
            embedding = PCA(n_components=n_component).fit_transform(data)
        elif method == "TSNE":
            embedding = TSNE(n_components=n_component, n_jobs=T_SNE_N_JOB).fit_transform(data)
        else:
            raise NotImplementedError
        toc = time.time()
        t = toc - tic
        print("DR with {} takes {} s".format(method, np.round(t, 4)))
        with open(os.path.join(cache_path, cache_name), 'wb') as f:
            pkl.dump(embedding, f)
    else:
        print("DR cache with {} found at {}".format(method, os.path.join(cache_path, cache_name)))
        with open(os.path.join(cache_path, cache_name), 'rb') as f:
            embedding = pkl.load(f)
        t = 0

    return embedding, t


if __name__ == "__main__":
    arg = parse_args()
    print("========Call with Arguments========")
    print(arg)

    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
        print(">>> Directory {} created.".format(RESULTS_PATH))

    if not os.path.exists(BCM_PATH):
        os.mkdir(BCM_PATH)
        print(">>> Directory {} created.".format(BCM_PATH))

    print("\n========Reading Data========")
    data, _= load_mat(arg.data_path, False, 1, 1, ',', True, False, None, None)
    label = load_label(arg.label_path, ',', '0')
    data = data["data"]
    k_means_logger = Logger(LOG_PATH, "Benchmark_K_MEANS.log", benchmark_logger=True)
    dbscan_logger = Logger(LOG_PATH, "Benchmark_DBSCAN.log", benchmark_logger=True)

    k_means_results = {}
    dbscan_results = {}

    print("\n========Benchmarking========")

    for dim in DR_DIM:
        for method in ["PCA", "TSNE"]:
            if method == "TSNE" and dim != 2:
                continue
            print("\n>>> Running Experiments: DR={} Dim={}".format(method, dim))
            embedding, t_dr = run_dr(data, method, "./cache", "{}_{}.pkl".format(method, dim), dim)
            for k in K_MEANS_DIM:
                print(">>> Running KMeans with k={}".format(k))
                tic = time.time()
                cls = KMeans(n_clusters=k).fit(embedding)
                toc = time.time()
                t_cls = toc - tic
                ari = adjusted_rand_score(label, cls.labels_)
                try:
                    sil = silhouette_score(data, cls.labels_)
                except ValueError:
                    sil = -2
                print("\tStats: ARI: {}\tSilhouette: {}".format(np.round(ari, 4), np.round(sil, 4)))
                k_means_logger.log("{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}".format(dim, method, len(np.unique(cls.labels_)), ari, sil, t_dr + t_cls))
            print(">>> Running DBSCAN")
            tic = time.time()
            cls = DBSCAN(eps=5, min_samples=20).fit(embedding)
            toc = time.time()
            t_cls = toc - tic
            ari = adjusted_rand_score(label, cls.labels_)
            try:
                sil = silhouette_score(data, cls.labels_)
            except ValueError:
                sil = -2
            print("\tDBSCAN finds {} clusters".format(len(np.unique(cls.labels_))))
            print("\tStats: ARI: {}\tSilhouette: {}".format(np.round(ari, 4), np.round(sil, 4)))
            dbscan_logger.log("{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}".format(dim, method, len(np.unique(cls.labels_)), ari, sil, t_dr + t_cls))

