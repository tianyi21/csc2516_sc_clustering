#   -*- coding: utf-8 -*-
#
#   cfgs.py
#   
#   Developed by Tianyi Liu on 2020-04-04 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import os
import numpy as np


K_MEANS_DIM = np.arange(5, 12)

LOSS_WEIGHT = {"mse": 1,
               "kl": 1}

VAL_STEP = 50

VIS_EPOCH = 50

SAVE_EPOCH = 100

NUMPY_SEED = 2516

TORCH_SEED = 2516

RESULTS_PATH = "./results"

VIS_PATH = os.path.join(RESULTS_PATH, "visualization")

CLS_PATH = os.path.join(RESULTS_PATH, "clustering")

LOG_PATH = os.path.join(RESULTS_PATH, "logs")

# For Benchmarking
DR_METHOD = ["TSNE", "PCA"]

DR_DIM = [64, 32, 16, 8, 4, 2]

BCM_PATH = os.path.join(RESULTS_PATH, "benchmark")
