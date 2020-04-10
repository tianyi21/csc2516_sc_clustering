#   -*- coding: utf-8 -*-
#
#   test.py
#
#   Developed by Tianyi Liu on 2020-03-05 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""

from cfgs import *
from utils import *
from eval import compute_loss
from model import _ConvAE
from analyze import t_sne_visualize

import argparse



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
    parser.add_argument('--cuda',
                        dest="cuda",
                        action="store_false",
                        help="Set -> GPU support")
    parser.add_argument('--mgpu',
                        dest="mgpu",
                        action="store_true",
                        help="Set -> Multiple GPU support")
    parser.add_argument('--tr',
                        dest="tr",
                        type=float,
                        default=0.8,
                        help="Training split ratio")
    parser.add_argument('--vl',
                        dest="vl",
                        type=float,
                        default=0.1,
                        help="Validation split ratio")
    parser.add_argument('--ts',
                        dest="ts",
                        type=float,
                        default=0.1,
                        help="Testing split ratio")
    parser.add_argument('--bs',
                        dest="batch_size",
                        type=int,
                        default=128,
                        help="Batch size for training")
    parser.add_argument('--model',
                        dest="model",
                        default="./cache/Trained_model.pth",
                        help="Model for evaluation")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    print("Call with arguments: {}".format(arg))

    device = "cuda" if arg.cuda else "cpu"

    # Read data, label
    data_dict, dim = load_data(arg.mm, arg.np, arg.cache, arg.path, arg.write_cache, arg.skip_row, arg.skip_col,
                               arg.seps, arg.transpose, arg.label, arg.path_label, arg.col_name)
    tr_loader, vl_loader, ts_loader = split_data(data_dict, arg.label, device, arg.batch_size, arg.tr, arg.vl,
                                                 arg.ts)

    # Def model
    _ConvAE.adjust_dim(dim)
    conv_ae = _ConvAE().to(device)

    # Test loop
    epoch_ts_loss = 0
    embedding = []
    for step, data_batch_ts in enumerate(ts_loader):

        try:
            (data_ts, label_ts) = data_batch_ts
        except ValueError:
            (data_ts) = data_batch_ts[0]

        conv_ae.eval()
        conv_ae.zero_grad()
        y, x_enc, q, p = conv_ae(data_ts)
        loss_w, loss_kl = compute_loss(data_ts, y, q, p)

        loss_w *= LOSS_WEIGHT["mse"]
        loss_kl *= LOSS_WEIGHT["kl"]
        loss = loss_w + loss_kl

        epoch_ts_loss += loss.item() * len(data_ts)

        embedding.extend(x_enc.detach().cpu().numpy())

        # Print
        if (step + 1) % 50 == 0:
            print("\tStep: {}/{}, Loss: {}".format(step + 1, int(np.ceil(len(ts_loader.dataset) / arg.batch_size)),
                                                   loss.item()))
            print("\t\tLoss_w: {}\tLoss_kl: {}".format(loss_w.item(), loss_kl.item()))
            print("\t\tLoss_w: {}".format(loss_w.item()))

    print("Averaged Test Loss: {}\n".format(epoch_ts_loss / len(ts_loader.dataset)))

    t_sne_visualize(embedding, None, "./")
