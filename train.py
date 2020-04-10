#   -*- coding: utf-8 -*-
#
#   train.py
#
#   Developed by Tianyi Liu on 2020-03-05 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


from cfgs import *
from utils import *
from eval import compute_loss
from analyze import t_sne_visualize, run_t_sne, run_decan
from model import _AESC, _VAESC, learning_rate_decay, target_distribution, vl_loop

import argparse
import torch


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
    parser.add_argument('--subsample',
                        dest="sub",
                        type=float,
                        default=1,
                        help="Subsample a portion of dataset")
    parser.add_argument('--epoch_vis',
                        dest="epo_vis",
                        default=50,
                        type=int,
                        help="# epochs for visualization")
    parser.add_argument('--cuda',
                        dest="cuda",
                        action="store_false",
                        help="Set -> NO GPU support")
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
    parser.add_argument('--lr',
                        dest="lr",
                        type=float,
                        default=1e-4,
                        help="Initial learning rate")
    parser.add_argument('--lrd',
                        dest="lr_decay",
                        type=int,
                        default=10,
                        help="Learning rate decays after * epoch")
    parser.add_argument('--lrg',
                        dest="lr_gamma",
                        type=float,
                        default=0.1,
                        help="Learning rate decays gamma")
    parser.add_argument('--epoch',
                        dest="epoch",
                        type=int,
                        default=25,
                        help="Number of epoch")
    parser.add_argument('-s',
                        dest="store",
                        action="store_true",
                        help="Set -> Store trained model")
    parser.add_argument('--path_save',
                        dest="path_save",
                        default="./trained_model/",
                        help="Path to store trained model")
    parser.add_argument('-p',
                        dest="pretrain",
                        action="store_true",
                        help="Set -> Fine tune pre-trained model")
    parser.add_argument('--pretrain_path',
                        dest="pretrain_path",
                        default="./trained_model/",
                        help="Path to pre-trained model")
    parser.add_argument('--finetune_save_path',
                        dest="finetune_save_path",
                        default="./trained_model/finetune/",
                        help="Path to store fine-tuned model")
    parser.add_argument('--lrf',
                        dest="lrf",
                        type=float,
                        default=1e-5,
                        help="Learning rate for pre-trained parameters")
    parser.add_argument('--model',
                        dest="model",
                        default="ae",
                        help="Model: AE / VAE")
    parser.add_argument('--noise',
                        dest="noise",
                        default="dropout",
                        help="Simulate noise. dropout/gaussian/d+g/none")
    parser.add_argument('--dprob',
                        dest="dprob",
                        default=0.2,
                        type=float,
                        help="Bernoulli prob, i.e., prob to be a dropout")
    parser.add_argument('--gsig',
                        dest="gsig",
                        default=0.5,
                        type=float,
                        help="Sigma of Gaussian noise"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    print("Call with arguments: {}\n".format(arg))
    device = "cuda" if arg.cuda else "cpu"

    if arg.mgpu or arg.pretrain:
        raise NotImplementedError("Not yet support for multiple GPUs.")

    if arg.store:
        if not os.path.exists(arg.path_save):
            os.mkdir(arg.path_save)
            print("Directory {} created.".format(arg.path_save))

    # Read data, label
    data_dict, dim = load_data(arg.mm, arg.np, arg.cache, arg.path, arg.write_cache, arg.skip_row, arg.skip_col,
                               arg.seps, arg.transpose, arg.label, arg.path_label, arg.col_name)
    data_dict = add_noise(data_dict, arg.noise, arg.dprob, arg.gsig)

    tr_loader, vl_loader, ts_loader = split_data(data_dict, arg.label, device, arg.batch_size, arg.tr, arg.vl,
                                                 arg.ts, arg.sub)
    (data, label) = vl_loader.dataset[:]

    # Def model
    if arg.model.lower() == 'ae':
        _AESC.adjust_dim(dim)
        model = _AESC().to(device)
    elif arg.model.lower() == 'vae':
        _VAESC.adjust_cls_par(dim, device)
        model = _VAESC().to(device)
    else:
        raise NotImplementedError

    if arg.pretrain:
        raise NotImplementedError
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

    for epoch in range(arg.epoch):
        epoch_tr_loss = 0
        print("Epoch: {}/{}\tlr: {}".format(epoch + 1, arg.epoch, optimizer.param_groups[0]['lr']))

        # LR decay
        if (epoch + 1) % arg.lr_decay == 0 and epoch != 0:
            learning_rate_decay(optimizer, arg.lr_gamma)

        # Train loop
        for step, data_batch_tr in enumerate(tr_loader):
            model.train()
            try:
                (data_tr, label_tr) = data_batch_tr
            except ValueError:
                (data_tr) = data_batch_tr[0]

            if arg.model.lower() == 'ae':
                y, x_enc, q = model(data_tr)
                p = target_distribution(x_enc)
                loss_w = compute_loss('ae', data_tr, y, x_enc, p)

                loss_w *= LOSS_WEIGHT["mse"]
                loss = loss_w
                epoch_tr_loss += loss.item() * len(data_tr)

            elif arg.model.lower() == 'vae':
                y, mu, logvar = model(data_tr)
                loss_w, loss_kl = compute_loss('vae', data_tr, y, mu=mu, logvar=logvar)

                loss_w *= LOSS_WEIGHT["mse"]
                loss_kl *= LOSS_WEIGHT["kl"]
                loss = loss_w + loss_kl
                epoch_tr_loss += loss.item() * len(data_tr)

            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % VAL_STEP == 0:
                print("\tStep: {}/{}, Loss: {}".format(step + 1, int(np.ceil(len(tr_loader.dataset) / arg.batch_size)),
                                                       loss.item()))
                if arg.model.lower() == 'vae':
                    print("\t\tLoss_w: {}\tLoss_kl: {}".format(loss_w.item(), loss_kl.item()))
                else:
                    print("\t\tLoss_w: {}".format(loss_w.item()))

                # Validation
                model.eval()
                vl_loop(model, vl_loader, arg.model, 'vl')

                # Eval metric
                """
                if arg.label:
                    (data_vl_full, label_vl_full) = vl_loader.dataset[:]
                    ari_vl = cal_ari(np.array(clusters), label_vl_full)
                    print("\tARI: {}".format(ari_vl))
                else:
                    (data_vl_full) = vl_loader.dataset[:]
                #sil_vl = cal_silhouette(data_vl_full, clusters)
                #print("\tSilhouette: {}".format(sil_vl))
                print(torch.cat((data_vl[0, :8].unsqueeze(1), y[0, :8].unsqueeze(1)), dim=1))
                """

        print("Averaged Epoch Loss: {}\n".format(epoch_tr_loss / len(tr_loader.dataset)))

        # Visualize
        if (epoch + 1) % VIS_EPOCH == 0:
            model.eval()
            embedding, label = vl_loop(model, vl_loader, arg.model, 'vis')
            # T-SNE plot current embedding
            cur_t_sne = t_sne_visualize(embedding, label, "./visualization/", epoch=epoch + 1, model=arg.model.lower())
            try:
                (data, label) = vl_loader.dataset[:]
                # T-SNE plot of vl_data
                t_sne_embedding = run_t_sne({"data": data.cpu().numpy(), "label": label.cpu().numpy()}, "./cache",
                                            misc="./visualization",
                                            sets="Validation", cache_name="tsne_vl.pkl")
                run_decan(data.cpu().numpy(), cur_t_sne, label.cpu().numpy(), t_sne_embedding, "./misc",
                          "./visualization", arg.model, epoch + 1)
            except ValueError:
                raise NotImplementedError

        # Save model
        if (epoch + 1) % SAVE_EPOCH == 0 and arg.store:
            save_name = os.path.join(arg.path_save, 'Trained_{}_{}.pth'.format(arg.model.upper(), epoch + 1)) if not arg.pretrain \
                else os.path.join(arg.path_save, '{}_finetune_{}_.pth'.format(arg.pretrained_path.split('.')[0], epoch + 1))
            save_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'lr': optimizer.param_groups[0]['lr']}
            torch.save(save_dict, save_name)
            print("Saving model to {}\n".format(os.path.join(arg.path_save, save_name)))

