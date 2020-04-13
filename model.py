#   -*- coding: utf-8 -*-
#
#   model.py
#
#   Developed by Tianyi Liu on 2020-03-05 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


from eval import compute_loss
from cfgs import LOSS_WEIGHT

import numpy as np
import torch
import torch.nn as nn


class _AESC(nn.Module):

    # dim = [8000, 2048, 1024, 512, 256, 64, 2]
    dim = [8000, 2048, 512, 256, 64, 32, 2]
    conv_para = {'kernel': 9,
                 'in_ch': 1,
                 'out_ch_1': 4,
                 'out_ch_2': 8}
    dyn = [32,16]

    def __init__(self):
        super(_AESC, self).__init__()
        self.enc_1_fc = nn.Sequential(nn.Linear(self.dim[0], self.dim[1]), nn.ReLU())
        self.enc_2_fc = nn.Sequential(nn.Linear(self.dim[1], self.dim[2]), nn.BatchNorm1d(self.dim[2]), nn.Dropout(0.5), nn.ReLU())
        self.enc_3_fc = nn.Sequential(nn.Linear(self.dim[2], self.dim[3]), nn.Dropout(0.5), nn.ReLU())
        self.enc_4_fc = nn.Sequential(nn.Linear(self.dim[3], self.dim[4]), nn.Dropout(0.5), nn.ReLU())
        self.enc_4_conv = nn.Sequential(
            nn.Conv1d(self.conv_para['in_ch'], self.conv_para['out_ch_1'], self.conv_para['kernel'],
                      padding=self.conv_para['kernel'] // 2),
            nn.BatchNorm1d(self.conv_para['out_ch_1']),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.enc_5_conv = nn.Sequential(
            nn.Conv1d(self.conv_para['out_ch_1'], self.conv_para['out_ch_2'], self.conv_para['kernel'],
                      padding=self.conv_para['kernel'] // 2),
            nn.BatchNorm1d(self.conv_para['out_ch_2']),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.enc_6_cls = nn.Sequential(nn.Linear(self.dim[4], self.dim[5]), nn.BatchNorm1d(self.dim[5]), nn.Dropout(0.5), nn.ReLU())
        self.enc_7_vis = nn.Sequential(nn.Linear(self.dim[5], self.dim[6]), nn.Dropout(0.5), nn.ReLU())
        self.dec_7_vis = nn.Sequential(nn.Linear(self.dim[6], self.dim[5]), nn.Dropout(0.5), nn.ReLU())
        self.dec_6_cls = nn.Sequential(nn.Linear(self.dim[5], self.dim[4]), nn.BatchNorm1d(self.dim[4]), nn.Dropout(0.5), nn.ReLU())
        self.dec_5_deconv = nn.Sequential(
            nn.ConvTranspose1d(self.conv_para['out_ch_2'], self.conv_para['out_ch_1'], self.conv_para['kernel'],
                               padding=self.conv_para['kernel'] // 2),
            nn.BatchNorm1d(self.conv_para['out_ch_1']),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Upsample(scale_factor=4))
        self.dec_4_deconv = nn.Sequential(
            nn.ConvTranspose1d(self.conv_para['out_ch_1'], self.conv_para['in_ch'], self.conv_para['kernel'],
                               padding=self.conv_para['kernel'] // 2),
            nn.BatchNorm1d(self.conv_para['in_ch']),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Upsample(scale_factor=2))
        self.dec_4_fc = nn.Sequential(nn.Linear(self.dim[4], self.dim[3]), nn.Dropout(0.5), nn.ReLU())
        self.dec_3_fc = nn.Sequential(nn.Linear(self.dim[3], self.dim[2]), nn.Dropout(0.5), nn.ReLU())
        self.dec_2_fc = nn.Sequential(nn.Linear(self.dim[2], self.dim[1]), nn.BatchNorm1d(self.dim[1]), nn.Dropout(0.5), nn.ReLU())
        self.dec_1_fc = nn.Sequential(nn.Linear(self.dim[1], self.dim[0]), nn.ReLU())

        # self.cls_fc_1 = nn.Sequential(nn.Linear(self.dim[4], self.dyn[0]), nn.Dropout(0.5), nn.ReLU())
        # self.cls_fc_2 = nn.Sequential(nn.Linear(self.dyn[0], self.dyn[1]), nn.Dropout(0.5), nn.Softmax(dim=1))

    def forward(self, x):
        # ENC: MLP
        x_enc = self.enc_6_cls(self.enc_4_fc(self.enc_3_fc(self.enc_2_fc(self.enc_1_fc(x)))))
        # ENC: CONV
        # x = self.enc_5_conv(self.enc_4_conv(x_enc.view(x_enc.size(0), self.conv_para['in_ch'], x_enc.size(1))))
        # ENC: MLP + + Residual Connection
        # vis = (self.enc_6_cls(x_enc + x.view(x.size(0), -1)))
        # DEC: CLF Layer
        # DEC: DECONV
        # x = self.dec_4_deconv(self.dec_5_deconv(x_dec.view(x_dec.size(0), self.conv_para['out_ch_2'], -1)))
        # DEC: MLP
        y = self.dec_1_fc(self.dec_2_fc(self.dec_3_fc(self.dec_4_fc(self.dec_6_cls(x_enc)))))

        # Clustering
        # q = self.cls_fc_2(self.cls_fc_1(x_enc))

        return y, x_enc

    @classmethod
    def adjust_dim(cls, dim):
        cls.dim[0] = dim

    @classmethod
    def adjust_dyn(cls, dim, idx):
        if idx != 0 or idx != 1:
            print("Invalid index provided. No change made.")
        else:
            cls.dyn[idx] = dim


class _VAESC(nn.Module):
    # dim = [8000, 2048, 1024, 512, 256, 128, 64]
    dim = [8000, 2048, 1024, 512, 256, 64, 32, 16, 8, 4, 2]
    device = "cuda"

    def __init__(self):
        super(_VAESC, self).__init__()
        self.enc_1_fc = nn.Sequential(nn.Linear(self.dim[0], self.dim[1]), nn.ReLU())
        self.enc_2_fc = nn.Sequential(nn.Linear(self.dim[1], self.dim[2]), nn.BatchNorm1d(self.dim[2]), nn.Dropout(0.5), nn.ReLU())
        self.enc_3_fc = nn.Sequential(nn.Linear(self.dim[2], self.dim[3]), nn.Dropout(0.5), nn.ReLU())
        self.enc_4_fc = nn.Sequential(nn.Linear(self.dim[3], self.dim[4]), nn.BatchNorm1d(self.dim[4]), nn.Dropout(0.5), nn.ReLU())
        self.enc_5_fc = nn.Sequential(nn.Linear(self.dim[4], self.dim[5]), nn.Dropout(0.5), nn.ReLU())
        self.enc_6_fc = nn.Sequential(nn.Linear(self.dim[5], self.dim[6]), nn.BatchNorm1d(self.dim[6]), nn.Dropout(0.5), nn.ReLU())

        self.enc_7_vae_mu = nn.Sequential(nn.Linear(self.dim[6], self.dim[7]), nn.BatchNorm1d(self.dim[7]), nn.ReLU())
        self.enc_7_vae_std = nn.Sequential(nn.Linear(self.dim[6], self.dim[7]), nn.BatchNorm1d(self.dim[7]), nn.ReLU())
        self.enc_8_vae_mu = nn.Sequential(nn.Linear(self.dim[7], self.dim[8]), nn.ReLU())
        self.enc_8_vae_std = nn.Sequential(nn.Linear(self.dim[7], self.dim[8]), nn.ReLU())
        self.enc_9_vae_mu = nn.Sequential(nn.Linear(self.dim[8], self.dim[9]), nn.BatchNorm1d(self.dim[9]), nn.ReLU())
        self.enc_9_vae_std = nn.Sequential(nn.Linear(self.dim[8], self.dim[9]), nn.BatchNorm1d(self.dim[9]), nn.ReLU())
        self.enc_10_vae_mu = nn.Sequential(nn.Linear(self.dim[9], self.dim[10]))
        self.enc_10_vae_std = nn.Sequential(nn.Linear(self.dim[9], self.dim[10]))

        self.dec_10_vae = nn.Sequential(nn.Linear(self.dim[10], self.dim[9]), nn.ReLU())
        self.dec_9_vae = nn.Sequential(nn.Linear(self.dim[9], self.dim[8]), nn.BatchNorm1d(self.dim[8]), nn.ReLU())
        self.dec_8_vae = nn.Sequential(nn.Linear(self.dim[8], self.dim[7]), nn.ReLU())
        self.dec_7_vae = nn.Sequential(nn.Linear(self.dim[7], self.dim[6]), nn.BatchNorm1d(self.dim[6]), nn.ReLU())

        self.dec_6_fc = nn.Sequential(nn.Linear(self.dim[6], self.dim[5]), nn.BatchNorm1d(self.dim[5]), nn.Dropout(0.5), nn.ReLU())
        self.dec_5_fc = nn.Sequential(nn.Linear(self.dim[5], self.dim[4]), nn.Dropout(0.5), nn.ReLU())
        self.dec_4_fc = nn.Sequential(nn.Linear(self.dim[4], self.dim[3]), nn.BatchNorm1d(self.dim[3]), nn.Dropout(0.5), nn.ReLU())
        self.dec_3_fc = nn.Sequential(nn.Linear(self.dim[3], self.dim[2]), nn.Dropout(0.5), nn.ReLU())
        self.dec_2_fc = nn.Sequential(nn.Linear(self.dim[2], self.dim[1]), nn.BatchNorm1d(self.dim[1]), nn.Dropout(0.5), nn.ReLU())
        self.dec_1_fc = nn.Sequential(nn.Linear(self.dim[1], self.dim[0]), nn.ReLU())

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu = self.enc_10_vae_mu(self.enc_9_vae_mu(self.enc_8_vae_mu(self.enc_7_vae_mu(h))))
        logvar = self.enc_10_vae_std(self.enc_9_vae_std(self.enc_8_vae_std(self.enc_7_vae_std(h))))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        h = self.enc_6_fc(self.enc_5_fc(self.enc_4_fc(self.enc_3_fc(self.enc_2_fc(self.enc_1_fc(x))))))
        z, mu, logvar = self.bottleneck(h)
        z = self.dec_7_vae(self.dec_8_vae(self.dec_9_vae(self.dec_10_vae(z))))
        y = self.dec_1_fc(self.dec_2_fc(self.dec_3_fc(self.dec_4_fc(self.dec_5_fc(self.dec_6_fc(z))))))
        return y, mu, logvar

    @classmethod
    def adjust_cls_par(cls, dim, device):
        cls.dim[0] = dim
        cls.device = device


def vl_loop(model, loader, sel, return_sel):
    epoch_loss, epoch_loss_w, epoch_loss_kl = 0, 0, 0
    embedding = []
    with torch.no_grad():
        for _, data_batch in enumerate(loader):
            try:
                (data, label) = data_batch
            except ValueError:
                (data) = data_batch[:]

            if sel.lower() == 'ae':
                y, x_enc, q = model(data)
                embedding.extend(x_enc.detach().cpu().numpy())
                loss_w = compute_loss('ae', data, y, x_enc)

                loss_w *= LOSS_WEIGHT["mse"]
                loss_vl = loss_w
                epoch_loss += loss_vl.item() * len(data)
                epoch_loss_w += loss_w.item() * len(data)

            elif sel.lower() == 'vae':
                y, mu, logvar = model(data)
                embedding.extend(mu.detach().cpu().numpy())
                loss_w, loss_kl = compute_loss('vae', data, y, mu=mu, logvar=logvar)

                loss_w *= LOSS_WEIGHT["mse"]
                loss_kl *= LOSS_WEIGHT["kl"]
                loss_vl = loss_w + loss_kl
                epoch_loss += loss_vl.item() * len(data)
                epoch_loss_w += loss_w.item() * len(data)
                epoch_loss_kl += loss_kl.item() * len(data)

    if return_sel == 'vl':
        print("\tValidation Loss: {}".format(epoch_loss / (len(loader.dataset))))
        if sel.lower() == 'vae':
            print("\t\tLoss_w: {}\tLoss_kl: {}".format(epoch_loss_w / (len(loader.dataset)),
                                                   epoch_loss_kl / len(loader.dataset)))
            return epoch_loss / (len(loader.dataset)), epoch_loss_w / (len(loader.dataset)), epoch_loss_kl / (len(loader.dataset))
        else:
            print("\t\tLoss_w: {}".format(epoch_loss_w / (len(loader.dataset))))
            return epoch_loss, epoch_loss_w
    elif return_sel == 'vis':
        try:
            (data, label) = loader.dataset[:]
            return np.array(embedding), label.cpu().numpy()
        except ValueError:
            return np.array(embedding)


def learning_rate_decay(optimizer, decay):
    if optimizer.param_groups[0]['lr'] <= 5e-5:
        return None
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


