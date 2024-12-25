# ------------------------------------------------------------------------------------------------
# Integrated AutoEncoder
# Copyright (c) 2024 Fanglin Liu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------

import torch.nn as nn

from models.utils import get_activation
from models.Encoder import IntegratedAutoEncoder
from models.wtconv2d import WTConv2d
from models.DCN import DeformConv2d
from models.FPN import FPN, BiFPN
import torch.nn.init as init


__all__ = ['ResNet']


class ResNet(nn.Module):
    def __init__(self,
                 spatial_shapes,
                 channel,
                 dim_feedforward,
                 input_shape,
                 classes,
                 dropout,
                 activation,
                 num_encoder_layers,
                 num_head,
                 num_feat_levels,
                 enc_n_points,
                 normalize_before=False):
        super(ResNet, self).__init__()
        self.normalize_before = normalize_before

        self.dcn = DeformConv2d(inc=channel,
                                outc=channel,
                                kernel_size=2,
                                padding=1,
                                stride=1,
                                bias=False,
                                modulation=False)

        self.wt_conv2d = WTConv2d(in_channels=1,
                                  out_channels=1,
                                  base_kernel_size=(2, 5),
                                  wavelet_kernel_size=(1, 3),
                                  stride=1,
                                  bias=False,
                                  wt_levels=1,
                                  wt_type='haar')

        self.bn1 = nn.BatchNorm2d(num_features=1)
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm2d(num_features=channel)

        self.inae = IntegratedAutoEncoder(spatial_shapes=spatial_shapes,
                                          d_model=channel,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation,
                                          num_encoder_layers=num_encoder_layers,
                                          num_head=num_head,
                                          num_feat_levels=num_feat_levels,
                                          enc_n_points=enc_n_points,
                                          normalize_before=normalize_before)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(channel * 1 * 1, dim_feedforward)
        # self.fc1 = nn.Linear(channel * 2 * 128, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, channel * 1 * 1)

        self.dropout2 = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(channel * 1 * 1)

        self.fc3 = nn.Linear(channel * 1 * 1, classes)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)

        init.xavier_uniform_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

        init.xavier_uniform_(self.fc3.weight)
        init.constant_(self.fc3.bias, 0)

    def forward_ffn(self, src):
        src = self.fc2(
            self.dropout1(self.activation(self.fc1(src))))
        return src

    def forward(self, x):
        # WT
        x = x + self.bn1(self.wt_conv2d(x))

        # Encoder
        x = self.inae(x)

        # residual = x
        # x = self.dcn(x)

        # Add & Norm
        # if self.normalize_before:
        #     x = self.bn2(x)
        # x = residual + self.dropout(x)
        # if not self.normalize_before:
        #     x = self.bn2(x)

        x = self.pool(x)
        x = self.flatten(x)

        residual = x
        # FFN
        x = self.forward_ffn(x)

        # Add & Norm
        if self.normalize_before:
            x = self.ln(x)
        x = residual + self.dropout2(x)
        if not self.normalize_before:
            x = self.ln(x)

        x = self.fc3(x)
        return x
