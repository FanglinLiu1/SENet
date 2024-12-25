# ------------------------------------------------------------------------------------------------
# Transformer
# Copyright (c) 2024 Fanglin Liu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------

import copy

import torch
import torch.nn as nn

from models.utils import get_activation
from models.Attention import MSDeformableAttention
from models.wtconv2d import WTConv2d
from models.FPN import FPN, BiFPN


__all__ = ['ConvNormLayer', 'TransformerEncoderLayer', 'TransformerEncoder', 'IntegratedAutoEncoder']


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super(ConvNormLayer, self).__init__()
        self.conv = nn.Conv2d(ch_in,
                              ch_out,
                              kernel_size,
                              stride,
                              padding=(kernel_size - 1) // 2 if padding is None else padding,
                              bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class TransformerEncoderLayer(nn.Module):
    """
    Args:

    Returns:

    """
    def __init__(self,
                 d_model,
                 dim_feedforward,
                 dropout,
                 activation,
                 n_head,
                 n_feat_levels,
                 n_points,
                 normalize_before):
        super(TransformerEncoderLayer, self).__init__()
        self.normalize_before = normalize_before

        # self attention
        self.self_attn = MSDeformableAttention(embed_dim=d_model,
                                               num_heads=n_head,
                                               num_feat_levels=n_feat_levels,
                                               num_points=n_points)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward_ffn(self, src):
        src = self.linear2(
            self.dropout2(self.activation(self.linear1(src))))
        return src

    def forward(self,
                src,
                pos_embed,
                reference_points,
                spatial_shapes,
                padding_mask):
        """
        Args:
            src (Tensor): [bs, Length_{query}, C]
            pos_embed (Tensor): [bs, Length_{query}, C]
            reference_points (Tensor): [bs, Length_{query}, n_feat_levels, 2]
                , range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
            spatial_shapes (List): [n_feat_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            padding_mask (Tensor): [bs, Length_{value}]
                , True for non-padding elements, False for padding elements

        Returns:
            src (Tensor): [bs, Length_{query}, C]
        """
        residual = src
        src = self.self_attn(\
            query=self.with_pos_embed(tensor=src, pos_embed=pos_embed),
                             reference_points=reference_points,
                             value=src,
                             value_spatial_shapes=spatial_shapes,
                             value_mask=padding_mask)

        # Add & Norm
        if self.normalize_before:
            src = self.norm1(src)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        # FFN
        src = self.forward_ffn(src)

        # Add & Norm
        if self.normalize_before:
            src = self.norm2(src)
        src = residual + self.dropout3(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    """
    Args:

    Returns:

    """
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        valid_ratios = valid_ratios.to(device)
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                                          indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                src,
                spatial_shapes,
                valid_ratios,
                pos,
                padding_mask):
        """
        Args:
            src (Tensor): [bs, Length_{query}, C]
            spatial_shapes (List): [n_feat_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            valid_ratios (Tensor): [bs, n_feat_levels, 2], [(h = H / H_0, w = W / W_0)]
            pos (Tensor): [bs, Length_{query}, C]
            padding_mask (Tensor): [bs, Length_{value}]
                , True for non-padding elements, False for padding elements
            reference_points (Tensor): [bs, Length_{query}, n_feat_levels, 2]
                , range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, src.device)
        for _, layer in enumerate(self.layers):
            output = layer(src=output,
                           pos_embed=pos,
                           reference_points=reference_points,
                           spatial_shapes=spatial_shapes,
                           padding_mask=padding_mask)

        return output


class IntegratedAutoEncoder(nn.Module):
    """
    Args:

    Returns:

    """
    def __init__(self,
                 spatial_shapes,
                 d_model,
                 dim_feedforward,
                 dropout,
                 activation,
                 num_encoder_layers,
                 num_head,
                 num_feat_levels,
                 enc_n_points,
                 in_channels=[1],
                 encoder_idx=[0],
                 pe_temperature=10000,
                 normalize_before=True,
                 two_stage=False):
        super(IntegratedAutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.spatial_shapes = spatial_shapes
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.encoder_idx = encoder_idx
        self.pe_temperature = pe_temperature
        self.two_stage = two_stage

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, d_model, kernel_size=1, bias=False),
                    nn.BatchNorm2d(d_model)
                )
            )

        # 在RML22数据集上初始化了
        # self.fpn = FPN(1, d_model)
        # self.bifpn = BiFPN(1, d_model)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                n_head=num_head,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                n_feat_levels=num_feat_levels,
                                                n_points=enc_n_points,
                                                normalize_before=normalize_before)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=num_encoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        for module in self.modules():
            if isinstance(module, MSDeformableAttention):
                module._reset_parameters()

    def build_2d_sincos_position_embedding(self, w, h, embed_dim, temperature):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([1 * out_w.sin(), 1 * out_w.cos(),
                          1 * out_h.sin(), 1 * out_h.cos()], dim=1)[None, :, :]

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, feats):
        # assert len(feats) == len(self.in_channels)

        proj_feats = []
        for i, proj_module in enumerate(self.input_proj):
            feat = feats[:, i:i + 1, :, :]
            proj_feat = proj_module(feat)
            proj_feats.append(proj_feat)
        proj_feats = torch.cat(proj_feats, dim=1)

        # FPN
        # proj_feats = self.fpn(feats)
        # proj_feats = self.bifpn(feats)

        # encoder
        if self.num_encoder_layers > 0:
            for i, idx_value in enumerate(self.encoder_idx):
                if idx_value >= len(self.input_proj):
                    raise IndexError("enc_idx is out of range")
                bs, c, h, w = proj_feats.shape

                spatial_shapes = self.spatial_shapes
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats.flatten(2).permute(0, 2, 1)
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.d_model, self.pe_temperature).to(src_flatten.device)
                mask_flatten = torch.zeros((bs, h * w), dtype=torch.bool).to(src_flatten.device)
                levels = len(spatial_shapes)
                valid_ratios = torch.ones((bs, levels, 2), dtype=torch.float32).to(src_flatten.device)
                """
                Args:
                    src_flatten (Tensor): [bs, Length_{query}, C]
                    spatial_shapes (Tensor|List): [n_feat_levels, 2]
                        , [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
                    valid_ratios (Tensor): [bs, n_feat_levels, 2]
                    pos_embed (Tensor): [bs, Length_{query}, C]
                    masks (Tensor): [bs, h, w]
                    mask_flatten (Tensor): [bs, Length_{value}]
                        , True for non-padding elements, False for padding elements

                Returns:
                    output (Tensor): [bs, Length_{query}, C]
                """
                memory = self.encoder(src=src_flatten,
                                      spatial_shapes=spatial_shapes,
                                      valid_ratios=valid_ratios,
                                      pos=None,  # pos_embed
                                      padding_mask=None)
                # reshape [B, HxW, C] to [B, C, H, W]
                proj_feats = memory.permute(0, 2, 1).reshape(-1, self.d_model, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        outs = proj_feats
        return outs
