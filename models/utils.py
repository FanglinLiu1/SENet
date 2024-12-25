# ------------------------------------------------------------------------------------------------
# Transformer
# Copyright (c) 2024 Fanglin Liu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    # math: 'x ∈ [eps, 1 - eps]'
    x = x.clamp(min=eps, max=1 - eps)

    inverse_x = torch.log(x / (1 - x))
    return inverse_x


def deformable_attention_core_func(value,
                                   value_spatial_shapes,
                                   sampling_locations,
                                   attention_weights):
    """
    Math:
        'O=\sum_{l=1}^L\sum_{p=1}^PW_{l,p}\cdot\text{sample}(V_l,S_{l,p})'

    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2] -> [(w1, h1), (w2, h2)]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, n_head * C]
    """
    # for debug and test only,
    # need to use cuda version instead
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # [bs, h * w, n_head, c] -> [bs, h * w, n_head*c] -> [bs, n_head * c, h * w] -> [bs * n_head, c, h, w]
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # [bs, h * w, n_head, n_points, 2] -> [bs, n_head, h * w, n_points, 2] -> [bs * n_head, h * w, n_points, 2]
        sampling_grid_l_ = (sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1))
        # [bs * n_head, c, h * w, n_points]
        sampling_value_l_ = F.grid_sample(value_l_,
                                          sampling_grid_l_,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # [bs, h * w, n_head, n_levels, n_points] -> [bs, n_head, h * w, n_levels, n_points] -> [bs * n_head, 1, h * w, n_levels * n_points]
    attention_weights = (attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points))
    output = ((torch.stack(
        sampling_value_list, dim=-2).flatten(-2) *
               attention_weights).sum(-1).reshape(bs, n_head * c, Len_q))
    return output.permute(0, 2, 1)


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    # math: 'y = -log((1 - x) / x)'
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def get_activation(act: str, inplace: bool = True) -> nn.Module:
    act = act.lower()

    if act == 'silu':
        f = nn.SiLU()
    elif act == 'relu':
        f = nn.ReLU()
    elif act == 'leaky_relu':
        f = nn.LeakyReLU()
    elif act == 'gelu':
        f = nn.GELU()
    elif act is None:
        f = nn.Identity()
    elif isinstance(act, nn.Module):
        f = act
    else:
        raise RuntimeError(f'Deprecated activation functions: {act}')

    if hasattr(f, 'inplace'):
        f.inplace = inplace
    return f


if __name__ == '__main__':
    value = torch.randn(64, 2 * 128, 8, 256)
    value_spatial_shapes = [(2, 128)]
    sampling_locations = torch.randn(64, 2 * 128, 8, 1, 4, 2)
    attention_weights = torch.randn(64, 2 * 128, 8, 1, 4)
    output = deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights)
    print(output.shape)  # [bs, Length_{query}, n_head * C]
