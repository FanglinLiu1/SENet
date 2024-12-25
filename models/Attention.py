# ------------------------------------------------------------------------------------------------
# Transformer
# Copyright (c) 2024 Fanglin Liu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.utils import get_activation, deformable_attention_core_func


__all__ = ['MLP', 'MSDeformableAttention']


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".
                         format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_feat_levels,
                 num_heads,
                 num_points):
        super(MSDeformableAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_feat_levels = num_feat_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.total_points = num_heads * num_feat_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn = deformable_attention_core_func

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = ((grid_init / grid_init.abs().
                     max(-1, keepdim=True)[0]).
                     view(self.num_heads, 1, 1, 2).
                     repeat(1, self.num_feat_levels, self.num_points, 1))
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask):
        """
        value is input_flatten [bs, h1 * w1 + h2 * w2, C]
        Args:
            query (Tensor): [bs, Length_{query}, C]
            reference_points (Tensor): [bs, Length_{query}, n_feat_levels, 2]
                , range in [0, 1], top-left (0, 0), bottom-right (1, 1), including padding area
            value (Tensor): [bs, Length_{value}, C]
            value_spatial_shapes (Tensor|List): [n_feat_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, Length_{value}], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        value_spatial_shapes = torch.tensor(value_spatial_shapes, dtype=torch.long, device=query.device)
        assert (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum() == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = (self.sampling_offsets(query).
                            view(bs, Len_q, self.num_heads, self.num_feat_levels, self.num_points, 2))
        attention_weights = (self.attention_weights(query).
                             view(bs, Len_q, self.num_heads, self.num_feat_levels * self.num_points))
        attention_weights = (F.softmax(attention_weights, -1).
                             view(bs, Len_q, self.num_heads, self.num_feat_levels, self.num_points))
        # N, Len_q, n_heads, n_feat_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([value_spatial_shapes[..., 1], value_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn(value=value,
                                         value_spatial_shapes=value_spatial_shapes,
                                         sampling_locations=sampling_locations,
                                         attention_weights=attention_weights)

        output = self.output_proj(output)
        return output


if __name__ == '__main__':
    model = MSDeformableAttention(256, 1, 8, 4)
    query = torch.randn(64, 2 * 128, 256)
    reference_points = torch.rand(64, 2 * 128, 1, 2)
    value = torch.randn(64, 2 * 128, 256)
    value_spatial_shapes = [(2, 128)]
    value_mask = torch.zeros(64, 2 * 128, dtype=torch.bool)
    output = model(query, reference_points, value, value_spatial_shapes, value_mask)
    print(output.shape)  # [bs, query_length, embed_dim]
