# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# Try to load the CUDA extension, falling back to pure PyTorch
# Prefer mmcv's implementation since it's built as part of VIAME
MSDA = None
_USE_PYTORCH_FALLBACK = False
_USE_MMCV = False

try:
    # Try mmcv's implementation first (built as part of VIAME)
    from mmcv.ops.multi_scale_deform_attn import ext_module as MSDA
    _USE_MMCV = True
except (ImportError, ModuleNotFoundError):
    try:
        # Fall back to standalone MultiScaleDeformableAttention extension
        import MultiScaleDeformableAttention as MSDA
    except ImportError:
        warnings.warn(
            "Neither mmcv extension nor MultiScaleDeformableAttention CUDA extension "
            "is available. Falling back to pure PyTorch implementation which may be slower."
        )
        _USE_PYTORCH_FALLBACK = True


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        ctx.use_pytorch_fallback = _USE_PYTORCH_FALLBACK
        ctx.use_mmcv = _USE_MMCV

        if _USE_PYTORCH_FALLBACK:
            # Use pure PyTorch implementation
            output = ms_deform_attn_core_pytorch(
                value, value_spatial_shapes, sampling_locations, attention_weights)
            ctx.save_for_backward(value, value_spatial_shapes, sampling_locations, attention_weights)
        elif _USE_MMCV:
            # mmcv uses keyword argument for im2col_step
            output = MSDA.ms_deform_attn_forward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step=ctx.im2col_step)
            ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        else:
            output = MSDA.ms_deform_attn_forward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, ctx.im2col_step)
            ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if ctx.use_pytorch_fallback:
            raise NotImplementedError(
                "Backward pass is not implemented for pure PyTorch fallback. "
                "Please build the MultiScaleDeformableAttention CUDA extension for training."
            )
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors

        if ctx.use_mmcv:
            # mmcv backward uses different signature with pre-allocated gradients
            grad_value = torch.zeros_like(value)
            grad_sampling_loc = torch.zeros_like(sampling_locations)
            grad_attn_weight = torch.zeros_like(attention_weights)
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, grad_output.contiguous(),
                grad_value, grad_sampling_loc, grad_attn_weight,
                im2col_step=ctx.im2col_step)
        else:
            grad_value, grad_sampling_loc, grad_attn_weight = \
                MSDA.ms_deform_attn_backward(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
