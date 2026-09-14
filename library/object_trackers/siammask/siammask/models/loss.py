# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def select_mask_logistic_loss(pred_mask, label_mask, label_weight,
                              mask_output_size=127, stride=8):
    """Mask loss over the search crop, at the positive anchor positions.

    The mask branch predicts a coarse mask for every position of its output
    grid, each covering a window of the search crop. The label for a position
    is the corresponding window of the groundtruth mask, which is what unfold
    below cuts out, and only positions the box branch marked positive are
    scored.

    Args:
        pred_mask: mask branch output, [B, o_sz*o_sz, H, W] or already
            selected as [N, g_sz*g_sz]
        label_mask: groundtruth over the whole search crop, [B, 1, S, S]
        label_weight: one per output position, [B, 1, H, W], one where that
            position is supervised and zero elsewhere
        mask_output_size: side of the window a position is scored over
        stride: distance in search crop pixels between output positions

    Returns:
        Scalar loss, zero when nothing in the batch is supervised.
    """
    g_sz = mask_output_size

    weight = label_weight.view(-1)
    pos = weight.data.eq(1).nonzero().squeeze()

    # A batch of box only groundtruth supervises nothing, and so does one
    # where no anchor matched. Keeping pred_mask in the expression leaves the
    # graph connected, so backward still runs.
    if pos.dim() == 0 or pos.numel() == 0:
        return pred_mask.sum() * 0

    if pred_mask.dim() == 4:
        b, c, h, w = pred_mask.size()
        o_sz = int(round(c ** 0.5))

        # To (b*h*w, 1, o_sz, o_sz), which orders the same way as the
        # flattened weight above, so pos indexes both
        pred_mask = pred_mask.permute(0, 2, 3, 1).contiguous()
        pred_mask = pred_mask.view(-1, 1, o_sz, o_sz)
        pred_mask = torch.index_select(pred_mask, 0, pos)
        pred_mask = F.interpolate(pred_mask, size=(g_sz, g_sz),
                                  mode='bilinear', align_corners=False)
        pred_mask = pred_mask.view(-1, g_sz * g_sz)
    else:
        pred_mask = torch.index_select(pred_mask, 0, pos)

    # The window for each output position. The padding is what makes the
    # number of windows match the output grid rather than being read off it.
    positions = label_weight.size(-1)
    search = label_mask.size(-1)
    padding = ((positions - 1) * stride + g_sz - search) // 2

    label = F.unfold(label_mask, (g_sz, g_sz), padding=padding, stride=stride)
    label = torch.transpose(label, 1, 2).contiguous().view(-1, g_sz * g_sz)
    label = torch.index_select(label, 0, pos)

    # Soft margin loss wants labels in {-1, 1}, the mask arrives as {0, 1}
    return F.soft_margin_loss(pred_mask, label * 2 - 1)
