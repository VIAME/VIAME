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


def select_mask_logistic_loss(pred_mask, label_mask, label_weight, mask_output_size=127):
    """Compute mask loss using soft margin loss on positive anchor locations.

    Args:
        pred_mask: Predicted mask logits [B, anchors, H, W] or [B, H*W]
        label_mask: Ground truth mask [B, 1, H_full, W_full]
        label_weight: Weight mask indicating positive anchors [B, anchors, H, W]
        mask_output_size: Size of mask output (default 127)

    Returns:
        Scalar loss value
    """
    # Find positive anchor positions (where weight == 1)
    weight = label_weight.view(-1)
    pos = weight.data.eq(1).nonzero().squeeze()

    if len(pos.size()) == 0 or pos.size() == torch.Size([0]):
        return pred_mask.sum() * 0  # Return zero loss if no positive samples

    # Handle different input shapes
    if len(pred_mask.size()) == 4:
        b, a, h, w = pred_mask.size()
        pred_mask = pred_mask.view(b, a, -1)
        # Upsample predictions to mask_output_size if needed
        if h != mask_output_size:
            pred_mask = pred_mask.view(b * a, 1, h, w)
            pred_mask = F.interpolate(pred_mask, size=(mask_output_size, mask_output_size),
                                      mode='bilinear', align_corners=False)
            pred_mask = pred_mask.view(b, a, -1)
        pred_mask = pred_mask.view(-1, mask_output_size * mask_output_size)
    else:
        pred_mask = pred_mask.view(-1, mask_output_size * mask_output_size)

    # Select predictions at positive anchor positions
    pred_mask = torch.index_select(pred_mask, 0, pos)

    # Extract mask patches from ground truth at corresponding locations
    # label_mask is [B, 1, H, W], we need to extract patches
    b = label_mask.size(0)
    label_mask = label_mask.view(b, -1)

    # Expand label_mask to match number of anchors per batch
    num_anchors = weight.size(0) // b
    label_mask = label_mask.unsqueeze(1).expand(b, num_anchors, -1)
    label_mask = label_mask.contiguous().view(-1, label_mask.size(-1))

    # Select labels at positive positions
    label_mask = torch.index_select(label_mask, 0, pos)

    # Ensure predictions and labels have same size
    if pred_mask.size(1) != label_mask.size(1):
        # Interpolate label mask to match prediction size
        label_h = int(label_mask.size(1) ** 0.5)
        label_mask = label_mask.view(-1, 1, label_h, label_h)
        label_mask = F.interpolate(label_mask, size=(mask_output_size, mask_output_size),
                                   mode='bilinear', align_corners=False)
        label_mask = label_mask.view(-1, mask_output_size * mask_output_size)

    # Convert label mask to {-1, 1} for soft margin loss
    # Ground truth mask is typically {0, 1}, convert to {-1, 1}
    label_mask = label_mask * 2 - 1

    # Compute soft margin loss: log(1 + exp(-y * x))
    loss = F.soft_margin_loss(pred_mask, label_mask)

    return loss
