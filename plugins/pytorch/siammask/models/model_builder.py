# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from viame.pytorch.siammask.core.config import cfg
from viame.pytorch.siammask.models.loss import (
    select_cross_entropy_loss, weight_l1_loss, select_mask_logistic_loss
)
from viame.pytorch.siammask.models.backbone import get_backbone
from viame.pytorch.siammask.models.head import get_rpn_head, get_mask_head, get_refine_head
from viame.pytorch.siammask.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        """Extract template features from exemplar image.

        Returns the template features instead of storing them, to support
        multiple concurrent tracker instances sharing this model.
        """
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        return zf

    def track(self, x, zf):
        """Track object in search image using template features.

        Args:
            x: Search image crop tensor
            zf: Template features (from template() call)

        Returns dict with 'cls', 'loc', 'mask', and for mask models also
        'xf' and 'mask_corr_feature' for use with mask_refine().
        """
        xf = self.backbone(x)
        xf_refine = None
        if cfg.MASK.MASK:
            xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)
        mask = None
        mask_corr_feature = None
        if cfg.MASK.MASK:
            mask, mask_corr_feature = self.mask_head(zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask,
                'xf': xf_refine,
                'mask_corr_feature': mask_corr_feature,
               }

    def mask_refine(self, pos, xf, mask_corr_feature):
        """Refine mask prediction at given position.

        Args:
            pos: Position tuple (y, x) for refinement
            xf: Backbone features from track() call
            mask_corr_feature: Mask correlation features from track() call
        """
        return self.refine_head(xf, mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            # Compute mask loss if label_mask is provided in training data
            if 'label_mask' in data and 'label_mask_weight' in data:
                label_mask = data['label_mask'].cuda()
                label_mask_weight = data['label_mask_weight'].cuda()
                mask_loss = select_mask_logistic_loss(
                    mask, label_mask, label_mask_weight,
                    mask_output_size=cfg.TRACK.MASK_OUTPUT_SIZE
                )
                outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
                outputs['mask_loss'] = mask_loss
            else:
                outputs['mask_loss'] = None
        return outputs
