# -*- coding: utf-8 -*-
"""
mkinit netharn.criterions -w
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss

__extra_all__ = [
    'CrossEntropyLoss', 'MSELoss',
]

# <AUTOGEN_INIT>
from netharn.criterions import contrastive_loss
from netharn.criterions import focal
from netharn.criterions import triplet

from netharn.criterions.contrastive_loss import (ContrastiveLoss,)
from netharn.criterions.focal import (ELEMENTWISE_MEAN, FocalLoss, focal_loss,
                                      nll_focal_loss,)
from netharn.criterions.triplet import (TripletLoss, all_pairwise_distances,
                                        approx_pdist, exact_pdist,
                                        labels_to_adjacency_matrix,)

__all__ = ['ContrastiveLoss', 'CrossEntropyLoss', 'ELEMENTWISE_MEAN',
           'FocalLoss', 'MSELoss', 'TripletLoss', 'all_pairwise_distances',
           'approx_pdist', 'contrastive_loss', 'exact_pdist', 'focal',
           'focal_loss', 'labels_to_adjacency_matrix', 'nll_focal_loss',
           'triplet']
# </AUTOGEN_INIT>
