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
from . import contrastive_loss
from . import focal
from . import triplet

from .contrastive_loss import (ContrastiveLoss,)
from .focal import (ELEMENTWISE_MEAN, FocalLoss, focal_loss,
                    nll_focal_loss,)
from .triplet import (TripletLoss, all_pairwise_distances,
                      approx_pdist, exact_pdist,
                      labels_to_adjacency_matrix,)

__all__ = ['ContrastiveLoss', 'CrossEntropyLoss', 'ELEMENTWISE_MEAN',
           'FocalLoss', 'MSELoss', 'TripletLoss', 'all_pairwise_distances',
           'approx_pdist', 'contrastive_loss', 'exact_pdist', 'focal',
           'focal_loss', 'labels_to_adjacency_matrix', 'nll_focal_loss',
           'triplet']
# </AUTOGEN_INIT>
