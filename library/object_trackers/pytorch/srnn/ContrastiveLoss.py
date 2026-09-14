# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import torch
import torch.nn


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0, size_average=True, eps=1e-9):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

        # sqrt is not differentiable at zero: its derivative goes to infinity,
        # so a pair whose embeddings coincide sends inf back through the
        # network and every weight becomes NaN from that step on. Two crops of
        # the same target can be identical, and an embedding that has started
        # to collapse produces them constantly, so this is not a rare case. A
        # floor inside the square root costs nothing and removes it.
        self.eps = eps

    def forward(self, input1, input2, y):
        assert input1.size() == input2.size(), "Input sizes must be equal."

        # euclidian distance
        diff = input1 - input2
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq + self.eps)

        mdist = self.margin - dist
        margin_dist = torch.clamp(mdist, min=0.0)
        loss = (1 + y) / 2.0 * dist_sq + \
               (1 - y) / 2.0 * torch.pow(margin_dist, 2)
        loss = torch.sum(loss) / 2.0

        if self.size_average:
            loss = loss / y.size(0)

        return loss
