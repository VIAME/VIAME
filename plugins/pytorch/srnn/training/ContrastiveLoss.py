# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import torch
import torch.nn


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0, size_average=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, y):
        assert input1.size() == input2.size(), "Input sizes must be equal."

        # euclidian distance
        diff = input1 - input2
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = (1 + y) / 2.0 * dist_sq + (1 - y) / 2.0 * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0

        if self.size_average:
            loss = loss / y.size(0)

        return loss
