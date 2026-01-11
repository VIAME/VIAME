import torch
import torch.nn as nn

__all__ = ['ContrastiveLoss']


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.

    References:
        https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py

    LaTeX:
        Let D be the distance computed between the network layers or the direct
        distance output of the network.

        y is 0 if the pair should be labled as an imposter
        y is 1 if the pair should be labled as genuine

        ContrastiveLoss = ((y * D) ** 2 + ((1 - y) * max(m - D, 0) ** 2)) / 2

        $(y E)^2 + ((1 - y) max(m - E, 0)^2)$

    CommandLine:
        python -m clab.criterions ContrastiveLoss --show


    DisableExample:
        >>> # DISABLE_DOCTEST
        >>> from clab.criterions import *
        >>> import utool as ut
        >>> import numpy as np
        >>> vecs1, vecs2, label = testdata_siam_desc()
        >>> output = torch.nn.PairwiseDistance(p=2)(vecs1, vecs2)
        >>> self = ContrastiveLoss()
        >>> ut.exec_func_src(self.forward, globals())
        >>> func = self.forward
        >>> loss2x, dist = ut.exec_func_src(self.forward, globals(), globals(), keys=['loss2x', 'dist'])
        >>> ut.quit_if_noshow()
        >>> loss2x, dist, label = map(np.array, [loss2x, dist, label])
        >>> label = label.astype(bool)
        >>> dist0_l2 = dist[~label]
        >>> dist1_l2 = dist[label]
        >>> loss0 = loss2x[~label] / 2
        >>> loss1 = loss2x[label] / 2
        >>> # xdoc: +REQUIRES(--show)
        >>> import plottool as pt
        >>> pt.plot2(dist0_l2, loss0, 'x', color=pt.FALSE_RED, label='imposter_loss', y_label='loss')
        >>> pt.plot2(dist1_l2, loss1, 'x', color=pt.TRUE_BLUE, label='genuine_loss', y_label='loss')
        >>> pt.gca().set_xlabel('l2-dist')
        >>> pt.legend()
        >>> ut.show_if_requested()

    Example:
        >>> # DISABLE_DOCTEST
        >>> import torch
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> xpu = nh.XPU(None)
        >>> imgs1 = xpu.move(torch.rand(1, 3, 224, 244))
        >>> imgs2 = xpu.move(torch.rand(1, 3, 224, 244))
        >>> label = (xpu.move(torch.rand(3)) * 2).long()

        >>> model = SiameseLP(input_shape=imgs1.shape[1:])
        >>> output = model(imgs1, imgs2)
        >>> self = ContrastiveLoss(margin=10)
        >>> self.forward(output, label)
    """

    def __init__(self, weight=None, margin=1.0):
        # ut.super2(ContrastiveLoss, self).__init__()
        super(ContrastiveLoss, self).__init__()
        self.weight = weight
        self.margin = margin

        self.neg_label = 0
        self.pos_label = 1

    def forward(self, output, label):
        # Output should be a Bx1 vector representing the predicted
        # distance between each pair in a patch of image pairs
        dist = torch.squeeze(output)

        # Build indicator vectors labeling which pairs are pos and neg
        is_genuine = label.float()
        is_imposter = (1 - is_genuine)

        # Negate and clamp the distance for imposter pairs so these pairs are
        # encouraged to predict a distance larger than the margin
        hinge_neg_dist = torch.clamp(self.margin - dist, min=0.0)
        # Apply this loss only to negative examples
        loss_imposter = is_imposter * torch.pow(hinge_neg_dist, 2)

        # THe loss for positive examples is simply the distance because we wish
        # to encourage the network to predict values close to zero
        loss_genuine = is_genuine * torch.pow(dist, 2)

        # Weight each class if desired
        if self.weight is not None:
            loss_imposter *= self.weight[0]
            loss_genuine *= self.weight[1]

        # Sum the loss together (actually there is a 2x factor here)
        loss2x = loss_genuine + loss_imposter

        # Divide by 2 after summing for efficiency
        ave_loss = torch.sum(loss2x) / 2.0 / label.size()[0]
        loss = ave_loss
        return loss
