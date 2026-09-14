# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import collections
import os

import torch
from torch.nn.functional import pairwise_distance as dist

from .all_training import (
    NTMathMixin, normalize_loss, train_model as _train_model,
)


@torch.no_grad()
def cal_metrics(output0, output1, label, g_config):
    """Threshold-free separability of the embedding, plus the raw distances.

    What this replaced was an accuracy: a pair was called "same" when its
    distance fell under g_config.margin, which is 1.0. On validation every
    distance falls under 1.0 -- same-pairs around 0.11, different-pairs
    around 0.44 -- so every pair was called "same" and the number collapsed
    to the positive fraction, a flat ~0.51 that reads as chance. It was not
    chance: the embedding separates held-out clips by roughly 4x. That
    reading was drawn and written down once already before anyone noticed
    the metric could not say otherwise.

    AUC has no threshold in it. It is the probability that a randomly drawn
    different-pair sits further apart than a randomly drawn same-pair, so
    0.5 is chance and 1.0 is perfect separation, wherever the distances
    happen to lie on the number line.

    Computed per batch and averaged by the caller, rather than pooled over
    the epoch, because that is the shape the metric accumulator has. With
    batches this size the two agree closely. A batch holding only one class
    has no AUC to report and contributes 0.5; those are rare here, and the
    bias is toward chance rather than away from it.
    """
    # detach before anything else. These come back as metrics and are summed
    # over the epoch, and .cpu() alone keeps them attached to the graph that
    # produced them, so the accumulator holds every batch's activations: the
    # stage grew past 170 GB across five processes by its fourth epoch. They
    # are returned as floats for the same reason.
    label_tensor = label.detach().cpu()
    l21_tensor = dist(output0, output1).detach().cpu()

    # Distance. label == -1 is a different-pair, anything else is a same-pair.
    _idx = label_tensor == -1
    same = l21_tensor[_idx == 0]
    diff = l21_tensor[_idx]

    s_dis = 0 if len(same) == 0 else same.mean()
    d_dis = 0 if len(diff) == 0 else diff.mean()

    # AUC by the Mann-Whitney identity: the rank sum of the different-pairs,
    # less the ranks they would hold if they were the smallest values, over
    # every same/different pairing. No threshold appears anywhere in it.
    if len(same) == 0 or len(diff) == 0:
        auc = 0.5
    else:
        combined = torch.cat([same, diff])
        n = len(combined)

        sorted_vals, order = torch.sort(combined)

        # Midranks, not raw positions. A collapsed embedding puts every
        # distance at the same value, and ranking those by position alone
        # hands one class all the low ranks on an arbitrary tie-break --
        # reporting 0.0 or 1.0 for what is precisely no information. Tied
        # values have to share the mean of the positions they span, which
        # sends that case to 0.5 where it belongs.
        _, inverse, counts = torch.unique(
            sorted_vals, return_inverse=True, return_counts=True)
        ends = torch.cumsum(counts, 0)
        starts = ends - counts
        group_rank = (starts + ends + 1).float() / 2.0

        ranks = torch.empty(n, dtype=torch.float)
        ranks[order] = group_rank[inverse]

        n_same, n_diff = len(same), len(diff)
        rank_sum_diff = ranks[n_same:].sum()
        u = rank_sum_diff - n_diff * (n_diff + 1) / 2.0
        auc = float(u / (n_same * n_diff))

    # How far apart the two populations sit, in units of the same-pair
    # distance. Reported alongside AUC because it is the quantity that
    # actually moved between runs and the one a human can sanity check.
    separation = 0.0 if s_dis == 0 else float(d_dis / s_dis)

    return float(auc), float(s_dis), float(d_dis), separation


_Metrics = collections.namedtuple('_Metrics', [
    'loss', 'auc', 'same_dis', 'diff_dis', 'separation',
])


class Metrics(NTMathMixin, _Metrics):
    pass


def train_model(model, criterion, train_loader, test_loader, g_config, lr_scheduler, epoch, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")

    def run_model(input_batch):
        input1, input2, label = (x.to(device) for x in input_batch)
        output0, output1 = model(input1), model(input2)
        label = label.float()
        separability = cal_metrics(output0, output1, label, g_config)

        loss = criterion(output0, output1, label)

        loss_value = normalize_loss(loss.item())
        return loss, Metrics(loss_value, *separability)

    def format_metrics(m):
        return ('loss:{:.5f} auc:{:.3f} | sdis:{:.3f} ddis:{:.3f} sep:{:.2f}x'
                .format(*m))

    lr, lr_step = 0.001, 2
    max_iterations = g_config.maxIterations
    # Zero by default; see the reasoning on the knob in g_config.
    weight_decay = getattr(g_config, 'siamese_weight_decay', 0.0)
    _train_model(
        model, train_loader, test_loader, g_config, lr_scheduler, epoch,
        lr, lr_step, max_iterations, run_model, Metrics._zero(),
        format_metrics, weight_decay=weight_decay,
    )
