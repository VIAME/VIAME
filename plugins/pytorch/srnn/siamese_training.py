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
def cal_accuracy(output0, output1, label, g_config):
    label_tensor = label.cpu()
    l21_tensor = dist(output0, output1).cpu()

    # Distance
    _idx = label_tensor == -1  # y==-1
    s_dis = 0 if len(l21_tensor[_idx == 0]) == 0 else l21_tensor[_idx == 0].mean()
    d_dis = 0 if len(l21_tensor[_idx]) == 0 else l21_tensor[_idx].mean()

    # accuracy
    idx = l21_tensor <= g_config.margin  # y==1's idx

    cur_score = torch.FloatTensor(label.size(0))
    cur_score.fill_(-1.0)
    cur_score[idx] = 1.0

    accuracy = torch.eq(cur_score, label_tensor).sum().float() / label_tensor.size(0)

    return accuracy, s_dis, d_dis


_Metrics = collections.namedtuple('_Metrics', [
    'loss', 'accuracy', 'same_dis', 'diff_dis',
])


class Metrics(NTMathMixin, _Metrics):
    pass


def train_model(model, criterion, train_loader, test_loader, g_config, lr_scheduler, epoch, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")

    def run_model(input_batch):
        input1, input2, label = (x.to(device) for x in input_batch)
        output0, output1 = model(input1), model(input2)
        label = label.float()
        accuracy_and_distances = cal_accuracy(output0, output1, label, g_config)

        loss = criterion(output0, output1, label)

        loss_value = normalize_loss(loss.item())
        return loss, Metrics(loss_value, *accuracy_and_distances)

    def format_metrics(m):
        return 'loss:{:.5f} acc:{:.2f} | sdis:{:.3f} ddis:{:.3f}'.format(*m)

    lr, lr_step = 0.001, 2
    max_iterations = g_config.maxIterations
    _train_model(
        model, train_loader, test_loader, g_config, lr_scheduler, epoch,
        lr, lr_step, max_iterations, run_model, Metrics._zero(),
        format_metrics,
    )
