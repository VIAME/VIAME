# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import collections
import os

import torch

from .all_training import (
    NTMathMixin, normalize_loss, train_model as _train_model,
)

_Metrics = collections.namedtuple('_Metrics', ['loss', 'accuracy'])


class Metrics(NTMathMixin, _Metrics):
    pass


def train_model(model, criterion, train_loader, test_loader, g_config, lr_scheduler, epoch, is_target_rnn=False):
    device = torch.device("cuda")

    def run_model(input_batch):
        *inputs, in_label = (x.to(device) for x in input_batch)
        output = model(*inputs)
        if not is_target_rnn:
            output = output[0][:, -1, :]

        loss = criterion(output, in_label.squeeze())

        pred_y = torch.max(output, 1)[1].detach().cpu().numpy().squeeze()
        y = in_label.detach().cpu().numpy().squeeze()
        accuracy = sum(pred_y == y) / y.size

        return loss, Metrics(normalize_loss(loss.item()), accuracy)

    def format_metrics(m):
        return 'loss:{:.5f} acc:{:.4f}'.format(*m)

    lr = g_config.lstm_init_lr
    lr_step = g_config.lstm_lr_step
    max_iterations = g_config.maxRNNIterations
    _train_model(
        model, train_loader, test_loader, g_config, lr_scheduler, epoch,
        lr, lr_step, max_iterations, run_model, Metrics._zero(),
        format_metrics,
    )
