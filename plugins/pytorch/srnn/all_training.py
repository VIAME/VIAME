# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import math
import os

import torch

from .utilities import checkpoint, logging


class NTMathMixin:
    def __add__(self, other):
        return self._make(e + f for e, f in zip(self, other))

    def __truediv__(self, x):
        return self._make(e / x for e in self)

    @classmethod
    def _zero(cls):
        return cls._make([0] * len(cls._fields))


def normalize_loss(loss):
    if math.isinf(loss):
        logging("WARNING: received an inf loss, setting loss value to 0")
        loss = 0
    return loss


def train_model(
        model, train_loader, test_loader, g_config,
        lr_scheduler, epoch, lr, lr_step, max_iterations,
        run_model, metric_zero, format_metrics,
):
    optimizer = g_config.optimizer(model.parameters(), lr=lr)

    def run_batch(input_batch, train):
        model.train(train)
        loss, metrics = run_model(input_batch)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return metrics

    def run_epoch(train):
        if train:
            loader = train_loader
            display_interval = g_config.displayInterval
        else:
            loader = test_loader
            display_interval = g_config.vali_displayInterval
        avg_metrics = metric_zero
        final_metrics = metric_zero
        for batch_idx, input_batch in enumerate(loader):
            cur_metrics = run_batch(input_batch, train=train)
            avg_metrics += cur_metrics
            final_metrics += cur_metrics
            # display training info
            if (batch_idx + 1) % display_interval == 0:
                logging(f'Epoch {epoch}: {batch_idx} / {len(loader)} | '
                        + (f'lr:{lr} - t' if train else 'v')
                        + format_metrics(avg_metrics / display_interval))
                avg_metrics = metric_zero

        # Not meaningful for training since the weights change
        if not train:
            final_metrics /= len(test_loader)
            logging(f'Epoch {epoch}: final v' + format_metrics(final_metrics))

    def resident_memory():
        """This process and its children, in GB, or None where unavailable.

        Reported each epoch because these stages have twice been killed by a
        signal with nothing in the log to say why, and a kill from outside
        looks identical whether it was memory or anything else. A number here
        settles it from the log alone rather than from a rerun.
        """
        try:
            import resource

            total = resource.getrusage( resource.RUSAGE_SELF ).ru_maxrss
            total += resource.getrusage( resource.RUSAGE_CHILDREN ).ru_maxrss

            return total / ( 1024.0 * 1024.0 )  # ru_maxrss is KB on linux
        except Exception:
            return None

    # train loop
    for epoch in range(epoch, max_iterations):
        # change learning rate
        optimizer, lr = lr_scheduler(optimizer, epoch, lr, lr_step)

        peak = resident_memory()

        if peak is not None:
            logging( 'Epoch {}: peak resident memory {:.1f} GB'
                     .format( epoch, peak ) )

        run_epoch(train=True)
        run_epoch(train=False)

        # save snapshot
        save_path = os.path.join(g_config.model_dir, 'snapshot_epoch_{}.pt'.format(epoch))
        torch.save(checkpoint(model, epoch), save_path)
        logging('Snapshot saved to {}'.format(save_path))

    else:
        # terminate
        logging('Maximum epoch reached, terminating ...')
