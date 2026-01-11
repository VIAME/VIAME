# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch.optim.lr_scheduler


class _LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Fixes call to epoch 0 twice

    See:
        https://github.com/pytorch/pytorch/issues/8837
    """
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, torch.optim.lr_scheduler.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch
        self.step(last_epoch)  # The major change is to remove the +1

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        # epoch is really last epoch
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class _LRScheduler2(_LRScheduler):
    """ Add a bit of extra functionality to the base torch LR scheduler """

    def current_lrs(self):
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        return lrs


class ListedLR(_LRScheduler2):
    """
    Simple scheduler that simply sets the LR based on the epoch.

    Allows for hard-coded schedules for quick prototyping. Good for reproducing
    papers, but bad for experimentation.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        points (dict): Mapping from epoch number to a learning rate.
            The epoch number indicates which epoch will be given a certain
            learning rate. Therefore if epoch 0 has a learning rate of 0.1 then
            that is what the LR will be set to BEFORE running epoch 0.
            Likewise if epoch 1 has a learning rate of 1.0, then AFTER epoch 0
            but BEFORE epoch 1, the learning rate will be set to 1.0.
        last_epoch (int): The index of last epoch. Default: -1.

    CommandLine:
        python ~/code/netharn/netharn/schedulers/listed.py ListedLR:1

    Example:
        >>> # Assuming optimizer has two groups.
        >>> import ubelt as ub
        >>> from viame.pytorch import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0)
        >>> points = {0: .01, 2: .02, 3: .1, 6: .05, 9: .025}
        >>> self = ListedLR(optimizer, points)
        >>> lrs = [self._get_epoch_lr(epoch) for epoch in range(0, 11)]
        >>> print(list(ub.flatten(lrs)))
        [0.01, 0.01, 0.02, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]
        >>> assert self.current_lrs() == [0.01]
        >>> self = ListedLR(optimizer, points, interpolate=True)
        >>> lrs = [self._get_epoch_lr(epoch) for epoch in range(0, 11)]
        >>> print(ub.repr2(list(ub.flatten(lrs)), precision=3, nl=0))
        [0.010, 0.015, 0.020, 0.100, 0.083, 0.067, 0.050, 0.042, 0.033, 0.025, 0.025]

    Example:
        >>> # Assuming optimizer has two groups.
        >>> import ubelt as ub
        >>> from viame.pytorch import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0)
        >>> points = {0: 1, 2: 2}
        >>> self = ListedLR(optimizer, points)
        >>> lrs = []
        >>> for i in range(5):
        >>>     lrs.append(self.get_lr())
        >>>     self.step()
        >>> print(list(ub.flatten(lrs)))
        [1, 1, 2, 2, 2]
    """

    def __init__(self, optimizer, points, interpolate=False,
                 last_epoch=-1):
        if not isinstance(points, dict):
            raise TypeError(points)
        self.interpolate = interpolate
        self.points = points

        # self.optimizer = optimizer
        # self.last_epoch = last_epoch

        # epochs where the lr changes
        self.key_epochs = sorted(self.points.keys())

        super(ListedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self._get_epoch_lr(self.last_epoch + 1)
        return lr

    def _get_epoch_lr(self, epoch):
        """ return lr based on the epoch """
        key_epochs = self.key_epochs
        points = self.points
        base_lrs = self.base_lrs

        assert epoch >= 0

        if epoch in key_epochs:
            prev_key_epoch = epoch
            next_key_epoch = epoch
        else:
            idx = np.searchsorted(key_epochs, epoch, 'left') - 1
            prev_key_epoch = key_epochs[idx]
            if idx < len(key_epochs) - 1:
                next_key_epoch = key_epochs[idx + 1]
            else:
                next_key_epoch = prev_key_epoch

        if self.interpolate:
            if next_key_epoch == prev_key_epoch:
                new_lr = points[next_key_epoch]
            else:
                prev_lr = points[next_key_epoch]
                next_lr = points[prev_key_epoch]

                alpha = (epoch - prev_key_epoch) / (next_key_epoch - prev_key_epoch)

                new_lr = alpha * prev_lr + (1 - alpha) * next_lr

            epoch_lrs = [new_lr for _ in base_lrs]
        else:
            if epoch < prev_key_epoch:
                epoch_lrs = base_lrs
            else:
                new_lr = points[prev_key_epoch]
                epoch_lrs = [new_lr for _ in base_lrs]
        return epoch_lrs


class Exponential(_LRScheduler2):
    """
    Decay learning rate by a factor of `gamma` every `stepsize` epochs.

    This class exists in torch, but lacks the stepsize parameter

    Example:
        >>> import ubelt as ub
        >>> from viame.pytorch import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        >>> self = Exponential(optimizer, gamma=0.01, stepsize=2)
        >>> rates = np.array([self._get_epoch_lr(i) for i in range(6)]).T[0]
        >>> target = np.array([1E-3, 1E-3, 1E-5, 1E-5, 1E-7, 1E-7])
        >>> assert np.allclose(target, rates)
    """
    def __init__(self, optimizer, gamma=0.1, stepsize=100):
        self.gamma = gamma
        self.stepsize = stepsize
        super(Exponential, self).__init__(optimizer)

    def get_lr(self):
        """
        If optimizer is specified, its learning rate is modified inplace.
        """
        new_lrs = self._get_epoch_lr(self.last_epoch + 1)
        return new_lrs

    def _get_epoch_lr(self, epoch):
        """ return lr based on the epoch """
        n_decays = epoch // self.stepsize
        new_lrs = [lr * (self.gamma ** n_decays) for lr in self.base_lrs]
        return new_lrs


class BatchLR(_LRScheduler2):
    __batchaware__ = True

    def __init__(self, optimizer, points, interpolate=False,
                 last_epoch=-1):
        if not isinstance(points, dict):
            raise TypeError(points)
        self.interpolate = interpolate
        self.points = points

        # self.optimizer = optimizer
        # self.last_epoch = last_epoch

        # epochs where the lr changes
        self.key_epochs = sorted(self.points.keys())

        super(BatchLR, self).__init__(optimizer, last_epoch)

        self.bx = 0

    def step_batch(self, bx=None):
        if bx is None:
            self.bx += 1
        else:
            self.bx = bx

    def step(self, *a, **kw):
        self.bx = 0
        return super(BatchLR, self).step(*a, **kw)

    step_epoch = step

    def get_lr(self):
        lr = self._get_epoch_lr(self.last_epoch + 1, self.bx)
        return lr

    def _get_epoch_lr(self, epoch, bx):
        """ return lr based on the epoch """
        key_epochs = self.key_epochs
        points = self.points
        base_lrs = self.base_lrs

        assert epoch >= 0

        if epoch in key_epochs:
            prev_key_epoch = epoch
            next_key_epoch = epoch
        else:
            idx = np.searchsorted(key_epochs, epoch, 'left') - 1
            prev_key_epoch = key_epochs[idx]
            if idx < len(key_epochs) - 1:
                next_key_epoch = key_epochs[idx + 1]
            else:
                next_key_epoch = prev_key_epoch

        if self.interpolate:
            if next_key_epoch == prev_key_epoch:
                new_lr = points[next_key_epoch]
            else:
                prev_lr = points[next_key_epoch]
                next_lr = points[prev_key_epoch]

                alpha = (epoch - prev_key_epoch) / (next_key_epoch - prev_key_epoch)

                new_lr = alpha * prev_lr + (1 - alpha) * next_lr

            epoch_lrs = [new_lr for _ in base_lrs]
        else:
            if epoch < prev_key_epoch:
                epoch_lrs = base_lrs
            else:
                new_lr = points[prev_key_epoch]
                epoch_lrs = [new_lr for _ in base_lrs]
        return epoch_lrs


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.schedulers.listed all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
