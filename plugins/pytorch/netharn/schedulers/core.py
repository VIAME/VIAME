import torch.optim.lr_scheduler
from collections import defaultdict

"""

# Notes on torch schedulers

import torch
from torch.optim import lr_scheduler
from torch import optim


parameters = list(torch.nn.Conv1d(1, 1, 1).parameters())

base_lr = 1e-3
optimizer = optim.SGD(parameters, lr=base_lr)


schedulers = {}
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
schedulers[scheduler.__class__.__name__] = scheduler
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr, total_steps=100)
schedulers[scheduler.__class__.__name__] = scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=30)
schedulers[scheduler.__class__.__name__] = scheduler
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
schedulers[scheduler.__class__.__name__] = scheduler

key = scheduler.__class__.__name__


xdata = list(range(100))
ydata = ub.ddict(list)

for key, scheduler in schedulers.items():

    # Reset optimizer LR
    for g in optimizer.param_groups:
        g['lr'] = base_lr

    for x in xdata:
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        ydata[key].append(lr)

import kwplot
kwplot.autompl()

kwplot.multi_plot(xdata=xdata, ydata=ydata)


"""


class CommonMixin(object):

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    # def get_lr(self):
    #     raise NotImplementedError

    def current_lrs(self):
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        return lrs


class TorchNetharnScheduler(CommonMixin, torch.optim.lr_scheduler._LRScheduler):
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

    @property
    def epoch(self):
        return self.last_epoch + 1

    def step(self, epoch=None):
        # epoch is really last epoch
        if epoch is None:
            epoch = self.epoch
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    # def get_lr(self):
    #     raise NotImplementedError

    # def step(self, epoch=None):
    #     if epoch is None:
    #         epoch = self.last_epoch + 1
    #     self.last_epoch = epoch
    #     for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
    #         param_group['lr'] = lr


class NetharnScheduler(CommonMixin):

    def get_lrs(self):
        lr = self.get_lr()
        n_gropus = 1 if self.optimizer is None else len(self.optimizer.param_groups)
        lrs = [lr] * n_gropus
        return lrs


class YOLOScheduler(NetharnScheduler):
    """
    Scheduler that changs learning rates on a per-ITERATION level

    Attributes:
        dset_size (int): number of items per epoch
        batch_size (int): number of items per batch
        burn_in (int or float): number of epochs (fractional is ok) to use
            burn-in modulated learning rates.

    CommandLine:
        xdoctest netharn.schedulers.core YOLOScheduler --show

    Example:
        >>> # Assuming optimizer has two groups.
        >>> from .schedulers.core import *
        >>> import ubelt as ub
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> points = {0: .01, 2: .02, 3: .1, 6: .05, 9: .025}
        >>> self = YOLOScheduler(dset_size=103, batch_size=10, burn_in=1.2,
        >>>                      points=points)
        >>> # Actual YOLO params
        >>> lr = 0.001
        >>> bsize = 32
        >>> bstep = 2
        >>> simbsize = bsize * bstep
        >>> points = {0: lr * 1.0 / simbsize, 155: lr * 0.1 / simbsize, 233: lr * 0.01 / simbsize}
        >>> self = YOLOScheduler(dset_size=16551, batch_size=bsize,
        >>>                      burn_in=3.86683584, points=points)
        >>> xdata = ub.ddict(list)
        >>> ydata = ub.ddict(list)
        >>> for epoch in range(300):
        >>>     lr = self.get_lr()
        >>>     xdata['epoch'].append(self.n_epochs_seen)
        >>>     xdata['iter'].append(self.n_items_seen)
        >>>     ydata['lr'].append(lr)
        >>>     for batch in range(self.dset_size // self.batch_size):
        >>>         lr = self.get_lr()
        >>>         self.step_batch()
        >>> #print('ydata = {}'.format(ub.repr2(ydata, precision=5, nl=0)))
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> xticklabels = sorted({1, 20} | set(points.keys()))
        >>> kwplot.multi_plot(xdata=xdata['epoch'], ydata=ydata, xlabel='epoch', fnum=1,
        >>>                    ylabel='lr', xticklabels=xticklabels, xticks=xticklabels)
        >>> kwplot.show_if_requested()

    """
    __batchaware__ = True

    def __init__(self,
                 optimizer=None,
                 last_epoch=-1,
                 dset_size=None,
                 batch_size=None,
                 burn_in=0,
                 points=None,
                 interpolate=False):
        super(YOLOScheduler, self).__init__()
        self.burn_in = burn_in

        assert batch_size is not None
        assert dset_size is not None

        self.batch_size = batch_size
        self.dset_size = dset_size

        # This keeps tack of progress at the finest possible granularity
        # It should be taken as cannonical over self.batch_num and self.epoch
        # which are integral and rounted.

        # self.epoch_to_n_items_seen = defaultdict(list)  # mapping from epoch to list of batch sizes seen in interation
        self.epoch_to_n_items_seen = defaultdict(int)  # mapping from epoch to number of items (not batches) seen in iteration
        self.n_items_seen = self.dset_size * (last_epoch + 1)

        self.optimizer = optimizer

        if not isinstance(points, dict):
            raise TypeError(points)
        self.interpolate = interpolate
        self.points = points
        self.key_epochs = sorted(self.points.keys())

        if optimizer:
            self._update_optimizer()

    @property
    def n_batches_seen(self):
        return self.n_items_seen / self.batch_size

    @property
    def n_epochs_seen(self):
        return self.n_items_seen / self.dset_size

    @property
    def epoch(self):
        return int(self.n_epochs_seen)

    @property
    def batch_num(self):
        return int(self.n_batches_seen)

    @property
    def n_batches_per_epoch(self):
        return self.dset_size / self.batch_size

    def reset_epoch(self, epoch):
        """
        Used when restarting after killing an epoch
        """
        n_full_batches = int(self.dset_size / self.batch_size)
        remainder = int(self.dset_size % self.batch_size)
        # n_items_seen = ([self.batch_size] * n_full_batches) + [remainder]
        n_items_seen = (self.batch_size * n_full_batches) + remainder
        for i in range(0, epoch):
            self.epoch_to_n_items_seen[i] = n_items_seen

        for i in list(self.epoch_to_n_items_seen.keys()):
            if i >= epoch:
                del self.epoch_to_n_items_seen[i]

        # self.n_items_seen = sum(map(sum, self.epoch_to_n_items_seen.values()))
        self.n_items_seen = sum(self.epoch_to_n_items_seen.values())
        self._update_optimizer()

    def step_batch(self, batch_size=None):
        """
        Args:
            batch_size (int): number of examples in the batch
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        # self.epoch_to_n_items_seen[self.epoch].append(batch_size)
        self.epoch_to_n_items_seen[self.epoch] += batch_size
        self.n_items_seen += batch_size
        self._update_optimizer()

    def step_epoch(self, epoch=None):
        # more intuitve interface (at least for me)
        epoch = epoch if epoch is not None else self.epoch + 1
        # FIXME: dont assume constant sizes
        self.n_items_seen = self.dset_size * epoch
        self._update_optimizer()

    # def step(self, epoch=None):
    #     # toch compatible interface where epoch is really last_epoch
    #     epoch = epoch - 1 if epoch is not None else epoch
    #     return self.step_epoch(epoch)

    def _update_optimizer(self):
        if self.optimizer:
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def get_lr(self):
        """ Return the current LR """
        eps = self.batch_size / self.dset_size
        lr = self._get_epoch_lr(self.epoch)
        if self.n_epochs_seen < self.burn_in:
            power = 4
            progress = min(self.n_epochs_seen / self.burn_in + eps, 1)
            lr = lr * (progress ** power)
        return lr

    def _get_epoch_lr(self, epoch):
        """ return lr based on the epoch """
        import numpy as np
        key_epochs = self.key_epochs
        points = self.points
        base_lrs = [points[0]]

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
        return epoch_lrs[0]
