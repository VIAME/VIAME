# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import numpy as np


class _Scheduler(object):
    """
    New Base object for scheduling optimizer parameters
    """

    __netharn_redesign__ = True

    def __init__(self, optimizer=None):
        self.optimizer = optimizer
        self.epoch = 0
        if self.optimizer is not None:
            self.step(self.epoch)

    def _optim_attrs(self):
        """ returns a collated dictionary of non-params optimizer attributes """
        keys = self.optimizer.param_groups[0].keys()
        return {k: _get_optimizer_values(self.optimizer, k)
                for k in keys if k != 'params'}

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self, epoch=None):
        """
        Simply updates the attributes (lr, momentum, etc.) of the optimizer to
        the current epoch.
        """
        raise NotImplementedError


class ListedScheduler(_Scheduler):
    """
    List key transitions points for every optimizer attribute.

    Notes:
        attributes for SGD are:
            'lr'
            'momentum'
            'dampening'
            'weight_decay'
            'nesterov'

    Example:
        >>> from .schedulers.scheduler_redesign import *
        >>> import ubelt as ub
        >>> from viame.pytorch import netharn as nh
        >>> import torch
        >>> import copy
        >>> model = nh.models.ToyNet2d()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0)
        >>> points = {
        >>>     'lr'       : {0: 1, 2: 21, 3: 31, 6: 61, 9: 91},
        >>>     'momentum' : {0: 2, 2: 22, 3: 32, 6: 62, 9: 92},
        >>> }
        >>> # Test SGD
        >>> self = ListedScheduler(points, optimizer=optimizer, interpolation='left')
        >>> states = []
        >>> for _ in range(4):
        ...     states.append((self.epoch, copy.deepcopy(self._optim_attrs())))
        ...     self.step()
        >>> print('states = {}'.format(ub.repr2(states, nl=1)))

        states = [
            (0, {'dampening': [0], 'lr': [1], 'momentum': [2], 'nesterov': [False], 'weight_decay': [0]}),
            (1, {'dampening': [0], 'lr': [1], 'momentum': [2], 'nesterov': [False], 'weight_decay': [0]}),
            (2, {'dampening': [0], 'lr': [21], 'momentum': [22], 'nesterov': [False], 'weight_decay': [0]}),
            (3, {'dampening': [0], 'lr': [31], 'momentum': [32], 'nesterov': [False], 'weight_decay': [0]}),
        ]

        >>> # Test ADAM
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0)
        >>> self = ListedScheduler(points, optimizer=optimizer, interpolation='linear')
        >>> states = []
        >>> for _ in range(4):
        ...     states.append((self.epoch, copy.deepcopy(self._optim_attrs())))
        ...     self.step()
        >>> print('states = {}'.format(ub.repr2(states, nl=1)))

        states = [
            (0, {'amsgrad': [False], 'betas': [(2, 0.999)], 'eps': [1e-08], 'lr': [1], 'weight_decay': [0]}),
            (1, {'amsgrad': [False], 'betas': [(12.0, 0.999)], 'eps': [1e-08], 'lr': [11.0], 'weight_decay': [0]}),
            (2, {'amsgrad': [False], 'betas': [(22, 0.999)], 'eps': [1e-08], 'lr': [21], 'weight_decay': [0]}),
            (3, {'amsgrad': [False], 'betas': [(32, 0.999)], 'eps': [1e-08], 'lr': [31], 'weight_decay': [0]}),
        ]
    """
    def __init__(self, points, interpolation='linear', optimizer=None):
        """
        Args:
            points (Dict[str, Dict[int, float]):
                dictionary mapping optimizer attribute names (e.g. lr) to a
                dictinoary mapping epochs to values.

            interpolation (str, default=linear):
                one of: left, nearest, linear,
        """
        self.points = points
        if not interpolation:
            interpolation = 'left'
        self.interpolation = interpolation
        super(ListedScheduler, self).__init__(optimizer=optimizer)

    def get_value(self, attr='lr', epoch=None):
        if epoch is None:
            epoch = self.epoch + 1
        points = self.points[attr]
        value = _interpolate(points, epoch, interpolation=self.interpolation)
        # print('attr, epoch, value = {}, {}, {}'.format(attr, epoch, value))
        return value

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.epoch + 1
        self.epoch = epoch
        for attr in self.points.keys():
            values = self.get_value(attr, epoch=epoch)
            _set_optimizer_values(self.optimizer, attr, values)


def _set_optimizer_values(optimizer, attr, value_or_values):
    """ sets the optimizer value from each param group """
    if ub.iterable(value_or_values):
        values = value_or_values
    else:
        values = [value_or_values] * len(optimizer.param_groups)
    for group, value in zip(optimizer.param_groups, values):
        if attr in group:
            group[attr] = value
        else:
            # Allow alias attr names to be used similar to fastai
            if attr == 'momentum' and 'betas' in group:
                momentum, beta = group['betas']
                group['betas'] = (value, beta)
            elif attr == 'beta' and 'betas' in group:
                momentum, beta = group['betas']
                group['betas'] = (momentum, value)
            else:
                raise KeyError(attr)


def _get_optimizer_values(optimizer, attr):
    """ gets the optimizer value from each param group """
    try:
        return [group[attr] for group in optimizer.param_groups]
    except KeyError:
        group = optimizer.param_groups[0]
        if attr == 'momentum' and 'betas' in group:
            return [group['betas'][0] for group in optimizer.param_groups]
        if attr == 'beta' and 'betas' in group:
            return [group['betas'][1] for group in optimizer.param_groups]


def _interpolate(points, x, interpolation='linear'):
    """
    Interpolates between sparse keypoints

    Args:
        points (Dict[float, float]): Mapping from x-axis to y-axis
        x (float): point on the x-axis to get a value for
        interpolation (str): how to interpolate between keypoints

    Returns:
        float: the interpolated value

    Example:
        >>> points = {0: .01, 2: .02, 3: .1, 6: .05, 9: .025}
        >>> _interpolate(points, x=1, interpolation='left')
        0.01
        >>> _interpolate(points, x=1, interpolation='linear')
        0.015
        >>> _interpolate(points, x=1.9, interpolation='nearest')
        0.02
    """
    keys = sorted(points.keys())

    if x in keys:
        key_left = x
        key_right = x
    else:
        idx = np.searchsorted(keys, x, 'left') - 1
        key_left = keys[idx]
        if idx < len(keys) - 1:
            key_right = keys[idx + 1]
        else:
            key_right = key_left
        if x < key_left:
            key_left = min(points.keys())

    if key_left == key_right:
        value = points[key_left]
    else:
        # Interpolate between the keypoints to sample an sub-point value
        if interpolation == 'left':
            value = points[key_left]
        elif interpolation == 'nearest':
            if (x - key_left) <= (key_right - x):
                value = points[key_left]
            else:
                value = points[key_right]
        elif interpolation == 'linear':
            value_left = points[key_left]
            value_right = points[key_right]
            alpha = (x - key_left) / float((key_right - key_left))
            value = alpha * value_right + (1 - alpha) * value_left
        else:
            raise KeyError(interpolation)
        # elif interpolation.startswith('log'):
        #     value_left = points[key_left]
        #     value_right = points[key_right]
        #     log = {
        #         'log': np.log,
        #         'log2': np.log2,
        #         'log10': np.log10,
        #     }[interpolation]
        #     if value_left < value_right:
        #         np.logspace(log(value_left), log(value_right))
        #     else:
        #         np.logspace(log(value_right), log(value_left))
    return value
