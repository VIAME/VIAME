# -*- coding: utf-8 -*-
r"""
Notes:
    when profiling ensure CUDA_LAUNCH_BLOCKING=1 <needs-citation>

Notes:
    to use, your training session must have the concept of:
        * epochs
        * batch_size
        * xpu
        * train / validation datasets

    or better yet:
        * a model
        * a criterion
        * an optimizer

Notes:
    In the following example we demonstrate how to use netharn to train a model
    to solve a toy problem.

    In this toy problem, we do not extend the nh.FitHarn object, so we are
    using the default behavior of ``run_batch``. The default ``on_batch``, and
    ``on_epoch`` do nothing, so only loss will be the only measurement of
    performance.

    For further examples please see the examples directory. These example show
    how to extend nh.FitHarn to measure performance wrt a particular problem.
    The MNIST and CIFAR examples are the most simple. The YOLO example is more
    complex.  The IBEIS example depends on non-public data / software, but can
    still be useful to look at.  Its complexity is more than CIFAR but less
    than YOLO.

Note:
    This file uses mixins to define :class:`FitHarn`. Mixin classes group
    related functionalities. This makes it slightly easier to navigate this
    rather large file.

CommandLine:
    xdoctest netharn.fit_harn __doc__:0
    xdoctest netharn.fit_harn __doc__:0 --debug
    xdoctest netharn.fit_harn __doc__:0 --profile --xpu=cpu

Example:
    >>> from viame.pytorch import netharn as nh
    >>> hyper = nh.HyperParams(**{
    >>>     # ================
    >>>     # Environment Components
    >>>     'workdir'     : ub.ensure_app_cache_dir('netharn/tests/demo'),
    >>>     'name'        : 'demo',
    >>>     'xpu'         : nh.XPU.coerce('argv'),
    >>>     # workdir is a directory where intermediate results can be saved
    >>>     # name symlinks <workdir>/fit/name/<name> -> ../runs/<hashid>
    >>>     # XPU auto select a gpu if idle and VRAM>6GB else a cpu
    >>>     # ================
    >>>     # Data Components
    >>>     'datasets'    : {  # dict of plain ol torch.data.Dataset instances
    >>>         'train': nh.data.ToyData2d(size=3, border=1, n=256, rng=0),
    >>>         'vali': nh.data.ToyData2d(size=3, border=1, n=64, rng=1),
    >>>         'test': nh.data.ToyData2d(size=3, border=1, n=64, rng=1),
    >>>     },
    >>>     'loaders'     : {'batch_size': 8}, # DataLoader instances or kw
    >>>     # ================
    >>>     # Algorithm Components
    >>>     # Note the (cls, kw) tuple formatting
    >>>     'model'       : (nh.models.ToyNet2d, {}),
    >>>     'optimizer'   : (nh.optimizers.SGD, {
    >>>         'lr': 0.0001
    >>>     }),
    >>>     # focal loss is usually better than nh.criterions.CrossEntropyLoss
    >>>     'criterion'   : (nh.criterions.FocalLoss, {}),
    >>>     'initializer' : (nh.initializers.KaimingNormal, {
    >>>         'param': 0,
    >>>     }),
    >>>     # these may receive an overhaul soon
    >>>     'scheduler'   : (nh.schedulers.ListedLR, {
    >>>         'points': {0: .0001, 2: .01, 5: .015, 6: .005, 9: .001},
    >>>         'interpolate': True,
    >>>     }),
    >>>     'monitor'     : (nh.Monitor, {
    >>>         'max_epoch': 10,
    >>>         'ignore_first_epochs': 2,
    >>>     }),
    >>>     # dynamics are a config option that modify the behavior of the main
    >>>     # training loop. These parameters effect the learned model.
    >>>     'dynamics'   : {'batch_step': 2},
    >>> })
    >>> harn = nh.FitHarn(hyper)
    >>> # non-algorithmic behavior configs (do not change learned models)
    >>> harn.preferences['use_tensorboard'] = False
    >>> harn.preferences['timeout'] = 0.5
    >>> harn.preferences['auto_prepare_batch'] = True
    >>> # start training.
    >>> harn.initialize(reset='delete')
    >>> harn.run()  # note: run calls initialize it hasn't already been called.
    >>> # xdoc: +IGNORE_WANT
    RESET HARNESS BY DELETING EVERYTHING IN TRAINING DIR
    Symlink: ...tests/demo/fit/runs/demo/keyeewlr -> ...tests/demo/fit/name/demo
    .... already exists
    .... and points to the right place
    Initializing tensorboard (dont forget to start the tensorboard server)
    Model has 824 parameters
    Mounting ToyNet2d model on CPU
    Initializing model weights
     * harn.train_dpath = '...tests/demo/fit/runs/demo/keyeewlr'
     * harn.name_dpath  = '...tests/demo/fit/name/demo'
    Snapshots will save to harn.snapshot_dpath = '...tests/demo/fit/runs/demo/keyeewlr/torch_snapshots'
    dont forget to start:
        tensorboard --logdir ...tests/demo/fit/name
    === begin training ===
    epoch lr:0.001 │ vloss: 0.1409 (n_bad_epochs=00, best=0.1409): 100%|█| 10/10 [00:01<00:00,  9.95it/s]  0:00<?, ?it/s]
    train x64 │ loss:0.147 │: 100%|███████████████████████████████████████████████████████| 8/8 [00:00<00:00, 130.56it/s]
    vali x64 │ loss:0.140 │: 100%|████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 342.04it/s]
    test x64 │ loss:0.140 │: 100%|████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 342.92it/s]
    <BLANKLINE>
    Maximum harn.epoch reached, terminating ...
    <BLANKLINE>
    training completed
    current lrs: [0.001]
    harn.train_dpath = '...tests/demo/fit/runs/demo/keyeewlr'
    harn.name_dpath  = '...tests/demo/fit/name/demo'
    view tensorboard results for this run via:
        tensorboard --logdir ...tests/demo/fit/name
    exiting fit harness.

TODO:
    [X] - output "monitor" curves to disk
    [x] - move logs to a logs folder. Keep a single master log in the root
    [ ] - log learning rate an a per-batch basis
    [ ] - ability to run an iteration of the validation data within an epoch,
          perhaps we could allow the user to redefine how long an epoch is.

    [ ] - Update for torch 1.1 lr scheduler behavior. Allow schedulers to be
          called either after each epoch or after each batch iteration (for
          schedulers like CyclicLR, OneCycleLR).

          [X] - Show LR in the batch progress bar (if updated on an iteration basis)
          [ ] - How does the netharn scheduler redesign interact with torch 1.1?
          [ ] - Stochastic Weight Averaging - https://pytorch.org/docs/stable/optim.html#putting-it-all-together

"""
import glob
import itertools as it
import logging
import parse
import shutil
import time
import sys
import warnings
import functools
import traceback
from os.path import join
from os.path import exists
from os.path import dirname

import torch
import numpy as np
import ubelt as ub

import scriptconfig as scfg
import torch_liberator

from viame.pytorch.netharn import hyperparams
from viame.pytorch.netharn import util
from viame.pytorch.netharn.util import profiler
from viame.pytorch.netharn.util import strip_ansi
from viame.pytorch.netharn.exceptions import (CannotResume, SkipBatch, StopTraining,
                                              TrainingDiverged)

from packaging.version import parse as Version


# Hack: patch collections so tensorboard_logger doesnt die
from viame.pytorch.netharn import monkey  # NOQA
try:
    import tensorboard_logger
except ImportError:
    tensorboard_logger = None


__all__ = ['FitHarn', 'FitHarnPreferences']


# Debugging flag to run your harness in "demo mode" which only runs DEMO=5
# epochs with DEMOBX=2 batches each.
DEMO = None
DEMO_BX = None
try:
    DEMO = int(ub.argval(('--demo', '--dummy')))
except Exception:
    pass
if DEMO is None:
    if ub.argflag('--demo') or ub.argflag('--dummy'):
        DEMO = 5
if DEMO is not None:
    DEMO_BX = int(ub.argval('--demobx', default=2))


MIXINS = []  # FitHarn will have the methods of every registered mixin class


def register_mixin(cls):
    """ decorator that marks that a class is part of FitHarn """
    MIXINS.append(cls)
    return cls


def _disjoint_dict_update(a, b):
    """
    Equivalent to a.update(b), but raises KeyError if a and b are not disjoint
    """
    if b:
        isect = set(a).intersection(set(b))
        if isect:
            raise KeyError('Conflicting keys: {}'.format(isect))
        a.update(b)


@register_mixin
class ExtraMixins(object):
    """
    Miscellaneous methods that will be mixed into FitHarn
    """

    @classmethod
    def demo(cls):
        """
        Creates a dummy FitHarn object for testing and demonstration purposes
        """
        from viame.pytorch import netharn as nh
        hyper = nh.HyperParams(**{
            # ================
            # Environment Components
            'workdir'     : ub.ensure_app_cache_dir('netharn/tests/demo'),
            'name'        : 'demo',
            'xpu'         : nh.XPU.coerce('cpu'),
            # workdir is a directory where intermediate results can be saved
            # "name" symlinks <workdir>/fit/name/<name> -> ../runs/<hashid>
            # XPU auto select a gpu if idle and VRAM>6GB else a cpu
            # ================
            # Data Components
            'datasets'    : {  # dict of plain ol torch.data.Dataset instances
                'train': nh.data.ToyData2d(size=3, border=1, n=256, rng=0),
                'vali': nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
                'test': nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
            },
            'loaders'     : {'batch_size': 64},  # DataLoader instances or kw
            # ================
            # Algorithm Components
            # Note the (cls, kw) tuple formatting
            'model'       : (nh.models.ToyNet2d, {}),
            'optimizer'   : (nh.optimizers.SGD, {
                'lr': 0.0001
            }),
            # focal loss is usually better than nh.criterions.CrossEntropyLoss
            'criterion'   : (nh.criterions.FocalLoss, {}),
            'initializer' : (nh.initializers.KaimingNormal, {
                'param': 0,
            }),
            # these may receive an overhaul soon
            'scheduler'   : (nh.schedulers.ListedLR, {
                'points': {0: .0001, 2: .01, 5: .015, 6: .005, 9: .001},
                'interpolate': True,
            }),
            'monitor'     : (nh.Monitor, {
                'max_epoch': 10,
            }),
            # dynamics are a config option that modify the behavior of the main
            # training loop. These parameters effect the learned model.
            'dynamics'   : {'batch_step': 4},
        })
        harn = cls(hyper)
        # non-algorithmic behavior configs (do not change learned models)
        harn.preferences['use_tensorboard'] = False
        harn.preferences['timeout'] = 0.5
        return harn

    def _demo_epoch(harn, tag='vali', learn=False, max_iter=np.inf,
                    call_on_epoch=False):
        """
        Runs an epoch (usually for testing / demo purposes).

        Args:
            tag (str, default='vali'):
                specifies the data split (e.g. train, vali, test)

            learn (bool, default=False):
                by default demo epochs do not update model weights even on the
                training dataset.

            max_iter (int, default=inf):
                Limits the number of batches to be less than `max_iter`

            call_on_epoch (bool, default=False):
                by default demo_epoch does not call the `on_epoch` callback

        Returns:
            Dict[str, float]: epoch_metrics: metrics computed on this epoch
        """
        harn.current_tag = None
        harn._run_metrics = {
            tag: util.WindowedMovingAve(window=len(loader))
            for tag, loader in harn.loaders.items()
        }
        loader = harn.loaders[tag]
        epoch_metrics = harn._run_epoch(loader, tag=tag, learn=learn,
                                        max_iter=max_iter,
                                        call_on_epoch=call_on_epoch)
        return epoch_metrics

    def _demo_batch(harn, index=0, tag='train', raw=False):
        """
        Returns a single batch for testing / demo purposes.

        Additionally, sets harn.current_tag to `tag`.

        Args:
            index (int): get the `index`-th batch
            tag (str): get batch from either train, vali, or test loader
            raw (bool): if True, does not prepare the batch

        Returns:
            object: output of the data loader
        """
        loader = harn.loaders[tag]
        harn.current_tag = tag
        for bx, batch in enumerate(iter(loader)):
            if bx >= index:
                break
        if raw:
            return batch
        else:
            return harn.prepare_batch(batch)

    def _check_thread_safety(harn):
        """
        References:
            https://github.com/pytorch/pytorch/issues/1355
        """
        n_workers = max(loader.num_workers for loader in harn.loaders.values()
                        if loader is not None)
        if n_workers > 1:
            try:
                import cv2
            except ImportError:
                pass
            else:
                n_threads = cv2.getNumThreads()
                if n_threads > 1:
                    msg = ('OpenCV threadcount of {} is non-zero and a DataLoader '
                           'is using {} workers. This may cause deadlocks '
                           'To be safe use cv2.setNumThreads(0)').format(
                               n_threads, n_workers)
                    warnings.warn(msg, RuntimeWarning)
                    harn.warn(msg)


@register_mixin
class InitializeMixin(object):
    """
    Methods for initializing logging, models, etc...
    """

    @profiler.profile
    def initialize(harn, reset=False, overwrite=True):
        """
        Uses the hyper parameters to initialize the necessary resources and
        restart from previous state if possible.

        Creating your model and mounting it on the XPU should be the only
        significant contributors to runtime. We have temporarilly disabled
        export on initialization until we can resolve a its speed issue.

        Args:
            reset (bool, default=False):
                if True the training directory is deleted and the entire
                run will be redone from scratch.

            overwrite (bool, default=True):
                if False, initialization will not overwrite existing
                configuration or log files. (however other parts of the harness
                might). This is mostly a debugging option.
        """
        if reset == 'delete':
            print('RESET HARNESS BY DELETING EVERYTHING IN TRAINING DIR')
            if harn.train_info is None:
                # Need to determine which path needs deletion.
                harn._setup_paths_and_train_info()
            for path in glob.glob(join(harn.train_dpath, '*')):
                ub.delete(path)
        elif reset:
            print('RESET HARNESS BY RESTARTING FROM EPOCH 0')

        if harn.train_info is None:
            harn._setup_paths_and_train_info()
        else:
            ub.ensuredir(harn.train_dpath)

        # Dump training info to disk
        # - [X] if train_info already exists, and it is not the same as this
        # train info, keep a backup of the old ones.
        if harn.train_dpath and overwrite:
            train_info_fpath = join(harn.train_dpath, 'train_info.json')
            if exists(train_info_fpath):
                if overwrite:
                    import json
                    try:
                        old_train_info = util.read_json(train_info_fpath)
                    except json.JSONDecodeError:
                        old_train_info = {}
                    if old_train_info != harn.train_info:
                        backup_dpath = ub.ensuredir((harn.train_dpath, '_backup'))
                        backup_fpath = join(backup_dpath, 'train_info.json.' + ub.timestamp() + '.backup')
                        shutil.move(train_info_fpath, backup_fpath)
                    util.write_json(train_info_fpath, harn.train_info)
            else:
                try:
                    util.write_json(train_info_fpath, harn.train_info)
                except Exception as ex:
                    harn.warn('Unable to write train info ex = {!r}'.format(ex))

        harn._setup_loggers(overwrite=overwrite)

        harn._setup_modules()

        if harn.model is None:
            raise ValueError('model is a required module')

        # TODO: we might simply default to SGD
        if harn.optimizer is None:
            raise ValueError('optimizer is a required module')

        # TODO: we could probably default the monitor to something reasonable
        if harn.monitor is None:
            raise ValueError('monitor is a required module')

        try:
            if reset:
                raise CannotResume
            harn.resume_from_previous_snapshots()
        except CannotResume:
            # This step is only run on a fresh start.
            harn.reset_weights()
            for group in harn.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        if harn.train_dpath:
            harn.info(' * harn.train_dpath = {!r}'.format(harn.train_dpath))
            harn.info(' * harn.name_dpath  = {!r}'.format(harn.name_dpath))
            harn.info('Snapshots will save to harn.snapshot_dpath = {!r}'.format(
                harn.snapshot_dpath))
        else:
            harn.warn('harn.train_dpath is None, all computation is in memory')

        if isinstance(harn.preferences['timeout'], str):
            import datetime
            import parse
            text = harn.preferences['timeout']

            def parse_timedelta_text(text):
                candidate_formats = [
                    '{microseconds:d}us',
                    '{milliseconds}ms',
                    '{minutes}m',
                    '{seconds}s',
                    '{hours:d}h',
                    '{days:d}d',
                    '{weeks:d}w',
                ]
                text_ = text.lower()
                for fmt in candidate_formats:
                    result = parse.parse(fmt, text_)
                    if result is not None:
                        delta = datetime.timedelta(**result.named)
                        break
                if delta is None:
                    raise Exception('Unknown time format {}'.format(text))
                return delta

            delta = parse_timedelta_text(text)
            print('delta = {!r}'.format(delta))
            harn.preferences['timeout'] = delta.total_seconds()

        harn._initialized = True
        harn.after_initialize()
        return harn

    @profiler.profile
    def _setup_paths_and_train_info(harn):
        if harn.hyper is None:
            harn.warn('harn.train_dpath is None, cannot setup_paths')
        else:
            # TODO: we may fold the functionality of Folders into Hyperparams
            train_info = harn.hyper.train_info(harn.train_dpath)
            ub.ensuredir(train_info['train_dpath'])

            if train_info['name_dpath']:
                ub.ensuredir(dirname(train_info['name_dpath']))

                # Make a very simple MRU (most recently used) link
                mru_dpath = join(harn.hyper.workdir, '_mru')
                try:
                    ub.symlink(train_info['train_dpath'], mru_dpath,
                               overwrite=True, verbose=0)
                except OSError as ex:
                    harn.warn('Unable to symlink: {!r}'.format(ex))

                # Link the hashed run dir to the human friendly "name" dir
                try:
                    ub.symlink(train_info['train_dpath'],
                               train_info['name_dpath'], overwrite=True,
                               verbose=0)
                except OSError as ex:
                    harn.warn('Unable to symlink: {!r}'.format(ex))

            if 'nice_dpath' in train_info:
                # backwards compatibility for "nice" dpaths
                ub.ensuredir(dirname(train_info['nice_dpath']))
                try:
                    ub.symlink(train_info['train_dpath'],
                               train_info['nice_dpath'], overwrite=True,
                               verbose=0)
                except OSError as ex:
                    harn.warn('Unable to symlink: {!r}'.format(ex))

            harn.train_info = train_info
            harn.name_dpath = train_info['name_dpath']
            harn.train_dpath = train_info['train_dpath']
            return harn.train_dpath

    @profiler.profile
    def _setup_loggers(harn, overwrite=True):
        """
        Setup file logging and / or tensorboard logging
        """
        if harn.train_dpath is None:
            harn.warn('harn.train_dpath is None, cannot setup loggers')
            return

        use_py_logger = True
        if use_py_logger and harn._log is None:

            _log = logging.getLogger(harn.__class__.__name__ + ':' + str(id(harn)))
            _log.propagate = False
            _log.setLevel(logging.DEBUG)

            f_formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
            s_formatter = logging.Formatter('%(levelname)s: %(message)s')

            # Add timestamped fpath write handler:
            # This file will be specific to this instance of the harness, which
            # means different intances of the harness wont clobber value here.
            flog_fname = 'fitlog_{}.log'.format(ub.timestamp())
            flog_dpath = ub.ensuredir(join(harn.train_dpath, 'logs'))
            w_flog_fpath = join(flog_dpath, flog_fname)
            w_handler = logging.FileHandler(w_flog_fpath, mode='w')
            w_handler.setFormatter(f_formatter)
            w_handler.setLevel(logging.DEBUG)

            # Add a simple root append handler:
            # This file is shared by all instances of the harness, so logs over
            # multiple starts and stops can be viewed in a consolidated file.
            if overwrite:
                a_flog_fpath = join(harn.train_dpath, 'fit.log')
                a_handler = logging.FileHandler(a_flog_fpath, mode='a')
                a_handler.setFormatter(f_formatter)
                a_handler.setLevel(logging.DEBUG)

            # Add a stdout handler:
            # this allows us to print logging calls to the terminal
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(s_formatter)

            if (harn.preferences['verbose'] > 1 or
                  ub.argflag(('--verbose', '--debug'))):
                stdout_handler.setLevel(logging.DEBUG)
            else:
                stdout_handler.setLevel(logging.INFO)

            _log.addHandler(w_handler)
            if overwrite:
                _log.addHandler(a_handler)
            _log.addHandler(stdout_handler)

            # hack in attribute for internal use
            _log._stdout_handler = stdout_handler

            harn._log = _log
            harn.debug('Initialized logging')

        if tensorboard_logger and harn.preferences['use_tensorboard']:
            # train_base = dirname(harn.name_dpath or harn.train_dpath)
            # harn.info('dont forget to start:\n    tensorboard --logdir ' + train_base)
            harn.info('Initializing tensorboard (dont forget to start the tensorboard server)')
            harn._tlog = tensorboard_logger.Logger(harn.train_dpath,
                                                   flush_secs=2)
        else:
            # TODO:
            # - [ ] setup an alternative database to record epoch measures for
            # plotting if tensorboard is not available.
            harn._tlog = None
            if tensorboard_logger is None:
                harn.warn('Tensorboard is not available')
            else:
                harn.debug('Tensorboard is disabled')

    @profiler.profile
    def _setup_modules(harn):
        """
        Construts the basic modules to be used by the harness, i.e:
            loaders, xpu, model, criterion, optimizer, initializer, scheduler,
            monitor, and dynamics.
        """
        if harn.hyper is None:
            raise ValueError(
                'Hyperparameters not specified, must setup modules yourself')

        harn.debug('harn.train_info[hyper] = {}'.format(ub.repr2(harn.train_info['hyper'], nl=3)))
        harn.debug('harn.hyper = {!r}'.format(harn.hyper))

        harn.debug('make XPU')
        harn.xpu = harn.hyper.make_xpu()
        harn.debug('harn.xpu = {!r}'.format(harn.xpu))
        harn.xpu.set_as_default()

        if harn.hyper.criterion_cls:
            harn.debug('Criterion: {}'.format(harn.hyper.criterion_cls.__name__))
        else:
            harn.debug('Criterion: Custom')

        harn.debug('Optimizer: {}'.format(harn.hyper.optimizer_cls.__name__))

        if harn.hyper.scheduler_cls:
            harn.debug('Scheduler: {}'.format(harn.hyper.scheduler_cls.__name__))
        else:
            harn.debug('No Scheduler')

        harn.debug('Making loaders')
        harn.datasets = harn.hyper.datasets
        harn.loaders = harn.hyper.make_loaders()

        for key, dset in harn.datasets.items():
            if dset is not None:
                harn.debug('len(harn.datasets[{}]) = {}'.format(key, len(dset)))
            else:
                harn.debug('harn.datasets[{}] = {}'.format(key, dset))

        for key, loader in harn.loaders.items():
            if loader is not None:
                harn.debug('len(harn.loaders[{}]) = {}'.format(key, len(loader)))
            else:
                harn.debug('harn.loaders[{}] = {}'.format(key, loader))

        harn.debug('Making model')
        harn.model = harn.hyper.make_model()
        harn.debug(harn.model)

        n_params = util.number_of_parameters(harn.model)
        harn.info('Model has {!r} parameters'.format(n_params))

        harn.info('Mounting {} model on {}'.format(
            harn.model.__class__.__name__, harn.xpu))
        harn.model = harn.xpu.mount(harn.model)

        harn.debug('Making initializer')
        harn.initializer = harn.hyper.make_initializer()

        harn.criterion = harn.hyper.make_criterion()
        if harn.criterion is not None:
            harn.debug('Move {} model to {}'.format(harn.criterion, harn.xpu))
            harn.criterion = harn.xpu.move(harn.criterion)

        harn.debug('Make optimizer')
        # TODO: allow for "discriminative fine-tuning"
        # References:
        # https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10
        # https://arxiv.org/pdf/1801.06146.pdf
        # https://discuss.pytorch.org/t/implementing-differential-learning-rate-by-parameter-groups/32903
        harn.optimizer = harn.hyper.make_optimizer(harn.model.named_parameters())

        harn.debug('Make scheduler')
        # Note: this will usually overwrite any default LR in the optimizer
        harn.scheduler = harn.hyper.make_scheduler(harn.optimizer)

        harn.debug('Make monitor')
        harn.monitor = harn.hyper.make_monitor()

        harn.debug('Make dynamics')
        harn.dynamics = harn.hyper.dynamics.copy()

        # TODO: export can be slow, do it in a separate process
        if harn.preferences['export_on_init']:
            harn._export()

    @profiler.profile
    def reset_weights(harn):
        """
        Use the initializer to set the weights for the model
        """
        harn.info('Initializing model weights with: {}'.format(harn.initializer))
        if harn.initializer:
            if harn.initializer.__class__.__name__ == 'LSUV':
                harn.debug('calling hacked LSUV initializer')
                #hack LSUV needs a batch of data to run
                with torch.set_grad_enabled(False):
                    loader = harn.loaders['train']
                    input, labels = next(iter(loader))
                    data = harn.xpu.move(input)
                    harn.initializer(harn.model, data)
            else:
                harn.debug('calling harn.initializer={!r}'.format(
                    harn.initializer))
                raw_model = harn.xpu.raw(harn.model)
                harn.initializer(raw_model)
        else:
            harn.warn('initializer was not specified')

        # Save the original weights for analysis
        harn.save_snapshot(mode='initial')

    @profiler.profile
    def resume_from_previous_snapshots(harn):
        """
        Attempts to load one of the states in prev_states.

        This will load states of the model, optimizer, scheduler, monitor, and
        any other custom structure defined in `set_snapshot_state`.
        """
        if harn.train_dpath is None:
            raise CannotResume('harn.train_dpath is None')

        prev_states = harn.prev_snapshots()
        harn.info('There are {} existing snapshots'.format(len(prev_states)))
        if not prev_states:
            raise CannotResume('no previous snapshots')

        harn.info('Loading previous states')
        success = False
        # Ignore corrupted snapshots
        for load_path in reversed(prev_states):
            try:
                harn.load_snapshot(load_path)
            except (RuntimeError, EOFError, Exception):
                harn.info('Failed to load {}. Skiping.'.format(load_path))
                harn.info('NOTE: This will sometimes cause torch to crash. Delete the skipped file if it does')
            else:
                success = True
                break
        if not success:
            raise CannotResume('Previous snapshots are invalid or corrupted')

        for i, group in enumerate(harn.optimizer.param_groups):
            if 'initial_lr' not in group:
                raise KeyError(
                    'param "initial_lr" is not specified '
                    'in param_groups[{}] when resuming an optimizer'.format(i))

        harn.info('Resuming from epoch={}'.format(harn.epoch))


@register_mixin
class ProgMixin(object):
    """
    Methods for displaying progress bars
    """

    def _make_prog(harn, *args, **kw):
        chunksize = kw.pop('chunksize', None)
        show_wall = kw.pop('show_wall', False)

        if harn.preferences['use_tqdm'] is not None:
            import warnings
            warnings.warn('use_tqdm is deprecated. Set prog_backend instead')
            harn.preferences['prog_backend'] = (
                'tqdm' if harn.preferences['use_tqdm'] else 'progiter')

        if harn.preferences['prog_backend'] == 'auto':
            try:
                import tqdm
            except ImportError:
                harn.preferences['prog_backend'] = 'progiter'
            else:
                harn.preferences['prog_backend'] = 'tqdm'

        if harn.preferences['prog_backend'] == 'tqdm':
            import tqdm  # NOQA
            Prog = tqdm.tqdm
        elif harn.preferences['prog_backend'] == 'progiter':
            if Version(ub.__version__) >= Version('0.9.3'):
                Prog = functools.partial(
                    ub.ProgIter, chunksize=chunksize, verbose=1,
                    time_thresh=2.0, show_wall=show_wall)
            else:
                Prog = functools.partial(
                    ub.ProgIter, chunksize=chunksize, verbose=1,
                    time_thresh=2.0)
        else:
            raise KeyError(harn.preferences['prog_backend'])
        return Prog(*args, **kw)

    def _batch_msg(harn, metric_dict, batch_size, learn=False):
        """
        Args:
            metric_dict (dict): metrics to be reported in the message
            batch_size (int): size of the current batch
            learn (bool): formats a message for train or vali/test.

        Returns:
            str : the message to be used in the progress bar
        """
        parts = ['{}:{:.4g}'.format(k, v) for k, v in metric_dict.items()]

        if learn and harn.epoch == 0:
            HACK_WARMUP = bool(harn.dynamics['warmup_iters'])
            if HACK_WARMUP:
                lrs = set(harn._current_lrs())
                lr_str = ','.join(['{:.4g}'.format(lr) for lr in lrs])
                parts.append('lr=' + lr_str)

        if harn.preferences['prog_backend'] == 'progiter':
            if learn and harn.scheduler and getattr(harn.scheduler, '__batchaware__', False):
                lr = harn.scheduler.get_lr()
                bs = '@ {:.4g}'.format(lr)
                parts = [bs] + parts
        else:
            if learn and harn.scheduler and getattr(harn.scheduler, '__batchaware__', False):
                lr = harn.scheduler.get_lr()
                bs = 'x{} @ {:.4g}'.format(batch_size, lr)
            else:
                bs = 'x{}'.format(batch_size)
            parts = [bs] + parts
        if not harn.preferences['allow_unicode']:
            msg = ' | ' .join(parts) + ' |'
        else:
            msg = ' │ ' .join(parts) + ' │'
        return msg

    def _close_prog(harn):
        if harn.main_prog is not None:
            harn.main_prog.close()
            harn.main_prog = None
            sys.stdout.write('\n\n\n\n')  # fixes progress bar formatting

    def _update_prog_postfix(harn, prog):
        if harn.preferences['prog_backend'] == 'tqdm':
            prog.set_postfix({
                'wall': time.strftime('%h:%m') + ' ' + time.tzname[0]
            })

    def _update_main_prog_desc(harn):
        lrs = set(harn._current_lrs())
        lr_str = ','.join(['{:.4g}'.format(lr) for lr in lrs])
        if not harn.preferences['allow_unicode']:
            desc = 'epoch lr:{} | {}'.format(lr_str, harn.monitor.message())
        else:
            desc = 'epoch lr:{} │ {}'.format(lr_str, harn.monitor.message())
        if not harn.preferences['colored']:
            desc = strip_ansi(desc)
        harn.main_prog.set_description(desc, refresh=False)
        if isinstance(harn.main_prog, ub.ProgIter):
            # Write progress message to the log file
            harn.debug(harn.main_prog.format_message().strip())
            if not harn.main_prog.started:
                # harn.main_prog.ensure_newline()
                harn.main_prog.clearline = False
                harn.main_prog.freq = 1
                harn.main_prog.adjust = False
                harn.main_prog.begin()
        else:
            harn.debug(desc)
            harn._update_prog_postfix(harn.main_prog)


@register_mixin
class LogMixin(object):
    """
    Methods for logging messages and data within FitHarn.
    """

    def _ensure_prog_newline(harn):
        # Try and make sure the progress bar does not clobber log outputs.
        # Only available with progiter. Not sure how to do with tqdm.
        try:
            if harn.epoch_prog is not None:
                harn.epoch_prog.ensure_newline()
            if harn.main_prog is not None:
                harn.main_prog.ensure_newline()
        except AttributeError:
            pass

    def log(harn, msg, level='info'):
        """
        Logs a message with a specified verbosity level.

        Args:
            msg (str): an info message to log
            level (str): either info, debug, error, or warn
        """
        if level == 'info':
            harn.info(msg)
        elif level == 'debug':
            harn.debug(msg)
        elif level == 'error':
            harn.error(msg)
        elif level == 'warn':
            harn.warn(msg)
        else:
            raise KeyError(level)

    def info(harn, msg):
        """
        Writes an info message to the logs

        Args:
            msg (str): an info message to log
        """
        if not harn.preferences['colored']:
            msg = strip_ansi(msg)
        harn._ensure_prog_newline()
        if harn._log:
            try:
                harn._log.info(msg)
            except Exception:
                pass
        else:
            print(msg)

    def error(harn, msg):
        """
        Writes an error message to the logs

        Args:
            msg (str): an error message to log
        """
        harn._ensure_prog_newline()
        if harn._log:
            msg = strip_ansi(msg)
            harn._log.error(msg)
        else:
            if not harn.preferences['colored']:
                msg = strip_ansi(msg)
            print(msg)

    def warn(harn, msg):
        """
        Writes a warning message to the logs

        Args:
            msg (str): a warning message to log
        """
        harn._ensure_prog_newline()
        if harn._log:
            msg = strip_ansi(msg)
            harn._log.warning(msg)
        else:
            if not harn.preferences['colored']:
                msg = strip_ansi(msg)
            print(msg)

    def debug(harn, msg):
        """
        Writes a debug message to the logs

        Args:
            msg (str): a debug message to log
        """
        if harn._log:

            if harn._log._stdout_handler.level <= logging.DEBUG:
                # Use our hacked attribute to ensure newlines if we are
                # writting debug info to stdout
                harn._ensure_prog_newline()

            msg = strip_ansi(str(msg))
            # Encode to prevent errors on windows terminals
            # On windows there is a sometimes a UnicodeEncodeError:
            # For more details see: https://wiki.python.org/moin/PrintFails
            if sys.platform.startswith('win32'):
                harn._log.debug(msg.encode('utf8'))
            else:
                harn._log.debug(msg)
            # except UnicodeEncodeError:
            #     stripped = ''.join(c if ord(c) < 128 else ' ' for c in msg)
            #     harn._log.debug('[UnicodeEncodeError]: ' + stripped)

    def log_value(harn, key, value, n_iter):
        """
        Records a scalar value to the logfile and tensorboard if available

        Args:
            key (str): identifier for your plot, good practice to include
               dataset tag and if it is an epoch or iter measurement.
            value (float): a scalar value
            n_iter (int): the current epoch or iteration number.
        """
        if harn._tlog:
            harn._tlog.log_value(key, value, n_iter)
        harn.debug('log_value({}, {}, {})'.format(key, value, n_iter))

    def log_histogram(harn, key, value, n_iter):
        """
        Records a histogram to tensorboard if available

        Args:
            key (str): identifier for your plot, good practice to include
               dataset tag and if it is an epoch or iter measurement.
            value (ndarray or tuple): either an array of data to compute
               histogram on, or a tuple of bins and counts.
            n_iter (int): the current epoch or iteration number.
        """
        if harn._tlog:
            # is this necessary?
            # if isinstance(value, np.ndarray):
            #     bins, counts = np.histogram(value)
            #     value = (bins, counts)
            harn._tlog.log_histogram(key, value, n_iter)
            harn.debug(
                'log histogram to tensorboard: {}, {}'.format(key, n_iter))
        else:
            harn.warn('cannot log histogram without tensorboard: {}, {}'.format(key, n_iter))

    def log_images(harn, key, value, n_iter):
        """
        Record an image to tensorboard if available

        Args:
            key (str): identifier for your plot, good practice to include
               dataset tag and if it is an epoch or iter measurement.
            value (ndarray): an image
            n_iter (int): the current epoch or iteration number.
        """
        if harn._tlog:
            harn._tlog.log_images(key, value, n_iter)
            harn.debug(
                'log image to tensorboard: {}, {}'.format(key, n_iter))
        else:
            harn.warn('cannot log image without tensorboard: {}, {}'.format(key, n_iter))


@register_mixin
class SnapshotMixin(object):
    """
    Methods for serializing the state of training.
    """

    @property
    def snapshot_dpath(harn):
        """
        Returns:
            str : path to the snapshot directory
        """
        # TODO: we should probably change the name of this directory to either
        # snapshots or checkpoints for simplicity.
        if harn.train_dpath is None:
            raise ValueError('harn.train_dpath is None')
        # return join(harn.train_dpath, 'torch_snapshots')
        return join(harn.train_dpath, 'checkpoints')

    def _epochs_to_remove(harn, existing_epochs, num_keep_recent,
                          num_keep_best, keep_freq):
        """
        Unit testable helper for `cleanup_snapshots`. Determines which epochs
        to remove given which epoches exist.

        Keeps `num_keep_recent` most recent, `num_keep_best` best, and one
        every `keep_freq` epochs.

        Returns:
            set: epoch numbers to remove

        Doctest:
            >>> from viame.pytorch import netharn as nh
            >>> harn = FitHarn({})
            >>> rng = np.random.RandomState(0)
            >>> harn.monitor = nh.Monitor(minimize=['loss'], maximize=['miou'])
            >>> for epoch in range(200):
            >>>     harn.monitor.update(epoch, {'loss': rng.rand(),
            >>>                                 'miou': rng.rand()})
            >>> existing_epochs = list(range(0, 200, 4))
            >>> num_keep_best = 10
            >>> num_keep_recent = 10
            >>> keep_freq = 10
            >>> to_remove = harn._epochs_to_remove(existing_epochs,
            >>>                                    num_keep_recent, num_keep_best,
            >>>                                    keep_freq)
            >>> assert len(existing_epochs) - len(to_remove) < 40
        """
        keep = set()

        recent = existing_epochs[-num_keep_recent:]
        keep.update(recent)

        # TODO: add a config for always keeping specific iterations in
        # multiples of X.

        if harn.monitor:
            for best_epochs in harn.monitor.best_epochs().values():
                best = ub.oset(best_epochs).intersection(existing_epochs)
            keep.update(best[:num_keep_best])

        # Keep a strided sampling of epochs
        epoch_arr = np.array(existing_epochs)
        flags = ((epoch_arr % keep_freq) == 0)
        sampled = epoch_arr[flags]
        keep.update(sampled)

        to_remove = set(existing_epochs) - keep
        return to_remove

    def cleanup_snapshots(harn):
        """
        Remove old snapshots according to configuration

        Notes:
            Influenced by 'num_keep' - the number of top ranking snapshots to
            keep, and 'keep_freq' - the number of intermitent snapshots to keep
        """
        snapshots = harn.prev_snapshots()
        existing_epochs = sorted([
            int(parse.parse('{}_epoch_{num:d}.pt', path).named['num'])
            for path in snapshots
        ])

        num_keep_recent = harn.preferences['num_keep']
        num_keep_best = harn.preferences['num_keep']
        keep_freq = harn.preferences['keep_freq']

        epoch_to_fpath = dict(zip(existing_epochs, snapshots))
        to_remove = harn._epochs_to_remove(existing_epochs, num_keep_recent,
                                           num_keep_best, keep_freq)
        for fpath in ub.take(epoch_to_fpath, to_remove):
            ub.delete(fpath)

    def backtrack_weights(harn, epoch):
        """
        Reset the weights to a previous good state

        Args:
            epoch (int): the epoch to backtrack to
        """
        load_path = join(harn.snapshot_dpath, '_epoch_{:08d}.pt'.format(epoch))
        snapshot = harn.xpu.load(load_path)

        harn.info('\n\n\n\n')
        harn.info('Backtracking to weights from previous state: {}'.format(load_path))
        # only load the model state to simulate a big step back
        harn.model.load_state_dict(snapshot['model_state_dict'])
        harn.optimizer.zero_grad()

    def prev_snapshots(harn):
        ub.ensuredir(harn.snapshot_dpath)
        prev_states = sorted(glob.glob(join(harn.snapshot_dpath, '_epoch_*.pt')))
        return prev_states

    def load_snapshot(harn, load_path):
        """
        Sets the harness to its state just after an epoch finished

        Args:
            str: path to previously saved snapshot
        """
        harn.info('Loading previous state: {}'.format(load_path))
        snapshot_state = harn.xpu.load(load_path)
        harn.set_snapshot_state(snapshot_state)
        harn.info('Previous snapshot loaded...')

    def save_snapshot(harn, explicit=False, mode='checkpoint'):
        """
        Checkpoint the current model state in an epoch-tagged snapshot.

        Args:
            mode (str, default='checkpoint'): the type of snapshot this is
                (changes the subdirectory where they are stored). Choices
                are: checkpoint, explicit, and initial.

            explicit (bool, default=False): if True, the snapshot is also
                tagged by a hash and saved to the explicit_checkpoints directory.
                DEPRECTATED, use mode.

        Returns:
            PathLike: save_fpath: the path to the saved snapshot

        Example:
            >>> from viame.pytorch import netharn as nh
            >>> harn = nh.FitHarn.demo()
            >>> # The "save_snapshot" method is called in initialize
            >>> harn.initialize()
        """
        if explicit:
            mode = 'explicit'

        if mode == 'explicit':
            dpath = ub.ensuredir((harn.train_dpath, 'explicit_checkpoints'))
            stamp = ub.timestamp()
            save_fname = '_epoch_{:08d}_{}.pt'.format(harn.epoch, stamp)
        elif mode == 'checkpoint':
            # TODO: make the transition smoother
            dpath = ub.ensuredir(harn.snapshot_dpath)
            _old_snapshot_dpath = join(harn.train_dpath, 'torch_snapshots')
            _new_snapshot_dpath = join(harn.train_dpath, 'checkpoints')

            if dpath == _new_snapshot_dpath:
                if not exists(_old_snapshot_dpath):
                    ub.symlink(_new_snapshot_dpath, _old_snapshot_dpath)

            save_fname = '_epoch_{:08d}.pt'.format(harn.epoch)
        elif mode == 'initial':
            dpath = ub.ensuredir((harn.train_dpath, 'initial_state'))
            save_fname = 'initial_state.pt'.format()
        else:
            raise KeyError(mode)

        save_fpath = join(dpath, save_fname)
        level = 'debug' if mode == 'checkpoint' else 'info'
        harn.log('Saving {} snapshot to {}'.format(mode.upper(), save_fpath), level)

        snapshot_state = harn.get_snapshot_state()

        try:
            import safer
            _open = safer.open
        except ImportError:
            _open = open

        with _open(save_fpath, 'wb') as save_file:
            torch.save(snapshot_state, save_file)

        harn.debug('Snapshot saved to {}'.format(save_fpath))
        return save_fpath

    def best_snapshot(harn):
        """
        Return the path to the current "best" snapshot.

        Returns:
            str - find the path to the best
        """
        # Netharn should populate best_snapshot.pt if there is a validation set.
        # Other names are to support older codebases.
        train_dpath = harn.train_dpath
        expected_names = [
            'best_snapshot.pt',
            'best_snapshot2.pt',
            'final_snapshot.pt',
            'deploy_snapshot.pt',
        ]
        for fname in expected_names:
            fpath = join(train_dpath, fname)
            if exists(fpath):
                break

        if not exists(fpath):
            fpath = None

        if not fpath:
            epoch_to_fpath = {
                parse.parse('{}_epoch_{num:d}.pt', path).named['num']: path
                for path in harn.prev_snapshots()
            }
            if epoch_to_fpath:
                fpath = epoch_to_fpath[max(epoch_to_fpath)]

        # if fpath is None:
        #     raise Exception('cannot find / determine the best snapshot')
        return fpath


@register_mixin
class SnapshotCallbacks(object):
    """
    Snapshot functions that may need to be extended for advanced usage

    Any special training state that you would like netharn to manage must
    returned and handled by the snapshot state setters and getters.
    """

    def get_snapshot_state(harn):
        """
        Returns a dictionary containing the base snapshot state.
        This can be overrided for specific applications.

        Returns:
            dict: snapshot_state
        """
        snapshot_state = {
            'epoch': harn.epoch,
            '_prev_iter_idxs': harn._prev_iter_idxs,
            'model_state_dict': harn.model.state_dict(),
            'optimizer_state_dict': harn.optimizer.state_dict(),
            'monitor_state_dict': harn.monitor.state_dict(),
        }
        if harn.scheduler:
            snapshot_state['scheduler_state_dict'] = harn.scheduler.state_dict()
        return snapshot_state

    def set_snapshot_state(harn, snapshot_state):
        """
        Sets harness state based on a previous snapshot.

        This can be overrided for specific applications.  In this case,
        it is the users responsibility to ensure that this handles all relevant
        items returned by `harn.get_snapshot_state`.

        Args:
            snapshot_state (dict): information corresponding to
        """
        if 'epoch' in snapshot_state:
            # the snapshot holds the previous epoch; add one to move to current
            harn.epoch = snapshot_state['epoch'] + 1

        if '_prev_iter_idxs' in snapshot_state:
            harn._prev_iter_idxs = snapshot_state['_prev_iter_idxs']

        if 'model_state_dict' in snapshot_state:
            harn.model.load_state_dict(snapshot_state['model_state_dict'])
            harn.debug('loaded model_state_dict')

        if 'monitor_state_dict' in snapshot_state:
            # hack: dont override patience, use whatever the current val is
            patience = harn.monitor.patience
            max_epoch = harn.monitor.max_epoch
            harn.monitor.load_state_dict(snapshot_state['monitor_state_dict'])
            harn.monitor.patience = patience
            harn.monitor.max_epoch = max_epoch
            harn.debug('loaded monitor_state_dict')

        if 'optimizer_state_dict' in snapshot_state:
            # NOTE: IF YOU CREATE AN OPTIMIZER WITH A DIFFERENT ORDER OF THE
            # PARAMS INSIDE THE PARAM GROUP THIS WILL FAIL.
            # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/4
            harn.optimizer.load_state_dict(snapshot_state['optimizer_state_dict'])
            harn.optimizer.zero_grad()
            harn.debug('loaded optimizer_state_dict')

        if 'scheduler_state_dict' in snapshot_state:
            harn.scheduler.load_state_dict(snapshot_state['optimizer_state_dict'])
            harn.debug('loaded scheduler_state_dict')

        # Ensure scheduler is given current information
        if harn.scheduler:
            if getattr(harn.scheduler, '__batchaware__', False):
                harn.scheduler.reset_epoch(epoch=harn.epoch)
            else:
                if harn.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    # This is handled via state
                    pass
                else:
                    if getattr(harn.scheduler, '__netharn_redesign__', False):
                        harn.scheduler.step(epoch=harn.epoch)
                    else:
                        harn.scheduler.step(epoch=harn.epoch - 1)


@register_mixin
class ScheduleMixin(object):
    """
    Internal methods for inspecting and modifying the training scheduler.
    """

    def _current_lrs(harn):
        """
        Get the of distinct learning rates (usually only 1) currently in use

        Returns:
            List[float]: list of current learning rates
        """
        # optim_lrs = {group['lr'] for group in harn.optimizer.param_groups}
        optim_lrs = [group['lr'] for group in harn.optimizer.param_groups]
        lrs = optim_lrs
        return lrs

    def _check_termination(harn):
        if harn.epoch >= harn.monitor.max_epoch:
            harn._close_prog()
            harn.info('Maximum harn.epoch reached, terminating ...')
            return True
        done_status = harn.monitor.is_done()
        if done_status:
            harn._close_prog()
            harn.info(done_status)
            # harn.info('Validation set is not improving, terminating ...')
            return True
        return False

    def _step_scheduler_batch(harn):
        """
        Ignore:
            from viame.pytorch import netharn as nh
            harn = nh.FitHarn(nh.HyperParams.demo()).initialize()
        """
        # TODO: proper warmup iters
        if harn.epoch == 0:
            HACK_WARMUP = bool(harn.dynamics['warmup_iters'])
            if HACK_WARMUP:
                from .schedulers.scheduler_redesign import _get_optimizer_values
                from .schedulers.scheduler_redesign import _set_optimizer_values
                # Based on mmdet logic, need to generalize better for netharn
                # So warmup can be used with any scheduler, even a torch one
                cur_iters = harn.batch_index
                warmup = 'linear'
                warmup_iters = harn.dynamics['warmup_iters']
                warmup_ratio = harn.dynamics['warmup_ratio']   # 1.0 / 3.0
                if cur_iters < warmup_iters:
                    # for cur_iters in range(0, warmup_iters):
                    regular_lr = _get_optimizer_values(harn.optimizer, 'initial_lr')
                    if warmup == 'linear':
                        k = (1 - (cur_iters + 1) / warmup_iters) * (1 - warmup_ratio)
                        warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
                    else:
                        raise KeyError(warmup)
                    # harn.debug('warmup_lr = {}'.format(warmup_lr))
                    _set_optimizer_values(harn.optimizer, 'lr', warmup_lr)

        # TODO: REFACTOR SO NETHARN HAS A PROPER ITERATION MODE
        if getattr(harn.scheduler, '__batchaware__', False):
            # TODO: can we determine what the batch size is at this point?
            harn.scheduler.step_batch()

    def _step_scheduler_epoch(harn, improved=None):
        """
        helper function to change the learning rate that handles the way that
        different schedulers might be used.

        Args:
            improved (bool | None): if specified flags if the validation
                metrics have improved (used by ReduceLROnPlateau scheduler)
        """
        epoch_that_just_finished = harn.epoch
        if harn.scheduler is None:
            pass
        elif getattr(harn.scheduler, '__netharn_redesign__', False):
            # New netharn style schedulers step to the epoch you want them to
            # step to. This means we step them to the next epoch. This is
            # different than the standard torch behavior, which uses prev_epoch
            # as the primative.
            harn.scheduler.step(epoch=epoch_that_just_finished + 1)
        elif getattr(harn.scheduler, '__batchaware__', False):
            # For netharn style detectors step_spoch will change epoch instead
            # of last_epoch

            # HACK: Currently dont step on epochs for batchaware schedulers
            # need to figure out how we want to track information when the
            # dataset size / batch size / are not constant.
            # harn.scheduler.step_epoch(epoch=epoch_that_just_finished + 1)
            pass
        elif harn.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            assert improved is not None, 'must validate for ReduceLROnPlateau schedule'

            def hack_lr_step(self, improved, epoch=None):
                if epoch is None:
                    epoch = self.last_epoch = self.last_epoch + 1
                self.last_epoch = epoch

                if improved:
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1

                if self.in_cooldown:
                    self.cooldown_counter -= 1
                    self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

                if self.num_bad_epochs > self.patience:
                    self._reduce_lr(epoch)
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0

                    # TODO: make a pytorch pr where there is a callback on
                    # lr_reduction.
                    # the scheduler has stepped, we should now backtrack the
                    # weights to the previous best state
                    backtrack = False
                    if backtrack:
                        harn.backtrack_weights(harn.monitor.best_epoch)

            # hack to determine if the rlrop scheduler stepped
            hack_lr_step(harn.scheduler, improved)
        else:
            # Note that for torch schedulers the epoch param indicates
            # the epoch that just finished, so calling
            # harn.scheduler.last_epoch will be the same as harn.epoch
            harn.scheduler.step(epoch=epoch_that_just_finished)


@register_mixin
class CoreMixin(object):
    """
    Methods to run and support the core main execution loop
    """
    def run(harn):
        """
        Runs the main training loop

        This starts the main loop which will run until a the monitor's
        terminator criterion is satisfied. If the initialize step loaded a
        checkpointed that already met the termination criterion, then this will
        simply return.

        Notes:
            If harn.preferences['keyboard_debug'] is True, then pressing Ctrl+C
            while this is running will result in an interactive prompt which
            allows some amount of manual control over the training run.

        Raises:
            TrainingDiverged: if training fails due to numerical issues

        Returns:
            PathLike: deploy_fpath: the path to the standalone deployed model
        """
        if not harn._initialized:
            # Executes the initialization of models, loading of weights,
            # creation of logs and directories, etc...
            harn.initialize()

        if ub.argflag('--profile-init'):
            # Debugging flag to help devs make initialization fast.
            return

        if ub.argflag('--pre-deploy'):
            # Create a deploy file before we start/resume training. This is
            # useful when the keyboard_debug is unavailable.
            harn._deploy()

        harn.info('ARGV:\n    ' + sys.executable + ' ' + ' '.join(sys.argv))

        if harn._tlog is not None:
            train_base = dirname(harn.name_dpath or harn.train_dpath)
            harn.info('dont forget to start:\n'
                      '    tensorboard --logdir ' + ub.shrinkuser(train_base))

        try:
            if harn._check_termination():
                raise StopTraining()

            action = 'resume' if harn.epoch > 0 else 'begin'
            if harn.preferences['prog_backend'] == 'progiter':
                text = '=== {} training {!r} / {!r} : {} ==='.format(
                    action, harn.epoch + 1, harn.monitor.max_epoch,
                    harn.hyper.name)
                harn.info(ub.color_text(text, 'white'))
            else:
                harn.info(ub.color_text('=== {} training : {} ==='.format(
                    action, harn.hyper.name), 'white'))

            harn.main_prog = harn._make_prog(
                desc='epoch', total=harn.monitor.max_epoch, disable=not
                harn.preferences['show_prog'], leave=True, dynamic_ncols=True,
                show_wall=True, position=0, initial=harn.epoch)
            harn._update_main_prog_desc()

            # Loader dict should be ordered
            harn.loaders = ub.odict([
                (key, harn.loaders[key]) for key in ['train', 'vali', 'test']
                if key in harn.loaders
            ])

            # keep track of moving metric averages across epochs
            harn._run_metrics = {
                tag: util.WindowedMovingAve(window=len(loader))
                for tag, loader in harn.loaders.items()
                if loader is not None
            }

            harn._check_thread_safety()

            train_loader = harn.loaders.get('train', None)
            vali_loader  = harn.loaders.get('vali', None)
            test_loader  = harn.loaders.get('test', None)

            if not vali_loader:
                harn.debug('Running without a validation dataset')
                if harn.scheduler:
                    if harn.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                        if vali_loader is None:
                            raise ValueError(ub.paragraph(
                                '''
                                A validation dataset is required to use
                                ReduceLROnPlateau, but None was given
                                '''))
                        else:
                            raise ValueError(ub.paragraph(
                                '''
                                A non-empty validation dataset is required to
                                use ReduceLROnPlateau
                                '''))

            #############################
            ### THIS IS THE MAIN LOOP ###
            #############################

            with ub.Timer() as _timer:
                harn._timer = _timer
                for harn.epoch in it.count(harn.epoch):
                    harn._run_tagged_epochs(
                        train_loader,
                        vali_loader,
                        test_loader
                    )
                    if DEMO and harn.epoch > DEMO:
                        raise StopTraining
                    elif _timer.toc() > harn.preferences['timeout']:
                        harn.info('timeout')
                        raise StopTraining

            ##############################
            ### THAT WAS THE MAIN LOOP ###
            ##############################

        except StopTraining:
            pass
        except KeyboardInterrupt:
            if not harn.preferences['keyboard_debug']:
                harn.warn('\n\n\n')
                harn.info('harn.train_dpath = {!r}'.format(harn.train_dpath))

                if harn.preferences['snapshot_after_error']:
                    harn.info('Attempting to checkpoint before crashing')
                    harn.save_snapshot(explicit=True)

                if harn.preferences['deploy_after_error']:
                    harn.info('Attempting to deploy before crashing')
                    harn._deploy()
                raise
            harn.warn('\n\n\n')
            harn.info('harn.train_dpath = {!r}'.format(harn.train_dpath))
            harn.warn('got keyboard interrupt')

            print(ub.codeblock(
                '''
                Training was interrupted. Starting interactive prompt.
                (Pressing Ctrl+C again will exit the program)
                '''))

            while True:
                print(ub.codeblock(
                    '''
                    Please enter one of the following one-character options:

                    [c] - manually checkpoint the EXACT current state of the model.
                    [d] - create a deploy package for the current BEST model.
                    [r] - try and resume training (experimental).
                    [e] - embed into an IPython shell (requires xdev)
                    [q] - quit this prompt and reraise the KeyboardInterrupt
                    '''))
                ans = input('>')
                if ans == 'q':
                    break
                elif ans == 'x':
                    harn._export()
                elif ans == 'd':
                    harn._deploy()
                elif ans == 'c':
                    harn.save_snapshot(explicit=True)
                elif ans == 'r':
                    # This might have issues because the referenes in this
                    # function are still held. Likely the better way to
                    # implement this is by handling the error gracefully and
                    # looping within this function. Might require a
                    # restructure.
                    return harn.run()
                elif ans == 'e':
                    import xdev
                    xdev.embed()
                else:
                    print('Invalid input: {!r}'.format(ans))
            raise
        except Exception as ex:
            harn.error('\n\n\n')
            harn.info('general exception')
            print('harn.preferences = {!r}'.format(harn.preferences))

            if harn.preferences['snapshot_after_error']:
                harn.info('Attempting to checkpoint before crashing')
                harn.save_snapshot(explicit=True)

            if harn.preferences['deploy_after_error']:
                harn.info('Attempting to deploy before crashing')
                harn._deploy()

            harn.info('harn.train_dpath = {!r}'.format(harn.train_dpath))
            harn.error('an {} error occurred in the train loop: {}'.format(
                type(ex), repr(ex)))
            tb = traceback.format_exc()
            harn.info(tb)
            harn._close_prog()
            raise

        #harn.info('\n\n\n')
        #harn.info('training completed')

        if harn._tlog is not None:
            train_base = dirname(harn.name_dpath or harn.train_dpath)
            harn.info('harn.train_dpath = {!r}'.format(harn.train_dpath))
            harn.info('harn.name_dpath  = {!r}'.format(harn.name_dpath))
            harn.info('view tensorboard results for this run via:\n'
                      '    tensorboard --logdir ' + ub.shrinkuser(train_base))

        harn.deploy_fpath = harn._deploy()

        harn.on_complete()
        harn.info('exiting fit harness.')
        return harn.deploy_fpath

    def _export(harn):
        """
        Export the model topology to the train_dpath

        Returns:
            str: path to the exported model topology
        """
        # Check if model explicitly declares it doesn't support deployment
        model_class = harn.hyper.model_cls
        if getattr(model_class, '__DEPLOY_SUPPORTED__', True) is False:
            harn.debug('Model does not support torch_liberator deployment, skipping export')
            return None

        # TODO: might be good to check for multiple model exports at this time
        harn.debug('exporting model topology')
        static_modpath = None
        try:
            model_params = harn.hyper.model_params
            export_modules = harn.preferences['export_modules']
            static_modpath = torch_liberator.export_model_code(
                harn.train_dpath, model_class, initkw=model_params,
                export_modules=export_modules)
            #harn.info('Exported model topology to {}'.format(static_modpath))
        except Exception as ex:
            harn.warn('Failed to export model topology: {}'.format(repr(ex)))
        return static_modpath

    def _deploy(harn):
        """
        Packages the best validation (or most recent) weights with the exported
        model topology into a single-file model deployment that is "mostly"
        independent of the code used to train the model.

        Returns:
            str: path to the deploy zipfile.
        """
        # Check if model explicitly declares it doesn't support deployment
        model_class = harn.hyper.model_cls
        if getattr(model_class, '__DEPLOY_SUPPORTED__', True) is False:
            harn.debug('Model does not support torch_liberator deployment, skipping deploy')
            harn.deploy_fpath = None
            return None

        static_modpath = harn._export()
        harn.debug('packaging deploying model')

        if True:
            snap_fpath = harn.best_snapshot()
            if snap_fpath is None:
                # if the best snapshot doesnt exist we need to make one
                harn.debug(
                    'Cannot find "best" snapshot, write an explit one instead')
                snap_fpath = harn.save_snapshot(explicit=True)

        try:
            train_info_fpath = join(harn.train_dpath, 'train_info.json')
            deploy_fpath = torch_liberator.DeployedModel.custom(
                snap_fpath=snap_fpath,
                model=static_modpath,
                train_info_fpath=train_info_fpath,
            ).package(harn.train_dpath)
            #harn.info('wrote single-file deployment to: {!r}'.format(
            #    deploy_fpath))

            if True:
                # symlink the deployed model to a static filename to make it
                # easier to put netharn in a pipelined system.
                static_deploy_fpath = join(harn.train_dpath, 'deploy.zip')
                try:
                    ub.symlink(deploy_fpath, static_deploy_fpath,
                               overwrite=True, verbose=0)
                except OSError as ex:
                    harn.warn('Unable to symlink: {!r}'.format(ex))

        except Exception as ex:
            deploy_fpath = None
            harn.warn('Failed to deploy: {}'.format(repr(ex)))

        harn.deploy_fpath = deploy_fpath
        return deploy_fpath

    @profiler.profile
    def _run_tagged_epochs(harn, train_loader, vali_loader, test_loader):
        """
        Runs one epoch of train, validation, and testing

        Args:
            train_loader (torch.utils.data.DataLoader | None): train loader
            vali_loader (torch.utils.data.DataLoader | None): vali loader
            test_loader (torch.utils.data.DataLoader | None): test loader
        """
        harn.debug('=== start epoch {} ==='.format(harn.epoch))

        # Log learning rate and momentum
        from .schedulers.scheduler_redesign import _get_optimizer_values
        for attr in ['lr', 'momentum']:
            try:
                lr = max(_get_optimizer_values(harn.optimizer, attr))
                harn.log_value('epoch ' + attr, lr, harn.epoch)
            except KeyError:
                harn.warn('No optimizer attr={}'.format(attr))
                raise

        harn.current_tag = None

        # Reset batch iteration indices to zero
        for tag in harn.bxs.keys():
            harn.bxs[tag] = 0

        harn.before_epochs()

        #########################
        ### Inner Epoch Loops ###
        #########################

        # Clear existing gradients and run training epoch
        if train_loader:
            harn.optimizer.zero_grad()
            harn._run_epoch(train_loader, tag='train', learn=True)

        # Run validation epoch and monitor metrics to guide training
        improved = None
        if vali_loader and harn.check_interval('vali', harn.epoch):
            vali_metrics = harn._run_epoch(vali_loader, tag='vali',
                                           learn=False)
            lr = max(harn._current_lrs())
            improved = harn.monitor.update(harn.epoch, vali_metrics, lr)
            harn._update_main_prog_desc()

        # Run test epoch; never use test results to guide training
        if test_loader and harn.check_interval('test', harn.epoch):
            harn._run_epoch(test_loader, tag='test', learn=False)

        #############################
        ### END Inner Epoch Loops ###
        #############################

        # Increment previous iteration indices, add one because bxs will be one
        # less than the number of batches.
        for tag in harn.bxs.keys():
            harn._prev_iter_idxs[tag] += (harn.bxs[tag] + 1)

        if harn.train_dpath is not None:
            if improved:
                save_fpath = harn.save_snapshot()
                if save_fpath:
                    harn.debug('new best_snapshot {}'.format(save_fpath))
                    # copy the best snapshot the the main directory
                    best_path = join(harn.train_dpath, 'best_snapshot.pt')
                    shutil.copy2(save_fpath, best_path)
            else:
                # todo: allow monitor to clean up old snapshots
                if harn.check_interval('snapshot', harn.epoch):
                    save_fpath = harn.save_snapshot()

            if harn.check_interval('cleanup', harn.epoch):
                harn.cleanup_snapshots()

        terminate_flag = harn._check_termination()

        if harn._tlog is not None and harn.preferences['dump_tensorboard']:
            if not harn.preferences['eager_dump_tensorboard']:
                # If we did not dump iteration metrics in the inner loop then
                # do it here.
                mode = ('epoch', 'iter')
            else:
                mode = ('epoch',)
            try:
                # Perhaps dump tensorboard metrics to png / pickle?
                from .mixins import _dump_monitor_tensorboard
                # If we are about to stop, then force serial mode
                serial = terminate_flag
                _dump_monitor_tensorboard(
                    harn, mode, harn.preferences['tensorboard_groups'],
                    serial=serial)
            except Exception as ex:
                harn.warn('Failed to dump tensorboard: {}'.format(repr(ex)))

        harn.after_epochs()

        # check for termination
        if terminate_flag:
            raise StopTraining()
        else:
            # Step to move to the next epoch
            # change learning rate (modified optimizer inplace)
            harn._step_scheduler_epoch(improved)

            if harn.preferences['prog_backend'] == 'progiter':
                harn.info(ub.color_text(
                    '=== finish epoch {!r} / {!r} : {} ==='.format(
                        harn.epoch + 1, harn.monitor.max_epoch, harn.hyper.name),
                    'white'))

            harn._update_main_prog_desc()
            harn.main_prog.update(1)

    @profiler.profile
    def _run_epoch(harn, loader, tag, learn=False, max_iter=np.inf,
                   call_on_epoch=True):
        """
        Run a single epoch of test / train / or validation

        Notes:
            THE CRITICAL LOOP LIVES HERE

        Args:
            loader (torch.utils.data.DataLoader):
                the loader for your current data split (this will usually be
                ``harn.loaders[tag]``)

            tag (str) : the label for the loader's data split

            learn (bool, default=False): if True, the weights of
                harn.model are updated by harn.optimizer

            max_iter (int, default=inf): limits the number of batches

            call_on_epoch (bool, default=True): if False then `on_epoch`
                is not called after the main loop.

        Returns:
            dict: epoch_metrics - scalar values measured in this epoch.
        """
        harn.debug('_run_epoch {}, tag={}, learn={}'.format(harn.epoch, tag, learn))
        harn.debug(' * len(loader) = {}'.format(len(loader)))

        try:
            bsize = loader.batch_sampler.batch_size
        except AttributeError:
            # Some loaders might have variable batch sizes
            bsize = None

        harn.debug(' * loader.batch_sampler.batch_size = {}'.format(bsize))

        harn.current_tag = tag

        # use exponentially weighted or windowed moving averages across epochs
        iter_moving_metrics = harn._run_metrics[tag]
        # use simple moving average within an epoch
        epoch_moving_metrics = util.CumMovingAve(nan_method='ignore')

        # Flag if model is training (influences batch-norm / dropout)
        # if harn.model.training != learn or learn:
        harn.model.train(learn)

        # call prepare epoch hook
        harn.prepare_epoch()

        msg = harn._batch_msg({'loss': -1}, bsize, learn)
        desc = tag + ' ' + msg
        if harn.main_prog is None:
            position = 1
        else:
            position = (list(harn.loaders.keys()).index(tag) +
                        harn.main_prog.pos + 1)

        n_batches = min(max_iter, len(loader))

        prog = harn._make_prog(desc=desc, total=n_batches,
                               disable=not harn.preferences['show_prog'],
                               position=position,
                               chunksize=bsize, leave=True, dynamic_ncols=True)
        harn.epoch_prog = prog
        harn._update_prog_postfix(prog)

        # Prepopulate local variables to make this critical loop faster.
        ignore_inf_loss_parts = harn.preferences['ignore_inf_loss_parts']
        display_interval = harn.intervals['display_' + tag]
        is_profiling = profiler.IS_PROFILING
        use_tqdm = harn.preferences['prog_backend'] == 'tqdm'
        timeout = harn.preferences['timeout']
        _timer = harn._timer

        if harn.preferences['log_resources']:
            harn.debug(ub.repr2(util.resource_usage(), nl=1))

        if isinstance(prog, ub.ProgIter):
            prog.begin()
        with torch.set_grad_enabled(learn):
            harn.debug('Making batch iterator')

            n_trys_remain = 3
            while n_trys_remain > 0:
                try:
                    batch_iter = iter(loader)
                except OSError as ex:
                    if 'Cannot allocate memory' in str(ex):
                        harn.warn('Cannot allocate memory for the data loader')
                    if n_trys_remain <= 0:
                        harn.error('Cannot allocate enough memory')
                        raise
                else:
                    break
                n_trys_remain -= 0

            harn.debug('Starting batch iteration for tag={}, epoch={}'.format(
                tag, harn.epoch))

            #################################
            ### THIS IS THE CRITICAL LOOP ###
            #################################

            STEP_LR_BEFORE = True

            for bx in range(n_batches):
                if DEMO and bx > DEMO_BX:
                    break
                if _timer is not None and _timer.toc() > timeout:
                    harn.info('timeout')
                    raise StopTraining

                try:
                    raw_batch = next(batch_iter)

                    harn.bxs[tag] = bx
                    # harn.debug('{} batch iteration {}'.format(tag, bx))

                    if STEP_LR_BEFORE:
                        if learn:
                            # Some schedulers update every batch
                            # TODO: needs further rectification
                            harn._step_scheduler_batch()

                    batch = harn.prepare_batch(raw_batch)

                    if is_profiling:
                        torch.cuda.synchronize()

                    # Run the forward pass to compute outputs and loss
                    loss_parts = None
                    outputs, loss = harn.run_batch(batch)

                    if is_profiling:
                        torch.cuda.synchronize()

                    if isinstance(loss, dict):
                        # if loss is a dictionary sum it up to achieve the
                        # total loss and then log each part in the metrics.
                        loss_parts = loss
                        if ignore_inf_loss_parts:
                            # check loss parts and ignore infinite parts
                            loss_parts_ = {}
                            for k, v in list(loss_parts.items()):
                                if np.isfinite(float(v)):
                                    loss_parts_[k] = v
                                else:
                                    harn.warn(
                                        'Ignoring infinite loss component. '
                                        'Setting to large value')

                            if not loss_parts_:
                                raise SkipBatch(
                                    'all loss components were infinite')

                            loss = sum(loss_parts_.values())
                        else:
                            loss = sum(loss_parts.values())

                    if learn:
                        harn.backpropogate(bx, batch, loss)

                    if is_profiling:
                        torch.cuda.synchronize()

                    # measure train accuracy and other informative metrics
                    cur_metrics = harn._on_batch(bx, batch, outputs, loss,
                                                 loss_parts)

                    # accumulate measures
                    epoch_moving_metrics.update(cur_metrics)
                    iter_moving_metrics.update(cur_metrics)

                    # display_train training info
                    if harn.check_interval('display_' + tag, bx) or bx == n_batches - 1:
                        ave_metrics = iter_moving_metrics.average()

                        msg = harn._batch_msg({'loss': ave_metrics['loss']},
                                              bsize, learn)
                        if not harn.preferences['colored']:
                            desc = strip_ansi(desc)
                        prog.set_description(tag + ' ' + msg, refresh=False)

                        # log_iter_train, log_iter_test, log_iter_vali
                        if harn.check_interval('log_iter_' + tag, bx, first=True):
                            # iter_idx = (harn.epoch * n_batches + bx)
                            iter_idx = harn.iter_index
                            for key, value in ave_metrics.items():
                                harn.log_value(tag + ' iter ' + key, value, iter_idx)

                            if harn.preferences['log_resources']:
                                usage = util.resource_usage()
                                key = 'ram'
                                if 'ram_percent' in usage:
                                    value = usage['ram_percent']
                                    harn.log_value(tag + ' iter ' + key, value, iter_idx)
                                harn.debug(ub.repr2(usage, nl=1))

                            if harn._tlog is not None:
                                if (harn.preferences['dump_tensorboard'] and harn.preferences['eager_dump_tensorboard']):
                                    # Dump tensorboard metrics to png / pickle.
                                    from .mixins import _dump_monitor_tensorboard
                                    _dump_monitor_tensorboard(
                                        harn, 'iter',
                                        special_groupers=harn.preferences['tensorboard_groups'])

                        if use_tqdm:
                            prog.update(display_interval)
                        else:
                            # hack to force progiter to reach 100% at the end
                            # This should be fixed in progiter.
                            steps_taken = (bx - prog._iter_idx) + 1
                            if bx == 0:
                                prog.step(steps_taken, force=True)
                                # HACK, after ubelt 0.9.3 we can use force=True
                                # prog._iter_idx += steps_taken
                                # prog._update_measurements()
                                # prog._update_estimates()
                                # prog.display_message()
                                harn.debug(prog.format_message().strip())
                            else:
                                prog_updated = prog.update(steps_taken)
                                if prog_updated:
                                    harn.debug(prog.format_message().strip())

                        if use_tqdm:
                            harn._update_prog_postfix(prog)

                    if not STEP_LR_BEFORE:
                        # old way that I think is buggy
                        if learn:
                            # Some schedulers update every batch
                            harn._step_scheduler_batch()
                except SkipBatch:
                    harn.warn('skipping batch')
                    if harn.check_interval('display_' + tag, bx):
                        prog.update(display_interval)
                        harn._update_prog_postfix(prog)

            # Ensure the data loader is shutdown properly
            if hasattr(batch_iter, 'shutdown'):
                batch_iter._shutdown_workers()
            batch_iter = None

        # do a final step when bstep > 1, so the last few batches arent skipped
        # if harn.dynamics['batch_step'] > 1:
        #     if any(param.grad is not None
        #            for name, param in harn.model.named_parameters()):
        #         harn.optimizer.step()
        #         harn.optimizer.zero_grad()

        if harn.preferences['log_resources']:
            usage = util.resource_usage()
            key = 'ram'
            if 'ram_percent' in usage:
                value = usage['ram_percent']
                harn.log_value(tag + ' epoch ' + key, value, harn.epoch)
            harn.debug(ub.repr2(usage, nl=1))

        prog.refresh()
        if not use_tqdm:
            harn.debug(prog.format_message().strip())
        prog.close()
        harn.epoch_prog = None

        # record a True average for the entire batch
        epoch_metrics = epoch_moving_metrics.average()

        # call hooks after every epoch
        if call_on_epoch:
            custom_metrics = harn.on_epoch()
            _disjoint_dict_update(epoch_metrics, custom_metrics)

        for key, value in epoch_metrics.items():
            harn.log_value(tag + ' epoch ' + key, value, harn.epoch)
        harn.debug('Finished batch iteration for tag={}, epoch={}'.format(
            tag, harn.epoch))

        return epoch_metrics

    @profiler.profile
    def _on_batch(harn, bx, batch, outputs, loss, loss_parts=None):
        """
        Internal function that prepares to call the
        :func:CoreCallbacks.on_batch callback.

        Args:
            bx (int): the current batch index

            batch (object): the current batch

            outputs (object): the first result of :func:CoreCallbacks.run_batch
                These are the raw network outputs.

            loss (Tensor): the second result of :func:CoreCallbacks.run_batch
                This is the batch loss computed by the criterion.

            loss_parts (Dict[str, Tensor]): components of the loss to be
                individually logged.

        Returns:
            Dict[str, float]: dictionary of logged metrics. This is the
                union of the metrics returned by the user as well as addition
                loss information added in this function.
        """
        loss_value = float(loss.data.cpu().item())
        harn._check_loss(loss_value)

        metrics_dict = ub.odict()
        metrics_dict['loss'] = loss_value

        if loss_parts is not None:
            # If loss is specified in parts, then log each part separately
            for key, value in loss_parts.items():
                if value is not None and torch.is_tensor(value):
                    metrics_dict[key + '_loss'] = float(value.data.cpu().item())

        custom_metrics = harn.on_batch(batch, outputs, loss)
        _disjoint_dict_update(metrics_dict, custom_metrics)
        return metrics_dict


@register_mixin
class ChecksMixin(object):
    """
    Helper functions to check if the optimization process is healthy
    """

    def _check_gradients(harn):
        """
        Checks that the the accumulated gradients are all finite.

        Raises:
            TrainingDiverged: if checks fail

        Example:
            harn = ...
            all_grads = harn._check_gradients()
            ub.map_vals(torch.norm, all_grads)
        """
        all_grads = ub.odict()
        for name, parameter in harn.model.named_parameters():
            if parameter.grad is not None:
                all_grads[name] = parameter.grad.data
        for key, value in all_grads.items():
            if torch.any(~torch.isfinite(value)):
                raise TrainingDiverged(
                    'NON-FINITE GRAD {}.grad = {!r}'.format(key, value))
        return all_grads

    @profiler.profile
    def _check_loss(harn, loss_value):
        """
        Checks that the the loss is not too large

        Raises:
            TrainingDiverged: if checks fail
        """
        if not np.isfinite(loss_value):
            harn.warn('WARNING: got inf loss, setting loss to a large value')
            loss_value = harn.preferences['large_loss'] * 10

        if harn.current_tag == 'train':
            if loss_value > harn.preferences['large_loss']:
                # if the loss is getting large, check if the weights are ok
                harn._check_divergence()

    @profiler.profile
    def _check_divergence(harn):
        """
        Checks that the model weights are all finite

        Raises:
            TrainingDiverged: if checks fail
        """
        # Eventually we may need to remove
        # num_batches_tracked once 0.5.0 lands
        state = harn.model.module.state_dict()
        sums = ub.map_vals(torch.sum, state)
        weight_sum = sum(s.float() for s in sums.values())
        if 'torch' in str(type(weight_sum)):  # torch 0.3 / 0.4 / 1.0 compat
            weight_sum = weight_sum.cpu().numpy()
        try:
            weight_sum = weight_sum.cpu().numpy()
        except AttributeError:
            pass
        if not np.isfinite(weight_sum):
            try:
                flags = [not np.isfinite(s.cpu().numpy()) for s in sums.values()]
            except AttributeError:
                flags = [not np.isfinite(s) for s in sums.values()]
            bad_layers = ub.odict(zip(
                ub.compress(sums.keys(), flags),
                ub.compress(sums.values(), flags)
            ))
            harn.error('NON-FINITE WEIGHTS: {}'.format(ub.repr2(bad_layers, nl=1)))
            raise TrainingDiverged(
                'NON-FINITE WEIGHTS weights.sum() = {!r}'.format(weight_sum))

    def _check_layer_rotation(harn):
        """
        References:
            "Layer rotation: a surprisingly powerful indicator of generalization in deep networks?" -
            https://arxiv.org/pdf/1806.01603.pdf

        TODO:
            - [ ] Requires storing network initialization state in memory.
            - [ ] Per layer rotation - cosine distance
            - [ ] Technique to combine into single number? Average? Rotation of flattened network?
        """

        pass


@register_mixin
class CoreCallbacks(object):
    """
    FitHarn's default callback methods. We encourage you to overwrite these.

    FitHarn allows you to customize the execution of the training loop via its
    callback system. You write a callback simply overloading one of these
    methods. There are callbacks with and without default behavior.

    The ones with default behavior directly influence the learning process.
    While these don't have to be overwritten, they usually should be as
    different tasks require slightly different ways of moving data through the
    training pipeline.

    The ones without default behavior allow the developer to execute custom
    code at special places in the training loop. These are usually used for
    logging custom metrics and outputing visualizations.

    The following note lists the callbacks in roughly the order in which
    they are called. The tree structure denotes loop nesting.

    .. code::

        ├─ after_initialize (no default) - runs after FitHarn is initialized
        │  │
        │  ├─ before_epochs (no default) - runs once before all train/vali/test
        │  │  │    epochs on each iteration
        │  │  │
        │  │  ├─ prepare_epoch (no default) - runs before each train, vali,
        │  │  │  │    and test epoch
        │  │  │  │
        │  │  │  ├─ prepare_batch (has default behavior) - transfer data from
        │  │  │  │    CPU to the XPU
        │  │  │  │
        │  │  │  ├─ run_batch (has default behavior) - execute the forward pass
        │  │  │  │    and compute the loss
        │  │  │  │
        │  │  │  ├─ backpropogate (has default behavior) - accumulate gradients
        │  │  │  │    and take an optimization step
        │  │  │  │
        │  │  │  └─ on_batch (no default) - runs after `run_batch` and
        │  │  │       `backpropogate` on every batch
        │  │  │
        │  │  └─ on_epoch (no default) - runs after each train, vali, and test
        │  │         epoch finishes.  Any custom scalar metrics returned in a
        │  │         dictionary will be recorded by the FitHarn loggers.
        │  │
        │  └─ after_epochs (no default) - runs after the all data splits are
        │         finished with  the current epoch.
        │
        └─ on_complete (no default) - runs after the main loop is complete

    """

    def after_initialize(harn):
        """
        Perform a custom initialization step (not usually needed)
        """
        pass

    def before_epochs(harn):
        """
        custom callback run only once before all (train/vali/test) epochs.
        """
        pass

    def prepare_epoch(harn):
        """
        custom callback that is run before each train, vali, and test epoch.
        """
        pass

    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure

        Overload Encouraged, but not always necessary

        Args:
            raw_batch (object): the raw batch generated by the loader

        Returns:
            object: batch - the prepared batch where relevant inputs have
                been moved onto the appropriate XPU(s).

        Notes:
            In the future the default behavior of this will change to simply
            return the raw batch without moving to the XPU. This may be
            necessary to support distributed training.
        """
        batch = raw_batch

        if harn.preferences['auto_prepare_batch']:
            # Automatically move data
            try:
                if isinstance(raw_batch, (tuple, list)):
                    batch = harn.xpu.move(raw_batch)
                elif isinstance(raw_batch, dict):
                    batch = raw_batch.copy()
                    batch = harn.xpu.move(batch)
                else:
                    print('ERROR: raw_batch = {}'.format(type(raw_batch)))
                    raise TypeError(
                        'could not prepare raw batch {}'.format(type(raw_batch)))

            except Exception:
                harn.warn('Error occurred in default prepare_batch. '
                          'Perhaps you should overload it?')
                raise
            return batch
        else:
            return batch

    def run_batch(harn, batch):
        """
        Basic connection inputs -> model -> outputs -> criterion -> loss

        This is the meat and potatoes of your deep learning algorithm,
        everything else is boilerplate. You define how to pass your inputs into
        your model and then compute your loss here. We provide a default
        implementation that will work for basic tasks as long as the model and
        loss are well defined, but you will typically need to overload this.

        Note:
            You may return loss as a flat dictionary mapping string keys to
            tensors. In this case, the total loss will be the sum of the values
            and each loss component will be automatically logged.

        Args:
            batch (object): the current batch as generated by the data loader.
                Note: use :func:`ExtraMixins.._demo_batch` (i.e.
                ``harn._demo_batch()``) to generate an example batch for
                interactive / testing / other usage.

        Returns:
            Tuple[object, Tensor|Dict]:
                tuple containing:
                    outputs - whatever the output of the model was
                    loss - either a single scalar loss or a dictionary of
                        scalar losses (the harness use the keys as labels to
                        track different losses).
        """
        # Simple forward prop and loss computation
        try:
            if isinstance(batch, dict):
                # The extensible case where your batch is a dictionary with
                # keys "input" and "label", which themselves are usually
                # dictionaries.
                outputs = harn.model(batch['input'])
                loss = harn.criterion(outputs, batch['label'])
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                # The "standard" non-extensible case you see in tutorials where
                # items from the dataset are returned as a input / label tuple
                inputs, labels = batch
                outputs = harn.model(inputs)
                loss = harn.criterion(outputs, labels)
            else:
                raise TypeError('Could not run batch: {}'.format(type(batch)))
        except Exception:
            if harn.criterion:
                harn.error('You must overwrite run_batch if '
                           'criterion is not specified')
            else:
                harn.warn('Error occurred in default run_batch. '
                          'Perhaps you should overload it?')
            raise
        return outputs, loss

    @profiler.profile
    def backpropogate(harn, bx, batch, loss):
        """Custom callback which can overwrite the default backward pass

        Backpropogate accumulates gradients, optionally checks and logs the
        gradients, steps the optimizer, and zeros the gradients.

        Overload is generally not necessary for this function.

        TODO:
            - [ ] perhaps remove dynamics as a netharn core component and
            simply allow the end-application to take care of that detail.

        Args:
            bx (int): the current batch index
            batch (object): the current batch
            loss (Tensor): the loss computed in `run_batch`.
        """
        loss.backward()

        if profiler.IS_PROFILING:
            torch.cuda.synchronize()

        # approximates a batch size of (bsize * bstep) if step > 1,
        bstep = harn.dynamics['batch_step']
        if (bx + 1) % bstep == 0:

            tag = harn.current_tag
            iter_idx = harn.iter_index

            if harn.dynamics['grad_norm_max']:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    harn.model.parameters(),
                    max_norm=harn.dynamics['grad_norm_max'],
                    norm_type=harn.dynamics['grad_norm_type'],
                )
                if harn.preferences['log_gradients']:
                    if harn.check_interval('log_iter_' + tag, iter_idx, first=True):
                        harn.log_value(tag + ' iter clipped total norm', total_norm, iter_idx)

                if total_norm > harn.dynamics['grad_norm_max'] * 100:
                    harn.warn('grad norm is too high: '
                              'total_norm = {!r}'.format(total_norm))
            elif harn.preferences['log_gradients']:
                if harn.check_interval('log_iter_' + tag, iter_idx, first=True):
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        harn.model.parameters(),
                        max_norm=float('inf'),
                        norm_type=harn.dynamics['grad_norm_type'],
                    )
                    harn.log_value(tag + ' iter total norm', total_norm, iter_idx)

            if harn.preferences['log_gradients']:
                all_grads = harn._check_gradients()

                if True:
                    layer_mag = {k: v.norm().data.cpu().numpy().tolist() for k, v in all_grads.items()}
                    mag_arr = np.array(list(layer_mag.values()))
                    harn.log_histogram(tag + ' iter layer norm', mag_arr, iter_idx)

            # harn.debug("STEP")
            harn.optimizer.step()
            harn.optimizer.zero_grad()

        if profiler.IS_PROFILING:
            torch.cuda.synchronize()

    def on_batch(harn, batch, outputs, loss):
        """custom callback typically used to compute batch evaluation measures
        or accumulate data.

        If a dict is returned its items are added to batch measures, and
        accumulated via moving averages into epoch measures.

        Overload Encouraged

        Args:
            batch (object): the current batch
            outputs (object): the first result of :func:CoreCallbacks.run_batch
                These are the raw network outputs.
            loss (object): the second result of :func:CoreCallbacks.run_batch
                This is the batch loss computed by the criterion.

        Returns:
            dict or None: dictionary of scalar batch measures
        """
        pass

    def on_epoch(harn):
        """custom callback typically used to compute epoch evaluation measures.

        Called after each train / vali / test datasets.

        If a dict is returned its items are added to epoch measures

        Overload Encouraged

        Returns:
            dict or None: dictionary of scalar epoch measures
        """
        pass

    def after_epochs(harn):
        """
        custom callback run only once after all (train/vali/test) epochs.
        """
        pass

    def on_complete(harn):
        """
        custom callback typically used to evaluate or deploy the final model.

        Overload Encouraged
        """
        pass


@register_mixin
class PropertyMixin(object):
    """
    Access commonly needed harness internals in a convenient way.
    """

    @property
    def raw_model(harn):
        """ returns `harn.model`, but unwraps it it is a `MountedModel` """
        return harn.xpu.raw(harn.model)

    @property
    def batch_index(harn):
        """ The index of the current batch in the current epoch """
        return harn.bxs.get(harn.current_tag, 0)

    @property
    def iter_index(harn):
        """ Returns the current iteration index of the current tag """
        iter_idx = (
            harn._prev_iter_idxs.get(harn.current_tag, 0) +
            harn.bxs.get(harn.current_tag, 0)
        )
        return iter_idx

    @property
    def stage(harn):
        """
        Returns the "tag" of the current dataset (e.g. training, validation,
        test, calibration)
        """
        return harn.current_tag


# Define the exposed class as a union of mixin classes
class FitHarn(ExtraMixins, InitializeMixin, ProgMixin, LogMixin, SnapshotMixin,
              SnapshotCallbacks, ScheduleMixin, CoreMixin, ChecksMixin,
              CoreCallbacks, PropertyMixin):
    """
    Basic harness for training a pytorch model.

    Note:
        The following methods can be overriden to customize the harness

            * prepare_batch(harn, raw_batch) - gets result of `next(loader)`
                and then moves the data onto the GPU / does dynamic preproc.

            * run_batch(harn, batch) - runs the `forward` method of your model,
                computes the loss, and returns both.

            * on_batch(harn, batch, outputs, loss) - does nothing by default.
                This is where you should log statistics about a batch.

            * on_epoch(harn) - does nothing by default. This is where you
                should compute quality metrics about the problem your
                addressing.

        Also see:

            * after_initialize(harn) - initialize your own custom harn
                attribute variables. Runs before `harn.run()`

            * before_epochs(harn) - before train/vali/test epochs are run

            * prepare_epoch(harn) - runs before each train/vali/test epoch

            * after_epochs(harn) - runs after train/vali/test epochs are done

            * backpropogate(harn, bx, batch, loss) - calls `loss.backward()`,
                steps the optimizer, and zeros the gradients. (also handles
                the "dynamics", but that might get deprecated, so ignore it.)

    Args:
        hyper (netharn.HyperParams | dict):
            Parameters that determine the system.  This serializable class
            encodes enough information to deterministically reproduce an
            experiment.

            Because it is serializable it also has a dict representation. If
            hyper is a dict then that dict is used as keyword arguments to
            construct an instance of `netharn.HyperParams`.

        train_dpath (str or None): if specified, all progress information is
            stored in this path and the path computed via hyper is ignored.
            Note: it is recommended that this is left None, and you allow
            `hyper` to create a directory based on the hyperparamters.

    Attributes:
        hyper (netharn.Hyperparams):
            The rectified `hyper` argument that was passed to the FitHarn
            constructor.  Note that hyper is the only (important) argument to
            FitHarn and all other attributes will be derived from `hyper`.
            SeeAlso: `netharn.hyperparameters`.

        model (torch.nn.Module) :
            An instance of your model architecture.
            SeeAlso: `netharn.models` and `netharn.layers` for models and
            layers that may not be in torchvision.

        initializer (netharn.Initializer):
            An initialization strategy (usually either KaimingNormal if
            starting from scratch or Pretrained if doing transfer learning)

        optimizer (torch.optim.optimizer.Optimizer) :
            Optimization algorithm like SGD or ADAM. SeeAlso:
                `netharn.optimizers`

        scheduler (torch.optim.lr_scheduler._LRScheduler) :
            Learning rate scheduler. SeeAlso: `netharn.schedulers` for a
            schedulers that are not currently implemented in torch. Note that
            the newstyle-netharn schedulers can control momentum as well as lr.

        criterion (torch.nn.modules.loss._Loss | None) :
            Objective function / loss criterion. SeeAlso: `netharn.criterions`.
            This is not strictly necessary if the loss is defined inline.

        monitor (netharn.Monitor) :
            monitors performance of the validation set. SeeAlso
            `netharn.monitor`.

    Note:
        hyper is optional. If you choose not to specify it then you must
        overwrite harn._setup_modules and create the requires class instances
        (i.e. model, optimizer, monitor, etc...). You also need to specify
        train_dpath yourself if you want to save progress snapshots.
    """
    def __init__(harn, hyper=None, train_dpath=None):
        if isinstance(hyper, dict):
            hyper = hyperparams.HyperParams(**hyper)

        harn.hyper = hyper

        if DEMO:
            # Hack to prefix the nice name in DEMO mode
            if harn.hyper.name is not None:
                harn.hyper.name = 'DEMO_' + harn.hyper.name
            else:
                raise AssertionError('should have a nice "name" in demo mode')

        harn.datasets = None
        harn.loaders = None

        # The following attributes will be initialized in harn._setup_modules()
        harn.model = None
        harn.initializer = None
        harn.optimizer = None
        harn.scheduler = None
        harn.criterion = None
        harn.monitor = None

        # Note: these default values are actually stored in hyperparams
        # TODO: find a better/general way of handling training dynamics
        # maybe a hook system like mmcv?
        harn.dynamics = {
            'batch_step': 1,        # simulates larger batch sizes
            'grad_norm_max': None,  # clips gradients to a max value (mmdet likes to use 35 for this)
            'grad_norm_type': 2,    # p-norm to use if clipping grads.

            'warmup_iters': 0,          # CURRENTLY HACKED AND EXPERIMENTAL
            'warmup_ratio': 1.0 / 3.0,  # CURRENTLY HACKED AND EXPERIMENTAL
        }

        # Output directories
        harn.train_dpath = train_dpath
        harn.name_dpath = None
        harn.train_info = None

        # Progress bars
        harn.main_prog = None
        harn.epoch_prog = None

        # Public internal state
        harn.epoch = 0  # Track current epoch number

        # Track current iteration within an epoch
        harn.bxs = {
            'train': 0,  # training dataset
            'vali': 0,   # validation dataset
            'test': 0,   # test dataset
            'cali': 0,   # TODO: calibration dataset
        }

        # Tracks the total number of iterations from all previous epochs
        # New in 0.5.3
        # (note, this does not include the current epoch)
        # Pehaps we should keep track of history (i.e. how many iterations were
        # in each batch)?
        harn._prev_iter_idxs = {
            'train': 0,  # training dataset
            'vali': 0,   # validation dataset
            'test': 0,   # test dataset
            'cali': 0,   # TODO: calibration dataset
        }

        harn.intervals = {
            'display_train': 1,
            'display_vali': 1,
            'display_test': 1,

            'log_iter_train': 100,
            'log_iter_test': None,
            'log_iter_vali': None,

            'vali': 1,
            'test': 10,
            # 'cali': 1,

            # how often to take a snapshot
            'snapshot': 1,

            # how often to remove old snapshots
            'cleanup': 10,
        }

        # This is only used as a dictionary.
        harn.preferences = FitHarnPreferences(cmdline=False)

        # This variable should be used to store your custom script
        # configuration
        harn.script_config = {}

        harn.current_tag = None

        # Private internal state
        harn._initialized = False
        harn._log = None
        harn._tlog = None

        harn._timer = None

    @property
    def config(harn):
        import warnings
        warnings.warn('harn.preferences is deprecated, use harn.preferences instead',
                      DeprecationWarning)
        return harn.preferences

    @property
    def nice_dpath(harn):
        import warnings
        warnings.warn('harn.nice_dpath is deprecated, use harn.name_dpath instead',
                      DeprecationWarning)
        return harn.name_dpath

    def check_interval(harn, tag, idx, first=False):
        """
        check if its time to do something that happens every few iterations

        Args:
            tag (str): the tag of the interval to test
            idx (int): the iteration number
            first (bool, default=False): if True, trigger on the first
                iteration, otherwise dont.

        Returns:
            bool: if it is time to do something or not
        """
        n = harn.intervals[tag]
        if n is None or n == 0:
            return False
        elif isinstance(n, int):
            # Intervals can be numbers corresponding to strides
            if first and idx == 0:
                return True
            return (idx + 1) % n == 0
        elif isinstance(n, slice):
            # Intervals can be slices
            if n.stop is not None and idx >= n.stop:
                return False
            start = 0 if n.start is None else n.start
            if idx < start:
                return False
            step = 1 if n.step is None else n.step
            return (idx + start + 1) % step == 0


class FitHarnPreferences(scfg.Config):
    """
    Using scriptconfig to declare defaults for netharn's preferences and
    options. This makes it easy to extend via the commandline.

    Example:
        >>> from .fit_harn import *  # NOQA
        >>> config = FitHarnPreferences()
        >>> config.argparse().print_help()
    """
    # TODO: it might be interesting for preferences to have two defaults, a
    # minimal default and a recommended default. The safe default is
    # statically defined to the minimum requirements, and recommended could
    # be manually or hueristically constructed.
    __default__ = {
        'keyboard_debug': scfg.Value(True, help=(
            'Catch keyboard interupt with a somewhat-interactive prompt')
        ),

        'snapshot_after_error': scfg.Value(True, help=(
            'Try to checkpoint before crashing')
        ),

        'deploy_after_error': scfg.Value(True, help=(
            'Try to deploy before crashing')
        ),

        'show_prog': scfg.Value(True, help=(
            'displays progress')
        ),
        'prog_backend': scfg.Value(
            'progiter', choices=['progiter', 'tqdm', 'auto'], help=(
                'which progress library to use')
        ),

        'ignore_inf_loss_parts': scfg.Value(False, help=(
            'If your loss criterion returns a dictionary of parts, '
            'ignore any infinite values before summing the total loss.')
        ),

        'log_gradients': scfg.Value(False, help=(
            'compute and log stats about gradients')
        ),

        'use_tensorboard': scfg.Value(True, help=(
            'enable logging to tensorboard if available')
        ),

        'eager_dump_tensorboard': scfg.Value(True, help=(
            'If True, logs tensorboard within inner iteration '
            '(experimental). No effect if dump_tensorboard is True')
        ),

        'dump_tensorboard': scfg.Value(True, help=(
            'If True, tensorboard information is visualized with '
            'matplotlib and dumped as an image'),
        ),

        'tensorboard_groups': scfg.Value(['loss'], help=(
            'patterns to be grouped in tensorboard. '
            'Each metric key is split into parts by "_". '
            'For every token X in this list, we group all metrics where '
            'token X is in their parts')
        ),

        'export_modules': scfg.Value([], help=(
            'Set this to a list of modules that the final standalone deployed'
            ' zipfile should not depend on. The exporter will expand any code'
            ' from these modules that are referenced by the model class.')
        ),

        'export_on_init': scfg.Value(True, help=(
            'Export the model topology by default'
            ' when you initialize a harness')
        ),

        'large_loss': scfg.Value(1000, help=(
            'A loss that would be considered large '
            '(This tells netharn when to check for divergence)')
        ),

        'num_keep': scfg.Value(2, help=(
            'number of recent / best snapshots to keep')
        ),
        'keep_freq': scfg.Value(20, help=(
            'Ensure we always keep a snapshot every `freq` epochs')
        ),

        'timeout': scfg.Value(float('inf'), help=(
            'limits the amount of time training can take')
        ),

        'auto_prepare_batch': scfg.Value(False, help=(
            'In the case where prepare_batch is not overwritten, '
            'changes the behavior of the default prepare_batch '
            'to automatically move tensors onto the model XPU'
        )),

        'verbose': scfg.Value(1, help=(
            'verbosity level, '
            'if >1 shows debug info in stdout')),

        'log_resources': scfg.Value(True, help=(
            'Track system resource usage like RAM and disk space')
        ),

        # Deprecated
        'use_tqdm': scfg.Value(None, help='deprecated'),

        'colored': scfg.Value(True, help=(
            'allow for ANSI colored text in stdout logs, '
            'otherwise it is stripped. '
            'DEPRECATED use NO_COLOR environ instead')),

        'allow_unicode': scfg.Value(True, help=(
            'allow for unicode characters in messages, otherwise '
            ' we approximate them with ascii')),
    }


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.fit_harn all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
