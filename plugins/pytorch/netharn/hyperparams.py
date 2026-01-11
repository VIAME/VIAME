# -*- coding: utf-8 -*-
"""
Torch version of hyperparams

TODO:
    [ ] - need to extract relavent params from loaders
    [ ] - need to extract relavent params from datasets
    [ ] - ensure monitor is handled gracefully
    [ ] - prevent non-relevant params from being used in the hash

CommandLine:
    python ~/code/netharn/netharn/hyperparams.py __doc__

Example:
    >>> from viame.pytorch import netharn as nh
    >>> datasets = {
    >>>     'train': nh.data.ToyData2d(size=3, border=1, n=256, rng=0),
    >>>     'vali': nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
    >>> }
    >>> hyper = nh.HyperParams(**{
    >>>     # --- Data First
    >>>     'datasets'    : datasets,
    >>>     'name'        : 'demo',
    >>>     'loaders'     : {'batch_size': 64},
    >>>     'xpu'         : nh.XPU.coerce('auto'),
    >>>     # --- Algorithm Second
    >>>     'model'       : (nh.models.ToyNet2d, {}),
    >>>     'optimizer'   : (nh.optimizers.SGD, {
    >>>         'lr': 0.0001
    >>>     }),
    >>>     'criterion'   : (nh.criterions.CrossEntropyLoss, {}),
    >>>     #'criterion'   : (nh.criterions.FocalLoss, {}),
    >>>     'initializer' : (nh.initializers.KaimingNormal, {
    >>>         'param': 0,
    >>>     }),
    >>>     'scheduler'   : (nh.schedulers.ListedLR, {
    >>>         'points': {0: .0001, 2: .01, 5: .015, 6: .005, 9: .001},
    >>>     }),
    >>>     'monitor'     : (nh.Monitor, {
    >>>         'max_epoch': 10
    >>>     }),
    >>> })
    >>> print(ub.repr2(hyper.get_initkw()))
    >>> print(ub.repr2(hyper.hyper_id()))


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import platform
import warnings
from os.path import join
from os.path import normpath
import sys
import numpy as np
import ubelt as ub
import torch
import six
from viame.pytorch.netharn import util
from viame.pytorch.netharn import initializers
from viame.pytorch.netharn import device
from collections import OrderedDict
# from netharn import criterions
from torch.optim.optimizer import required
import torch.utils.data as torch_data
from viame.pytorch.netharn.util import util_json
from viame.pytorch.netharn.util import util_inspect


# backwards compatibility
_ensure_json_serializable = util_json.ensure_json_serializable  # NOQA


try:
    import imgaug
    Augmenter = imgaug.augmenters.meta.Augmenter
except ImportError:
    imgaug = None


def _hash_data(data):
    return ub.hash_data(data, hasher='sha512', base='abc', types=True)


def _rectify_class(arg, kw, lookup=None):
    """
    Helps normalize and serialize hyperparameter inputs.

    Args:
        arg (Tuple[type, dict] | type | object):
            Either a (cls, initkw) tuple, a class, or an instance.
            It is recommended that you don't pass an instance.

        kw (Dict[str, object]):
            augments initkw if arg is in tuple form otherwise becomes initkw

        lookup (func | None):
            transforms arg or arg[0] into the class type

    Returns:
        Dict: containing
            'cls' (type): the type of the object
            'cls_kw' (Dict): the initialization keyword args
            'instance': (object): None or the actual instanciated object

            We will use this cls and cls_kw to construct an instance unless one
            is already specified.

    Example:
        >>> # The ideal case is that we have a cls, initkw tuple
        >>> from viame.pytorch import netharn as nh
        >>> kw = {'lr': 0.1}
        >>> cls = torch.optim.SGD
        >>> rectified1 = _rectify_class(cls, kw.copy())
        >>> print('rectified1 = {!r}'.format(rectified1))
        >>> # But we can also take an instance of the object, however, you must
        >>> # now make sure to specify the _initkw attribute.
        >>> model = nh.models.ToyNet2d()
        >>> self = cls(model.parameters(), **kw)
        >>> self._initkw = kw
        >>> rectified2 = _rectify_class(self, {})
        >>> print('rectified2 = {!r}'.format(rectified2))
    """
    if lookup is None:
        lookup = ub.identity

    if arg is None:
        cls = None
        cls_kw = {}
        instance = None
    else:
        instance = None

        # Extract the part that identifies the class we care about
        if isinstance(arg, tuple):
            cls_key = arg[0]
            kw2 = arg[1]
        else:
            cls_key = arg
            kw2 = {}

        cls = lookup(cls_key)

        if not isinstance(cls, type):
            # Rectified an instance to an instance
            cls = cls.__class__

        instance = None
        if isinstance(cls_key, cls):
            # We were passed an actual instance of the class. (for shame)
            instance = cls_key

        cls_kw = util_inspect.default_kwargs(cls).copy()

        if instance is not None:
            # Try and introspect the initkw, which is needed for model
            # deployment and proper hyperparam tracking
            for key in cls_kw:
                if hasattr(instance, key):
                    cls_kw[key] = getattr(instance, key)
            if hasattr(instance, '_initkw'):
                # Special attribute that allows safe instance specification and
                # supresses the instance warning.
                cls_kw.update(instance._initkw)
            else:
                import warnings
                warnings.warn(ub.paragraph(  # _initkw warning
                    '''
                    netharn.HyperParams objects are expected to be specified as
                    (type, kw) tuples, but we received a preconstructed
                    instance. This is only ok if you know what you are doing.
                    To disable this warning set the _initkw instance attribute
                    to the correct keyword arguments needed to reconstruct this
                    class. Offending data is arg={!r}, kw={!r}
                    ''').format(arg, kw))

        # Update with explicitly specified information
        cls_kw.update(kw2)
        for key in cls_kw:
            if key in kw:
                cls_kw[key] = kw.pop(key)

    cls_kw = util_json.ensure_json_serializable(cls_kw)
    rectified = {
        'cls': cls,
        'cls_kw': cls_kw,
        'instance': instance,
    }
    return rectified


def _rectify_criterion(arg, kw):
    if arg is None:
        # arg = 'CrossEntropyLoss'
        return _rectify_class(None, kw)

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                # criterions.ContrastiveLoss,
                torch.nn.CrossEntropyLoss,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    rectified = _rectify_class(arg, kw, _lookup)
    return rectified


def _rectify_optimizer(arg, kw):
    """
    Create a rectified tuple

    Example:
        >>> # Test using a (cls, kw) tuple and an instance object.
        >>> from viame.pytorch import netharn as nh
        >>> optim_ = nh.api.Optimizer.coerce({
        >>>     'optim': 'adam', 'lr': 0.1, 'weight_decay': 1e-4})
        >>> cls, kw = optim_
        >>> assert kw.pop('params') is None
        >>> #
        >>> model = nh.models.ToyNet2d()
        >>> params = dict(model.named_parameters())
        >>> grouped_keys = {}
        >>> grouped_keys['bias'] = [k for k in params.keys() if 'bias' in k]
        >>> grouped_keys['weight'] = [k for k in params.keys() if 'weight' in k]
        >>> named_param_groups = {
        >>>     k: {'params': list(ub.take(params, sorted(v)))}
        >>>     for k, v in grouped_keys.items()
        >>> }
        >>> named_param_groups['bias']['weight_decay'] = 0
        >>> param_groups = list(ub.sorted_keys(named_param_groups).values())
        >>> #
        >>> optim = cls(param_groups, **kw)
        >>> rectified1 = _rectify_optimizer(cls, kw)
        >>> rectified2 = _rectify_optimizer(optim, {})
    """
    if arg is None:
        arg = 'SGD'
        if kw is None:
            kw = {}
        kw = kw.copy()
        if 'lr' not in kw:
            kw['lr'] = .001

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                torch.optim.Adam,
                torch.optim.SGD,
            ]
            cls = {c.__name__.lower(): c for c in options}[arg.lower()]
        else:
            cls = arg
        return cls

    rectified = _rectify_class(arg, kw, _lookup)
    kw2 = rectified['cls_kw']

    for k, v in kw2.items():
        if v is required:
            raise ValueError('Must specify {} for {}'.format(k, rectified['cls']))

    return rectified


def _rectify_lr_scheduler(arg, kw):
    if arg is None:
        return _rectify_class(None, kw)

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                torch.optim.lr_scheduler.LambdaLR,
                torch.optim.lr_scheduler.StepLR,
                torch.optim.lr_scheduler.MultiStepLR,
                torch.optim.lr_scheduler.ExponentialLR,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    rectified = _rectify_class(arg, kw, _lookup)
    return rectified


def _rectify_initializer(arg, kw):
    if arg is None:
        arg = 'NoOp'
        # arg = 'CrossEntropyLoss'
        # return None, None

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                initializers.KaimingNormal,
                initializers.NoOp,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    rectified = _rectify_class(arg, kw, _lookup)
    return rectified


def _rectify_monitor(arg, kw):
    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = []
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls
    rectified = _rectify_class(arg, kw, _lookup)
    return rectified


def _rectify_dynamics(arg, kw):
    """
    Special params that control the dynamics of learning at the harness level
    at point that doesnt correspond to a decoupled class component.
    """
    if arg is None:
        arg = {}
    arg = arg.copy()
    dynamics = {
        # batch_step simulates larger batch sizes
        'batch_step': arg.pop('batch_step', 1),
        # Clips gradients
        'grad_norm_max': arg.pop('grad_norm_max', None),
        'grad_norm_type': arg.pop('grad_norm_type', 2),

        'warmup_iters': arg.pop('warmup_iters', None),  # HACKED AND EXPERIMENTAL
        'warmup_ratio': arg.pop('warmup_ratio', 1.0 / 3.0),  # HACKED AND EXPERIMENTAL
    }
    if not isinstance(dynamics['batch_step'], int):
        raise ValueError('batch_step must be an integer')
    if arg:
        raise KeyError('UNKNOWN dynamics: {}'.format(arg))
    return dynamics


def _rectify_model(arg, kw):
    if arg is None:
        return _rectify_class(None, kw)

    def _lookup_model(arg):
        import torchvision
        if isinstance(arg, six.string_types):
            options = [
                torchvision.models.AlexNet,
                torchvision.models.DenseNet,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    # Unwrap the model if was mounted
    if isinstance(arg, device.MountedModel):
        arg = arg.module

    rectified = _rectify_class(arg, kw, _lookup_model)
    return rectified


def _rectify_loaders(arg, kw):
    """
    Loaders are handled slightly differently than other classes
    We construct them eagerly (if they are not already constructed)

    Example:
        >>> # test that dict-base spec words
        >>> _rectify_loaders({'batch_size': 4}, {})
    """
    if arg is None:
        arg = {}

    loaders = None

    if isinstance(arg, dict):
        # Check if all args are already data loaders
        # if isinstance(arg.get('train', None), torch_data.DataLoader):
        if len(arg) and all(isinstance(v, torch_data.DataLoader) for v in arg.values()):
            # loaders were custom specified
            loaders = arg
            # TODO: extract relevant loader params efficiently
            cls = None
            if 'train' in loaders:
                kw2 = {
                    'batch_size': loaders['train'].batch_sampler.batch_size,
                }
            else:
                kw2 = {}
        else:
            # loaders is kwargs for `torch_data.DataLoader`
            arg = (torch_data.DataLoader, arg)
            rectified = _rectify_class(arg, kw)
            cls = rectified['cls']
            kw2 = rectified['cls_kw']
    else:
        raise ValueError('Loaders should be a dict')

    kwnice = ub.dict_subset(kw2, ['batch_size'], default=None)
    return loaders, cls, kw2, kwnice


class HyperParams(object):
    """
    Holds hyperparams relavent to training strategy

    The idea is that you tell it what is relevant FOR YOU, and then it makes
    you nice ids based on that. If you give if enough info it also allows you
    to use the training harness.

    CommandLine:
        python -m netharn.hyperparams HyperParams

    Example:
        >>> from .hyperparams import *
        >>> hyper = HyperParams(
        >>>     criterion=('CrossEntropyLoss', {
        >>>         'weight': torch.FloatTensor([0, 2, 1]),
        >>>     }),
        >>>     optimizer=(torch.optim.SGD, {
        >>>         'nesterov': True, 'weight_decay': .0005,
        >>>         'momentum': 0.9, 'lr': .001,
        >>>     }),
        >>>     scheduler=('ReduceLROnPlateau', {}),
        >>> )
        >>> # xdoctest: +IGNORE_WANT
        >>> print(hyper.hyper_id())
        NoOp,SGD,dampening=0,lr=0.001,momentum=0.9,nesterov=True,weight_decay=0.0005,ReduceLROnPlateau,cooldown=0,eps=1e-08,factor=0.1,min_lr=0,mode=min,patience=10,threshold=0.0001,threshold_mode=rel,verbose=False,CrossEntropyLoss,ignore_index=-100,reduce=None,reduction=mean,size_average=None,weight=[0.0,2.0,1.0],DataLoader,batch_size=1,Dynamics,batch_step=1,grad_norm_max=None
    """

    def __init__(hyper,
                 # ----
                 datasets=None,
                 name=None,
                 workdir=None,
                 xpu=None,
                 loaders=None,
                 # ----
                 model=None,
                 criterion=None,
                 optimizer=None,
                 initializer=None,
                 scheduler=None,
                 # ---
                 dynamics=None,
                 monitor=None,
                 augment=None,
                 other=None,  # incorporated into the hash
                 extra=None,  # ignored when computing the hash
                 nice=None,  # deprecated, alias of name
                 ):
        kwargs = {}

        hyper.datasets = datasets
        if name is None:
            import warnings
            warnings.warn(
                'The "nice" argument is deprecated and will be removed. '
                'Specify "name" instead.', DeprecationWarning)
            name = nice
        if name is None:
            # raise ValueError('you must specify a name for HyperParams')
            name = 'untitled'
        hyper.name = name
        hyper.workdir = workdir
        hyper.xpu = xpu

        loaders, cls, kw, kwnice = _rectify_loaders(loaders, kwargs)
        hyper.loaders = loaders
        hyper.loader_cls = cls
        hyper.loader_params = kw
        hyper.loader_params_nice = kwnice

        hyper._model_info = _rectify_model(model, kwargs)
        hyper.model_cls = hyper._model_info['cls']
        hyper.model_params = hyper._model_info['cls_kw']

        hyper._optimizer_info = _rectify_optimizer(optimizer, kwargs)
        hyper.optimizer_cls = hyper._optimizer_info['cls']
        hyper.optimizer_params = hyper._optimizer_info['cls_kw']

        hyper._scheduler_info = _rectify_lr_scheduler(scheduler, kwargs)
        hyper.scheduler_cls = hyper._scheduler_info['cls']
        hyper.scheduler_params = hyper._scheduler_info['cls_kw']

        hyper._criterion_info = _rectify_criterion(criterion, kwargs)
        hyper.criterion_cls = hyper._criterion_info['cls']
        hyper.criterion_params = hyper._criterion_info['cls_kw']

        hyper._initializer_info = _rectify_initializer(initializer, kwargs)
        hyper.initializer_cls = hyper._initializer_info['cls']
        hyper.initializer_params = hyper._initializer_info['cls_kw']

        hyper._monitor_info = _rectify_monitor(monitor, kwargs)
        hyper.monitor_cls = hyper._monitor_info['cls']
        hyper.monitor_params = hyper._monitor_info['cls_kw']

        hyper.dynamics = _rectify_dynamics(dynamics, kw)

        hyper.augment = augment
        hyper.other = other
        hyper.extra = extra

    @property
    def nice(hyper):
        """ alias of name for backwards compatibility """
        return hyper.name

    def make_model(hyper):
        """ Instantiate the model defined by the hyperparams """
        if hyper._model_info['instance'] is not None:
            return hyper._model_info['instance']
        model = hyper.model_cls(**hyper.model_params)
        return model

    def make_optimizer(hyper, named_parameters):
        r"""
        Instantiate the optimizer defined by the hyperparams

        Contains special logic to create param groups

        Example:
            >>> from viame.pytorch import netharn as nh
            >>> config = {'optimizer': 'sgd', 'params': [
            >>>     {'lr': 3e-3, 'params': '.*\\.bias'},
            >>>     {'lr': 1e-3, 'params': '.*\\.weight'},
            >>>     #{'lr': 100, 'params': '.*\\.doesnotmatch'},
            >>> ]}
            >>> optim_ = nh.api.Optimizer.coerce(config)
            >>> hyper = nh.HyperParams(optimizer=optim_)
            >>> model = nh.models.ToyNet1d()
            >>> named_parameters = list(model.named_parameters())
            >>> optimizer = hyper.make_optimizer(named_parameters)
            >>> print('optimizer = {!r}'.format(optimizer))
        """
        if hyper._optimizer_info['instance'] is not None:
            return hyper._optimizer_info['instance']
        # What happens if we want to group parameters
        optim_kw = hyper.optimizer_params.copy()
        params = optim_kw.pop('params', None)
        if params is None:
            param_groups = [p for (name, p) in named_parameters]
        else:
            import re
            named_parameters = list(named_parameters)
            name_to_param = dict(named_parameters)
            param_groups = []
            if isinstance(params, dict):
                # remember the group key
                groups = [{'key': k, **g} for k, g in params.items()]
            if isinstance(params, list):
                groups = params

            PREVENT_DUPLICATES = 1

            seen_ = set()
            for group in groups:
                # Transform param grouping specifications into real params
                group = group.copy()
                spec = group.pop('params')
                if isinstance(spec, list):
                    if len(spec):
                        first = ub.peek(spec)
                        if isinstance(first, str):
                            real_params = [name_to_param[k] for k in spec]
                        elif isinstance(first, torch.nn.Parameter):
                            real_params = spec
                        else:
                            raise TypeError(type(first))
                    else:
                        real_params = []

                # Python 3.6 doesn't have re.Pattern
                elif isinstance(spec, str) or hasattr(spec, 'match'):
                    if hasattr(spec, 'match'):
                        pat = spec
                    else:
                        pat = re.compile(spec)
                    real_params = [p for name, p in name_to_param.items()
                                   if pat.match(name)]
                else:
                    raise TypeError(type(spec))

                if PREVENT_DUPLICATES:
                    # give priority to earlier params
                    # This is Python 3.6+ only
                    real_params = list(ub.oset(real_params) - seen_)
                    seen_.update(real_params)

                group['params'] = real_params
                param_groups.append(group)

            CHECK = 1
            if CHECK:
                # Determine if we are using the same param more than once
                # or if we are not using a param at all.
                # NOTE: torch does do a duplicate check.
                param_group_ids = []
                for group in param_groups:
                    ids = list(map(id, group['params']))
                    param_group_ids.append(ids)

                all_param_ids = [id(p) for n, p in named_parameters]
                flat_ids = list(ub.flatten(param_group_ids))
                freq = ub.dict_hist(flat_ids, labels=all_param_ids)
                num_unused = any(v == 0 for v in freq.values())
                num_dups = any(v > 1 for v in freq.values())
                if num_unused:
                    warnings.warn('There are {} unused params'.format(num_unused))
                if num_dups:
                    warnings.warn('There are {} duplicate params'.format(num_dups))

        optimizer = hyper.optimizer_cls(param_groups, **optim_kw)
        return optimizer

    def make_scheduler(hyper, optimizer):
        """ Instantiate the lr scheduler defined by the hyperparams """
        if hyper._scheduler_info['instance'] is not None:
            return hyper._scheduler_info['instance']
        if hyper.scheduler_cls is None:
            return None
        kw = hyper.scheduler_params.copy()
        kw['optimizer'] = optimizer
        scheduler = hyper.scheduler_cls(**kw)
        return scheduler

    def make_initializer(hyper):
        """ Instantiate the initializer defined by the hyperparams """
        if hyper._initializer_info['instance'] is not None:
            return hyper._initializer_info['instance']
        initializer = hyper.initializer_cls(**hyper.initializer_params)
        return initializer

    def make_criterion(hyper):
        """ Instantiate the criterion defined by the hyperparams """
        if hyper._criterion_info['instance'] is not None:
            return hyper._criterion_info['instance']
        if hyper.criterion_cls is None:
            return None
        criterion = hyper.criterion_cls(**hyper.criterion_params)
        return criterion

    def make_loaders(hyper):
        if hyper.loaders is not None:
            return hyper.loaders
        else:
            loaders = {
                key: torch_data.DataLoader(dset, **hyper.loader_params)
                for key, dset in hyper.datasets.items()
            }
        return loaders

    def make_xpu(hyper):
        """ Instantiate the criterion defined by the hyperparams """
        xpu = device.XPU.coerce(hyper.xpu)
        return xpu

    def make_monitor(hyper):
        """ Instantiate the monitor defined by the hyperparams """
        if hyper._monitor_info['instance'] is not None:
            return hyper._monitor_info['instance']
        if hyper.monitor_cls is None:
            return None
        monitor = hyper.monitor_cls(**hyper.monitor_params)
        return monitor

    def other_id(hyper):
        """
            >>> from .hyperparams import *
            >>> hyper = HyperParams(other={'augment': True, 'n_classes': 10, 'n_channels': 5})
            >>> hyper.hyper_id()
        """
        otherid = util.make_short_idstr(hyper.other, precision=4)
        return otherid

    def get_initkw(hyper):
        """
        Make list of class / params relevant to reproducing an experiment

        CommandLine:
            python ~/code/netharn/netharn/hyperparams.py HyperParams.get_initkw

        Example:
            >>> from .hyperparams import *
            >>> hyper = HyperParams(
            >>>     criterion='CrossEntropyLoss',
            >>>     optimizer='Adam',
            >>>     loaders={'batch_size': 64},
            >>> )
            >>> print(ub.repr2(hyper.get_initkw()))
        """
        initkw = OrderedDict()
        def _append_part(key, cls, params, initkw):
            """
            append an id-string derived from the class and params.
            TODO: what if we have an instance and not a cls/params tuple?
            """
            if cls is None:
                initkw[key] = None
            else:
                d = OrderedDict()
                for k, v in sorted(params.items()):
                    # if k in total:
                    #     raise KeyError(k)
                    if isinstance(v, torch.Tensor):
                        v = v.numpy()
                    if isinstance(v, np.ndarray):
                        if v.dtype.kind == 'f':
                            try:
                                v = list(map(float, v))
                            except Exception:
                                v = v.tolist()
                        else:
                            raise NotImplementedError()
                    d[k] = v
                    # total[k] = v
                if isinstance(cls, six.string_types):
                    type_str = cls
                else:
                    modname = cls.__module__
                    type_str = modname + '.' + cls.__name__
                # param_str = util.make_idstr(d)
                initkw[key] = (type_str, d)

        _append_part('model', hyper.model_cls, hyper.model_params, initkw)
        _append_part('initializer', hyper.initializer_cls, hyper.initializer_params, initkw)
        _append_part('optimizer', hyper.optimizer_cls, hyper.optimizer_params, initkw)
        _append_part('scheduler', hyper.scheduler_cls, hyper.scheduler_params, initkw)
        _append_part('criterion', hyper.criterion_cls, hyper.criterion_params, initkw)

        # TODO: should other be included in initkw? I think it should.
        # probably should also include monitor, xpu, name

        # Loader is a bit hacked
        _append_part('loader', hyper.loader_cls, hyper.loader_params_nice, initkw)
        _append_part('dynamics', 'Dynamics', hyper.dynamics, initkw)

        return initkw

    def augment_json(hyper):
        """
        Get augmentation info in json format

        Example:
            >>> from .hyperparams import *
            >>> hyper = HyperParams(augment=OrderedDict())
            >>> assert hyper.augment_json() == {}
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> import imgaug
            >>> augment = imgaug.augmenters.Affine()
            >>> hyper = HyperParams(augment=augment)
            >>> info = hyper.augment_json()
            >>> assert info['__class__'] == 'Affine'
        """
        if hyper.augment is None:
            return None
        elif imgaug is not None and isinstance(hyper.augment, imgaug.augmenters.Augmenter):
            from .data.transforms.augmenter_base import ParamatarizedAugmenter
            augment_json = ParamatarizedAugmenter._json_id(hyper.augment)
        elif isinstance(hyper.augment, six.string_types):
            return hyper.augment
        # Some classes in imgaug (e.g. Sequence) inherit from list,
        # so we have to check for Augmenter before we check for list type
        # if isinstance(hyper.augment, (dict, list)):
        elif isinstance(hyper.augment, OrderedDict):
            # dicts are specified in json format
            try:
                # hashable data should be loosely json-compatible
                _hash_data(hyper.augment)
            except TypeError:
                raise TypeError(
                    'NOT IN ORDERED JSON FORMAT hyper.augment={}'.format(
                        hyper.augment))
            augment_json = hyper.augment
        else:
            raise TypeError('Specify augment in json format')
        return augment_json

    def input_id(hyper, short=False, hashed=False):
        pass

    def _parts_id(hyper, parts, short=False, hashed=False):
        id_parts = []
        for key, value in parts.items():
            if value is None:
                continue
            clsname, params = value
            type_str = clsname.split('.')[-1]
            id_parts.append(type_str)

            # Precidence of specifications (from lowest to highest)
            # SF=single flag, EF=explicit flag
            # SF-short, SF-hash, EF-short EF-hash
            request_short = short is True
            request_hash = hashed is True
            if (ub.iterable(short) and key in short):
                request_hash = False
                request_short = True
            if (ub.iterable(hashed) and key in hashed):
                request_hash = True
                request_short = False

            if request_hash:
                param_str = util.make_idstr(params)
                param_str = _hash_data(param_str)[0:6]
            elif request_short:
                param_str = util.make_short_idstr(params)
            else:
                param_str = util.make_idstr(params)

            if param_str:
                id_parts.append(param_str)
        idstr = ','.join(id_parts)
        return idstr

    def hyper_id(hyper, short=False, hashed=False):
        """
        Identification string that uniquely determined by training hyper.
        Suitable for hashing.

        Note:
            setting short=True is deprecated

        CommandLine:
            python -m netharn.hyperparams HyperParams.hyper_id

        Example:
            >>> from .hyperparams import *
            >>> hyper = HyperParams(criterion='CrossEntropyLoss', other={'n_classes': 10, 'n_channels': 5})
            >>> print(hyper.hyper_id())
            >>> print(hyper.hyper_id(hashed=True))
            >>> #print(hyper.hyper_id(short=['optimizer']))
            >>> #print(hyper.hyper_id(short=['optimizer'], hashed=True))
            >>> #print(hyper.hyper_id(short=['optimizer', 'criterion'], hashed=['criterion']))
        """
        parts = hyper.get_initkw()
        return hyper._parts_id(parts, short, hashed)

    def train_info(hyper, train_dpath=None):
        """
        Create json metadata that details enough information such that it would
        be possible for a human to reproduce the experiment.

        Example:
            >>> from viame.pytorch import netharn as nh
            >>> datasets = {
            >>>     'train': nh.data.ToyData2d(size=3, border=1, n=256, rng=0),
            >>>     'vali': nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
            >>> }
            >>> hyper = nh.hyperparams.HyperParams(**{
            >>>     # --- Data First
            >>>     'datasets'    : datasets,
            >>>     'name'        : 'demo',
            >>>     'workdir'     : ub.ensure_app_cache_dir('netharn/demo'),
            >>>     'loaders'     : {'batch_size': 64},
            >>>     'xpu'         : nh.XPU.coerce('auto'),
            >>>     # --- Algorithm Second
            >>>     'model'       : (nh.models.ToyNet2d, {}),
            >>>     'optimizer'   : (nh.optimizers.SGD, {
            >>>         'lr': 0.001
            >>>     }),
            >>>     'criterion'   : (nh.criterions.CrossEntropyLoss, {}),
            >>>     #'criterion'   : (nh.criterions.FocalLoss, {}),
            >>>     'initializer' : (nh.initializers.KaimingNormal, {
            >>>         'param': 0,
            >>>     }),
            >>>     'scheduler'   : (nh.schedulers.ListedLR, {
            >>>         'step_points': {0: .001, 2: .01, 5: .015, 6: .005, 9: .001},
            >>>         'interpolate': True,
            >>>     }),
            >>>     'monitor'     : (nh.Monitor, {
            >>>         'max_epoch': 10
            >>>     }),
            >>> })
            >>> info = hyper.train_info()
            >>> print(ub.repr2(info))
        """
        given_explicit_train_dpath = train_dpath is not None
        # TODO: needs MASSIVE cleanup and organization

        # TODO: if pretrained is another netharn model, then we should read that
        # train_info if it exists and append it to a running list of train_info

        if hyper.model_cls is None:
            # import utool
            # utool.embed()
            raise ValueError('model_cls is None')
        # arch = hyper.model_cls.__name__

        train_dset = hyper.datasets.get('train', None)
        if train_dset is not None and hasattr(train_dset, 'input_id'):
            input_id = train_dset.input_id
            if callable(input_id):
                input_id = input_id()
        else:
            warnings.warn(
                'FitHarn cannot track the training dataset state because '
                'harn.datasets["train"] is missing the "input_id" attribute.'
            )
            input_id = 'none'

        def _hash_data(data):
            return ub.hash_data(data, hasher='sha512', base='abc', types=True)

        train_hyper_id_long = hyper.hyper_id()
        train_hyper_id_brief = hyper.hyper_id(short=False, hashed=True)
        train_hyper_hashid = _hash_data(train_hyper_id_long)[:8]

        # TODO: hash this to some degree
        other_id = hyper.other_id()

        augment_json = hyper.augment_json()

        aug_brief = 'AU' + _hash_data(augment_json)[0:6]
        # extra_hash = _hash_data([hyper.centering])[0:6]

        train_id = '{}_{}_{}_{}'.format(
            _hash_data(input_id)[:6], train_hyper_id_brief,
            aug_brief, other_id)

        # Gather all information about this run into a single hash

        """
        NOTE:
            On choosing the length to truncate the hash.

            If we have an alphabet of size A=26, and we truncate to M=8
            samples, then the number of possible hash values is N = A ** M.
            The probability we will have a collision (assuming an ideal hash
            function where all outputs are equally likely) in r different
            inputs is given by the following function. Note this is the
            birthday paradox problem [1].


            ```python
            from numpy import exp, log
            from scipy.special import loggamma
            def prob_unique(N, r):
                return exp( gammaln(N+1) - gammaln(N-r+1) - r*log(N) )

            A = 26  # size of the alphabet for _hash_data
            M = 8   # number of characters we truncate at
            N = A ** M  # number of possible hash values

            r = 1000

            prob_collision = 1 - prob_unique(N, r)
            print('prob_collision = {!r}'.format(prob_collision))
            ```

            This is approximately 0.00056 or about 1 in 1784.

            Should probably bump the size in a later version.  Note, the above
            code does not seem to be producing the correct number, likely due
            to floating point errors.

            References:
                ..[1] https://www.johndcook.com/blog/2016/01/30/general-birthday-problem/
        """
        train_hashid = _hash_data(train_id)[0:8]

        name = hyper.name

        nice_dpath = None
        name_dpath = None
        if not given_explicit_train_dpath:
            # setup a cannonical and a linked symlink dir
            train_dpath = normpath(
                    join(hyper.workdir, 'fit', 'runs', name, train_hashid))
            # also setup a custom "name", which may conflict. This will
            # overwrite an existing "name" symlink, but the real runs directory
            # is based on a hash, so it wont be overwritten with astronomicaly
            # high probability.
            if name:
                try:
                    name_dpath = normpath(
                            join(hyper.workdir, 'fit', 'name', name))
                    nice_dpath = normpath(
                            join(hyper.workdir, 'fit', 'nice', name))
                except Exception:
                    print('hyper.workdir = {!r}'.format(hyper.workdir))
                    print('hyper.name = {!r}'.format(hyper.name))
                    raise

        # make temporary initializer so we can infer the history
        temp_initializer = hyper.make_initializer()
        init_history = temp_initializer.history()

        # TODO: software versions

        train_info =  ub.odict([
            ('train_hashid', train_hashid),

            ('train_id', train_id),

            ('workdir', hyper.workdir),

            ('aug_brief', aug_brief),

            ('input_id', input_id),

            ('other_id', other_id),

            ('hyper', hyper.get_initkw()),

            ('train_hyper_id_long', train_hyper_id_long),
            ('train_hyper_id_brief', train_hyper_id_brief),
            ('train_hyper_hashid', train_hyper_hashid),
            ('init_history', init_history),
            ('init_history_hashid', _hash_data(util.make_idstr(init_history))),

            ('name', hyper.name),
            ('nice', hyper.name),

            ('old_train_dpath', normpath(
                join(hyper.workdir, 'fit', 'runs', train_hashid))),

            ('train_dpath', train_dpath),
            # ('link_dpath', link_dpath),

            # "nice" will be deprecated for "name_dpath"
            ('nice_dpath', nice_dpath),
            ('name_dpath', name_dpath),

            ('given_explicit_train_dpath', given_explicit_train_dpath),

            # TODO, add in classes if applicable
            # TODO, add in centering if applicable
            # ('centering', hyper.centering),

            ('other', hyper.other),

            # HACKED IN
            ('augment', hyper.augment_json()),

            ('extra', hyper.extra),

            ('argv', sys.argv),
            ('hostname', platform.node()),
        ])
        return train_info

    @classmethod
    def demo(HyperParams):
        from viame.pytorch import netharn as nh
        hyper = HyperParams(**{
            # ================
            # Environment Components
            'workdir'     : ub.ensure_app_cache_dir('netharn/tests/demo'),
            'name'        : 'demo',
            'xpu'         : nh.XPU.coerce('argv'),
            # workdir is a directory where intermediate results can be saved
            # name symlinks <workdir>/fit/name/<name> -> ../runs/<hashid>
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
        return hyper


def module_version_infos():
    """

    References:
        https://packaging.python.org/guides/single-sourcing-package-version/
    """
    try:
        from importlib import metadata
    except ImportError:
        # Running on pre-3.8 Python; use importlib-metadata package
        import importlib_metadata as metadata
    import sys
    modnames = ['torch', 'cv2', 'netharn', 'PIL', 'numpy']
    infos = []
    for modname in modnames:
        info = {'name': modname}

        try:
            module = sys.modules[modname]
            version_0 = getattr(module, '__version__', None)
        except Exception:
            version_0 = None

        try:
            version_1 = metadata.version(modname)
        except Exception:
            version_1 = None

        possible_versions = {version_1, version_0} - {None}
        if len(possible_versions) == 1:
            info['version'] = ub.peek(possible_versions)
        else:
            info['possible_versions'] = possible_versions

        if modname == 'torch':
            info['torch.version.cuda'] = torch.version.cuda
            info['torch.cuda.is_available()'] = torch.cuda.is_available()

        infos.append(info)

    # The conda info step is too slow (3 seconds)
    from .util.collect_env import get_env_info
    env_info = get_env_info()._asdict()
    info['__env__'] = env_info

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.hyperparams
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
