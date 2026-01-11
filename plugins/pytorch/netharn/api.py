"""
Newest parts of the top-level API


Concepts:
    # Netharn API Concepts

    TODO: documentation
"""
import ubelt as ub
import torch

try:  # nocover
    from packaging.version import parse as LooseVersion
except ImportError:
    from distutils.version import LooseVersion

_TORCH_IS_GE_1_2_0 = LooseVersion(torch.__version__) >= LooseVersion('1.2.0')


class Datasets(object):
    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts 'datasets', 'train_dataset', 'vali_dataset', and 'test_dataset'.

        Args:
            config (dict | str): coercable configuration dictionary.

        Returns:
            Dict: coco_datasets - note these are not torch datasets.
            They need to be used with ndsampler.

        Examples:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from viame.pytorch import netharn as nh
            >>> config = kw = {'datasets': 'special:shapes'}
            >>> print(ub.repr2(nh.api.Datasets.coerce(config, **kw)))

            >>> config = kw = {'datasets': 'special:shapes256'}
            >>> print(ub.repr2(nh.api.Datasets.coerce(config, **kw)))
        """
        from ndsampler import coerce_data
        config = _update_defaults(config, kw)
        torch_datasets = coerce_data.coerce_datasets(config)
        return torch_datasets


class DatasetInfo(object):
    """
    experimental, attempts to do more heavy lifting
    """
    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts 'datasets', 'train_dataset', 'vali_dataset', and 'test_dataset'.

        Args:
            config (dict | str): coercable configuration dictionary.
        """
        config = _update_defaults(config, kw)
        dataset_info = _coerce_datasets(config)
        return dataset_info


def _coerce_datasets(config):
    from viame.pytorch import netharn as nh
    import ndsampler
    import numpy as np
    from torchvision import transforms
    coco_datasets = nh.api.Datasets.coerce(config)
    print('coco_datasets = {}'.format(ub.repr2(coco_datasets, nl=1)))
    for tag, dset in coco_datasets.items():
        dset._build_hashid(hash_pixels=False)

    workdir = ub.ensuredir(ub.expandpath(config['workdir']))
    samplers = {
        tag: ndsampler.CocoSampler(dset, workdir=workdir, backend=config['sampler_backend'])
        for tag, dset in coco_datasets.items()
    }

    for tag, sampler in ub.ProgIter(list(samplers.items()), desc='prepare frames'):
        sampler.frames.prepare(workers=config['workers'])

    # TODO: basic ndsampler torch dataset, likely has to support the transforms
    # API, bleh.

    transform = transforms.Compose([
        transforms.Resize(config['input_dims']),
        transforms.CenterCrop(config['input_dims']),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    torch_datasets = {
        key: SamplerDataset(
            sapmler, transform=transform,
            # input_dims=config['input_dims'],
            # augmenter=config['augmenter'] if key == 'train' else None,
        )
        for key, sapmler in samplers.items()
    }
    # self = torch_dset = torch_datasets['train']

    if config['normalize_inputs']:
        # Get stats on the dataset (todo: turn off augmentation for this)
        import kwarray
        _dset = torch_datasets['train']
        stats_idxs = kwarray.shuffle(np.arange(len(_dset)), rng=0)[0:min(1000, len(_dset))]
        stats_subset = torch.utils.data.Subset(_dset, stats_idxs)

        cacher = ub.Cacher('dset_mean', depends=_dset.input_id + 'v3')
        input_stats = cacher.tryload()

        from .data.channel_spec import ChannelSpec
        channels = ChannelSpec.coerce(config['channels'])

        if input_stats is None:
            # Use parallel workers to load data faster
            from .data.data_containers import container_collate
            from functools import partial
            collate_fn = partial(container_collate, num_devices=1)

            loader = torch.utils.data.DataLoader(
                stats_subset,
                collate_fn=collate_fn,
                num_workers=config['workers'],
                shuffle=True,
                batch_size=config['batch_size'])

            # Track moving average of each fused channel stream
            channel_stats = {key: nh.util.RunningStats()
                             for key in channels.keys()}
            assert len(channel_stats) == 1, (
                'only support one fused stream for now')
            for batch in ub.ProgIter(loader, desc='estimate mean/std'):
                if isinstance(batch, (tuple, list)):
                    inputs = {'rgb': batch[0]}  # make assumption
                else:
                    inputs = batch['inputs']

                for key, val in inputs.items():
                    try:
                        for part in val.numpy():
                            channel_stats[key].update(part)
                    except ValueError:  # final batch broadcast error
                        pass

            perchan_input_stats = {}
            for key, running in channel_stats.items():
                running = ub.peek(channel_stats.values())
                perchan_stats = running.simple(axis=(1, 2))
                perchan_input_stats[key] = {
                    'std': perchan_stats['mean'].round(3),
                    'mean': perchan_stats['std'].round(3),
                }

            input_stats = ub.peek(perchan_input_stats.values())
            cacher.save(input_stats)
    else:
        input_stats = {}

    torch_loaders = {
        tag: dset.make_loader(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            num_workers=config['workers'],
            shuffle=(tag == 'train'),
            balance=(config['balance'] if tag == 'train' else None),
            pin_memory=True)
        for tag, dset in torch_datasets.items()
    }

    dataset_info = {
        'torch_datasets': torch_datasets,
        'torch_loaders': torch_loaders,
        'input_stats': input_stats
    }
    return dataset_info


class SamplerDataset(torch.utils.data.Dataset):
    def __init__(self, sampler, transform=None, return_style='torchvision'):
        self.sampler = sampler
        self.transform = transform
        self.return_style = return_style
        self.input_id = self.sampler.hashid

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        item = self.sampler.load_item(index)
        numpy_im = item['im']

        if self.transform:
            from PIL import Image
            pil_im = Image.fromarray(numpy_im)
            torch_chw = self.transform(pil_im)
        else:
            torch_chw = torch.from_numpy(numpy_im).permute(2, 0, 1).float()
            # raise NotImplementedError

        if self.return_style == 'torchvision':
            cid = item['tr']['category_id']
            cidx = self.sampler.classes.id_to_idx[cid]
            return torch_chw, cidx
        else:
            raise NotImplementedError

    def make_loader(self, batch_size=16, num_batches='auto', num_workers=0,
                    shuffle=False, pin_memory=False, drop_last=False,
                    balance=None):

        import kwarray
        if len(self) == 0:
            raise Exception('must have some data')

        def worker_init_fn(worker_id):
            import numpy as np
            for i in range(worker_id + 1):
                seed = np.random.randint(0, int(2 ** 32) - 1)
            seed = seed + worker_id
            kwarray.seed_global(seed)
            # if self.augmenter:
            #     rng = kwarray.ensure_rng(None)
            #     self.augmenter.seed_(rng)

        loaderkw = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'worker_init_fn': worker_init_fn,
        }
        if balance is None:
            loaderkw['shuffle'] = shuffle
            loaderkw['batch_size'] = batch_size
            loaderkw['drop_last'] = drop_last
        elif balance == 'classes':
            from .data.batch_samplers import BalancedBatchSampler
            index_to_cid = [
                cid for cid in self.sampler.regions.targets['category_id']
            ]
            batch_sampler = BalancedBatchSampler(
                index_to_cid, batch_size=batch_size,
                shuffle=shuffle, num_batches=num_batches)
            loaderkw['batch_sampler'] = batch_sampler
        else:
            raise KeyError(balance)

        loader = torch.utils.data.DataLoader(self, **loaderkw)
        return loader


class Initializer(object):
    """
    Base class for all netharn initializers
    """
    def __call__(self, model, *args, **kwargs):
        return self.forward(model, *args, **kwargs)

    def forward(self, model):
        """
        Abstract function that does the initailization
        """
        raise NotImplementedError('implement me')

    def history(self):
        """
        Initializer methods have histories which are short for algorithms and
        can be quite long for pretrained models
        """
        return None

    def get_initkw(self):
        """
        Initializer methods have histories which are short for algorithms and
        can be quite long for pretrained models
        """
        initkw = self.__dict__.copy()
        # info = {}
        # info['__name__'] = self.__class__.__name__
        # info['__module__'] = self.__class__.__module__
        # info['__initkw__'] = initkw
        return initkw

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts 'init', 'pretrained', 'pretrained_fpath', 'leftover', and
        'noli'.

        Args:
            config (dict | str): coercable configuration dictionary.
                if config is a string it is taken as the value for "init".

        Returns:
            Tuple[nh.Initializer, dict]: initializer_ = initializer_cls, kw

        Examples:
            >>> from viame.pytorch import netharn as nh
            >>> print(ub.repr2(nh.Initializer.coerce({'init': 'noop'})))
            (
                <class 'netharn.initializers.core.NoOp'>,
                {},
            )
            >>> config = {
            ...     'init': 'pretrained',
            ...     'pretrained_fpath': '/fit/nice/untitled'
            ... }
            >>> print(ub.repr2(nh.Initializer.coerce(config)))
            (
                <class 'netharn.initializers.pretrained.Pretrained'>,
                {... 'fpath': '/fit/nice/untitled', 'leftover': None, 'mangle': False},
            )
            >>> print(ub.repr2(nh.Initializer.coerce({'init': 'kaiming_normal'})))
            (
                <class 'netharn.initializers.core.KaimingNormal'>,
                {'param': 0},
            )
        """
        from viame.pytorch import netharn as nh
        import six
        if isinstance(config, six.string_types):
            config = {
                'init': config,
            }

        config = _update_defaults(config, kw)

        pretrained_fpath = config.get('pretrained_fpath', config.get('pretrained', None))
        init = config.get('initializer', config.get('init', None))

        # Allow init to specify a pretrained fpath
        if isinstance(init, six.string_types) and pretrained_fpath is None:
            from os.path import exists
            pretrained_cand = ub.expandpath(init)
            if exists(pretrained_cand):
                pretrained_fpath = pretrained_cand

        config['init'] = init
        config['pretrained_fpath'] = pretrained_fpath
        config['pretrained'] = pretrained_fpath

        if pretrained_fpath is not None:
            config['init'] = 'pretrained'

        # ---
        initializer_ = None
        if config['init'].lower() in ['kaiming_normal']:
            initializer_ = (nh.initializers.KaimingNormal, {
                # initialization params should depend on your choice of
                # nonlinearity in your model. See the Kaiming Paper for details.
                'param': 1e-2 if config.get('noli', 'relu') == 'leaky_relu' else 0,
            })
        elif config['init'] == 'noop':
            initializer_ = (nh.initializers.NoOp, {})
        elif config['init'] == 'pretrained':
            initializer_ = (nh.initializers.Pretrained, {
                'fpath': ub.expandpath(config['pretrained_fpath']),
                'leftover': kw.get('leftover', None),
                'mangle': kw.get('mangle', False),
                'association': kw.get('association', None),
            })
        elif config['init'] == 'cls':
            # Indicate that the model will initialize itself
            # We have to trust that the user does the right thing here.
            pass
        else:
            raise KeyError('Unknown coercable init: {!r}'.format(config['init']))
        return initializer_


class Optimizer(object):

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts keywords:
            optimizer / optim :
                can be sgd, adam, adamw, rmsprop

            learning_rate / lr :
                a float

            weight_decay / decay :
                a float

            momentum:
                a float, only used if the optimizer accepts it

            params:
                This is a SPECIAL keyword that is handled differently.  It is
                interpreted by `netharn.hyper.Hyperparams.make_optimizer`.

                In this simplest case you can pass "params" as a list of torch
                parameter objects or a list of dictionaries containing param
                groups and special group options (just as you would when
                constructing an optimizer from scratch). We don't recommend
                this while using netharn unless you know what you are doing
                (Note, that params will correctly change device if the model is
                mounted).

                In the case where you do not want to group parameters with
                different options, it is best practice to simply not specify
                params.

                In the case where you want to group parameters set params to
                either a List[Dict] or a Dict[str, Dict].

                The items / values of this collection should be a dictionary.
                The keys / values of this dictionary should be the per-group
                optimizer options. Additionally, there should be a key "params"
                (note this is a nested per-group params not to be confused with
                the top-level "params").

                Each per-group "params" should be either (1) a list of
                parameter names (preferred), (2) a string that specifies a
                regular expression (matching layer names will be included in
                this group), or (3) a list of parameter objects.

                For example, the top-level params might look like:

                    params={
                        'head': {'lr': 0.003, 'params': '.*head.*'},
                        'backbone': {'lr': 0.001, 'params': '.*backbone.*'},
                        'preproc': {'lr': 0.0, 'params': [
                            'model.conv1', 'model.norm1', , 'model.relu1']}
                    }

                Note that head and backbone specify membership via regular
                expression whereas preproc explicitly specifies a list of
                parameter names.

        Notes:
            pip install torch-optimizer

        Returns:
            Tuple[type, dict]: a type and arguments to construct it

        References:
            https://datascience.stackexchange.com/questions/26792/difference-between-rmsprop-with-momentum-and-adam-optimizers
            https://github.com/jettify/pytorch-optimizer

        CommandLine:
            xdoctest -m /home/joncrall/code/netharn/netharn/api.py Optimizer.coerce

        Example:
            >>> config = {'optimizer': 'sgd', 'params': [
            >>>     {'lr': 3e-3, 'params': '.*head.*'},
            >>>     {'lr': 1e-3, 'params': '.*backbone.*'},
            >>> ]}
            >>> optim_ = Optimizer.coerce(config)

            >>> # xdoctest: +REQUIRES(module:torch_optimizer)
            >>> from .api import *  # NOQA
            >>> config = {'optimizer': 'DiffGrad'}
            >>> optim_ = Optimizer.coerce(config, lr=1e-5)
            >>> print('optim_ = {!r}'.format(optim_))
            >>> assert optim_[1]['lr'] == 1e-5

            >>> config = {'optimizer': 'Yogi'}
            >>> optim_ = Optimizer.coerce(config)
            >>> print('optim_ = {!r}'.format(optim_))

            >>> from .api import *  # NOQA
            >>> Optimizer.coerce({'optimizer': 'ASGD'})

        TODO:
            - [ ] https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
        """
        from viame.pytorch import netharn as nh
        config = _update_defaults(config, kw)
        key = config.get('optimizer', config.get('optim', 'sgd')).lower()
        lr = config.get('learning_rate', config.get('lr', 3e-3))
        decay = config.get('weight_decay', config.get('decay', 0))
        momentum = config.get('momentum', 0.9)
        params = config.get('params', None)
        # TODO: allow for "discriminative fine-tuning"
        if key == 'sgd':
            cls = torch.optim.SGD
            kw = {
                'lr': lr,
                'weight_decay': decay,
                'momentum': momentum,
                'nesterov': True,
            }
        elif key == 'adam':
            cls = torch.optim.Adam
            kw = {
                'lr': lr,
                'weight_decay': decay,
                # 'betas': (0.9, 0.999),
                # 'eps': 1e-8,
                # 'amsgrad': False
            }
        elif key == 'adamw':
            if _TORCH_IS_GE_1_2_0:
                from torch.optim import AdamW
                cls = AdamW
            else:
                cls = nh.optimizers.AdamW
            kw = {
                'lr': lr,
                'weight_decay': decay,
                # 'betas': (0.9, 0.999),
                # 'eps': 1e-8,
                # 'amsgrad': False
            }
        elif key == 'rmsprop':
            cls = torch.optim.RMSprop
            kw = {
                'lr': lr,
                'weight_decay': decay,
                'momentum': momentum,
                'alpha': 0.9,
            }
        else:
            from .util import util_inspect
            _lut = {}

            optim_modules = [
                torch.optim,
            ]

            try:
                # Allow coerce to use torch_optimizer package if available
                import torch_optimizer
            except Exception:
                torch_optimizer = None
            else:
                optim_modules.append(torch_optimizer)
                _lut.update({
                    k.lower(): c.__name__
                    for k, c in torch_optimizer._NAME_OPTIM_MAP.items()})

            _lut.update({
                k.lower(): k for k in dir(torch.optim)
                if not k.startswith('_')})

            key = _lut[key]

            cls = None
            for module in optim_modules:
                cls = getattr(module, key, None)
                if cls is not None:
                    defaultkw = util_inspect.default_kwargs(cls)
                    kw = defaultkw.copy()
                    kw.update(ub.dict_isect(config, kw))
                    # Hacks for common cases, otherwise if learning_rate is
                    # given, but only lr exists in the signature, it will be
                    # incorrectly ignored.
                    if 'lr' in kw:
                        kw['lr'] = lr
                    if 'weight_decay' in kw:
                        kw['weight_decay'] = decay
                    break

        if cls is None:
            raise KeyError(key)

        kw['params'] = params
        optim_ = (cls, kw)
        return optim_


class Dynamics(object):
    """
    Dynamics are essentially configurations of "tricks" that can be used for
    training deep networks.
    """

    @staticmethod
    def coerce(config={}, **kw):
        """
        Kwargs:
            bstep / batch_step,
                Controls how many batches to process before taking a step in the
                gradient direction. Effectively simulates a batch_size that is
                `bstep` times bigger.

            grad_norm_max:
                clips gradients to a max value (mmdet likes to use 35 for this)

            grad_norm_type:
                p-norm to use if clipping grads.

            warmup_iters: EXPERIMENTAL

            warmup_ratio: EXPERIMENTAL

        Example:
            >>> print(Dynamics.coerce({'bstep': 2}))
            >>> print(Dynamics.coerce({'batch_step': 3}))
            >>> print(Dynamics.coerce({'grad_norm_max': 35}))
        """
        config = _update_defaults(config, kw)
        _default_dynamics = {
            'batch_step': 1,        # simulates larger batch sizes
            'grad_norm_max': None,  # clips gradients to a max value (mmdet likes to use 35 for this)
            'grad_norm_type': 2,    # p-norm to use if clipping grads.

            'warmup_iters': 0,          # CURRENTLY HACKED AND EXPERIMENTAL
            'warmup_ratio': 1.0 / 3.0,  # CURRENTLY HACKED AND EXPERIMENTAL
        }
        _aliases = {
            'batch_step': ['bstep'],
        }
        dynamics_ = {}
        for primary_key, default_value in _default_dynamics.items():
            value = default_value
            _keys = [primary_key] + _aliases.get(primary_key, [])
            for alias_key in _keys:
                if alias_key in config:
                    value = config.get(alias_key, default_value)
                    break
            dynamics_[primary_key] = value
        return dynamics_


class Scheduler(object):

    def step_batch(self, bx=None):
        raise NotImplementedError

    def step_epoch(self, epoch=None):
        raise NotImplementedError

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts keywords:
            scheduler / schedule
            learning_rate / lr

            for scheduler == exponential:
                gamma
                stepsize

            scheduler accepts several special strings which involves a keyword
            followed by a special coded string that can be used to modify
            parameters. Some examples:

                step-10-30-50-100 - multiply LR by 0.1 at every point

                onecycle90 - a cyclic scheduler peaking at the epoch 90 // 2

                onecycle90-p0.2 - a cyclic scheduler peaking at the int(90 * 0.2)

                ReduceLROnPlateau-p2-c2 - a ReduceLROnPlateau scheduler with
                    a patience of 2 and a cooldown of 2

                Exponential-g0.98-s1 - exponential decay of 0.98 every 1-th
                    epoch
        """
        from viame.pytorch import netharn as nh
        import parse
        config = _update_defaults(config, kw)
        key = config.get('scheduler', config.get('schedule', 'step90'))
        lr = config.get('learning_rate', config.get('lr', 3e-3))

        if key.startswith('onecycle'):
            result = parse.parse('onecycle{:d}-{}', key)
            size = result.fixed[0]
            suffix = result.fixed[1]

            parts = suffix.split('-')
            kw = {
                'peak': size // 2,
            }
            try:
                for part in parts:
                    if not part:
                        continue
                    if part.startswith('p'):
                        valstr = part[1:]
                        if valstr.startswith('0.'):
                            kw['peak'] = int(size * float(valstr))
                        else:
                            kw['peak'] = int(valstr)
                    else:
                        raise ValueError('unknown {} part'.format(suffix))
            except Exception:
                raise ValueError('Unable to parse {} specs: {}'.format(
                    result, suffix))

            scheduler_ = (nh.schedulers.ListedScheduler, {
                'points': {
                    'lr': {
                        size * 0   : lr * 0.1,
                        kw['peak'] : lr * 1.0,
                        size * 1   : lr * 0.01,
                        size + 1   : lr * 0.001,
                    },
                    'momentum': {
                        size * 0   : 0.95,
                        kw['peak'] : 0.90,
                        size * 1   : 0.95,
                        size + 1   : 0.999,
                    },
                },
            })
            return scheduler_

        prefix = 'step'.lower()
        if key.lower().startswith(prefix):
            # Allow step to specify `-` separated step points
            suffix = key[len(prefix):]
            param_parts = suffix.split('-')
            if param_parts and param_parts[-1].startswith('f'):
                factor = float(param_parts[-1][1:])
                param_parts = param_parts[:-1]
            else:
                factor = 10
            points = [int(p) for p in param_parts if p]
            assert sorted(points) == points, 'points must be in order'
            lr_pts = {0: lr}
            for i, epoch in enumerate(points, start=1):
                lr_pts[epoch] = lr / (factor ** i)

            scheduler_ = (nh.schedulers.ListedScheduler, {
                'points': {
                    'lr': lr_pts,
                },
                'interpolation': 'left'
            })
            return scheduler_

        prefix = 'ReduceLROnPlateau'.lower()
        if key.lower().startswith(prefix):
            # Allow specification of scheduler params
            suffix = key[len(prefix):]
            parts = suffix.split('-')
            kw = {
                'patience': 10,
                'cooldown': 0,
                'factor': 0.1,
            }
            try:
                for part in parts:
                    if not part:
                        continue
                    if part.startswith('f'):
                        kw['factor'] = float(part[1:])
                    elif part.startswith('p'):
                        kw['patience'] = int(part[1:])
                    elif part.startswith('c'):
                        kw['cooldown'] = int(part[1:])
                    else:
                        raise ValueError('unknown {} part'.format(prefix))
            except Exception:
                raise ValueError('Unable to parse {} specs: {}'.format(
                    prefix, suffix))

            scheduler_ = (torch.optim.lr_scheduler.ReduceLROnPlateau, kw)
            return scheduler_

        prefix = 'Exponential'.lower()
        if key.lower().startswith(prefix):
            # Allow specification of scheduler params
            suffix = key[len(prefix):]
            parts = suffix.split('-')
            kw = {
                'gamma': config.get('gamma', 0.1),
                'stepsize': config.get('stepsize', 100),
            }
            try:
                for part in parts:
                    if not part:
                        continue
                    if part.startswith('g'):
                        kw['gamma'] = float(part[1:])
                    elif part.startswith('s'):
                        kw['stepsize'] = int(part[1:])
                    else:
                        raise ValueError('unknown {} part'.format(prefix))
            except Exception:
                raise ValueError('Unable to parse {} specs: {}'.format(
                    prefix, suffix))
            scheduler_ = (nh.schedulers.Exponential, kw)
            return scheduler_

        raise KeyError(key)


class Loaders(object):

    @staticmethod
    def coerce(datasets, config={}, **kw):
        config = _update_defaults(config, kw)
        loaders = {}
        for key, dset in datasets.items():
            if hasattr(dset, 'make_loader'):
                loaders[key] = dset.make_loader(
                    batch_size=config['batch_size'],
                    num_workers=config['workers'], shuffle=(key == 'train'),
                    pin_memory=True)
            else:
                loaders[key] = torch.utils.data.DataLoader(
                    dset, batch_size=config['batch_size'], num_workers=config['workers'],
                    shuffle=(key == 'train'), pin_memory=True)
        return loaders


class Criterion(object):

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts keywords:
            criterion / loss - one of: (
                contrastive, focal, triplet, cross_entropy, mse)
        """
        raise NotImplementedError


def configure_hacks(config={}, **kw):
    """
    Configures hacks to fix global settings in external modules

    Args:
        config (dict): exected to contain certain special keys.

            * "workers" with an integer value equal to the number of dataloader
                processes.

            * "sharing_strategy" to specify the torch multiprocessing backend

        **kw: can also be used to specify config items

    Modules we currently hack:
        * cv2 - fix thread count
        * torch sharing strategy
    """
    import torch
    config = _update_defaults(config, kw)

    if config.get('workers', 0) > 0:
        try:
            import cv2
        except ImportError:
            pass
        else:
            cv2.setNumThreads(0)

    strat = config.get('sharing_strategy', None)
    if strat is not None and strat != 'default':
        if strat == 'auto':
            # TODO: can we add a better auto test?
            strat = torch.multiprocessing.get_sharing_strategy()
        valid_strats = torch.multiprocessing.get_all_sharing_strategies()
        if strat not in valid_strats:
            raise KeyError('start={} is not in valid_strats={}'.format(strat, valid_strats))
        torch.multiprocessing.set_sharing_strategy(strat)

    if 0:
        """
        References:
            https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
        """
        # torch.multiprocessing.get_all_start_methods()
        # torch_multiprocessing.get_context()
        torch.multiprocessing.set_start_method('spawn')


def configure_workdir(config={}, **kw):
    config = _update_defaults(config, kw)
    if config['workdir'] is None:
        config['workdir'] = kw['workdir']
    workdir = config['workdir'] = ub.expandpath(config['workdir'])
    ub.ensuredir(workdir)
    return workdir


def _update_defaults(config, kw):
    config = dict(config)
    for k, v in kw.items():
        if k not in config:
            config[k] = v
    return config
