import six
import torch
import ubelt as ub
from os.path import dirname
from os.path import exists
from os.path import join
from os.path import normpath
from viame.pytorch.netharn import api
from .functional import load_partial_state


class Pretrained(api.Initializer, ub.NiceRepr):
    """
    This class initializes a model with pretrained weights from a file on disk.

    If the model topology is slightly different (e.g. the shape of the final
    layer changes), the weights are partially applied. See
    `netharn.initializers.functional.load_partial_state` for mor details on
    this process.

    Attributes:
        fpath (str | PathLike): location of the pretrained weights file.

            This can be a pytorch '.pt' file containing the model state, a path
            to a netharn deploy '.zip' file.

            While it is best practice to use an explicit filepath, we do allow
            `fpath` be a "fuzzy" glob string as long as the pattern resolves to
            a single file, otherwise an error will be thrown.

        leftover (netharn.Initializer | str): backup initializer if the weights
            can only be partially applied. I.E. The initializer applied to the
            leftover weights that were not in the pretrained file. Can either
            be an initializer class or a coercable initializer string.

        mangle (bool, default=True): If True, mangles tensors that have the
            same key, but different shapes forcing them to fit. This might
            destroy information when forcing a a larger tensor into a smaller
            tensor, or leave extra uninitialized room when a small tensor is
            placed in a larger one. Note be careful when mangling a
            classification layer if class indexes are not aligned.

        association (str): controls how we search for the association between
            the two model states. Can be strict, module-hack, prefix-hack, or
            embedding.

        info (dict, optional): specify explicit history info

        initializer (netharn.Initializer): DEPRECATED use the `leftover`.

    Example:
        >>> from .initializers.pretrained import *
        >>> from .models import toynet
        >>> from os.path import join
        >>> # Save "pretrained" weights to disk
        >>> model1 = toynet.ToyNet2d()
        >>> dpath = ub.ensure_app_cache_dir('netharn', 'tests')
        >>> fpath = join(dpath, 'toynet_weights.pt')
        >>> torch.save(model1.state_dict(), fpath)
        >>> # Create the initializer and point to the pretrained weights
        >>> self = Pretrained(fpath)
        >>> # Apply the pretrained weights to a new model
        >>> model2 = toynet.ToyNet2d()
        >>> self(model2)
        >>> # xdoctest: +SKIP
        >>> # use experimental ubelt features to hash the model state
        >>> ub.util_hash._HASHABLE_EXTENSIONS._register_torch_extensions()
        >>> ub.util_hash._HASHABLE_EXTENSIONS._register_agressive_extensions()
        >>> hash1 = ub.hash_data(model2.state_dict())
        >>> hash2 = ub.hash_data(model1.state_dict())
        >>> assert hash1 == hash2

    Example:
        >>> from .initializers.pretrained import *
        >>> from .models import toynet
        >>> from os.path import join
        >>> # Save "pretrained" weights to disk
        >>> model1 = toynet.ToyNet2d(num_classes=2)
        >>> model2 = toynet.ToyNet2d(num_classes=3)
        >>> dpath = ub.ensure_app_cache_dir('netharn', 'tests')
        >>> fpath = join(dpath, 'toynet_weights1.pt')
        >>> torch.save(model1.state_dict(), fpath)
        >>> # Create the initializer and point to the pretrained weights
        >>> self = Pretrained(fpath, leftover='kaiming_normal')
        >>> # Apply the partial pretrained weights to a new model
        >>> self(model2)
    """
    def __init__(self, fpath, leftover=None, mangle=True, info=None,
                 initializer=None, association=None):
        if initializer is not None:
            import warnings
            warnings.warn('Pretrained `initializer` kwarg is deprecated '
                          'in favor of `leftover`', DeprecationWarning)
            leftover = initializer

        self.fpath = fpath
        if isinstance(leftover, six.string_types):
            initializer_ = api.Initializer.coerce(initializer=leftover)
            leftover = initializer_[0](**initializer_[1])

        self.leftover = leftover
        self.association = association
        self.mangle = mangle
        self.info = info

    def __nice__(self):
        return self.fpath

    def _rectify_deploy_zip_weights_path(self):
        # Find the path to the weights inside the zipfile
        import zipfile
        fpath = None
        candidates = []
        with zipfile.ZipFile(self.fpath, 'r') as myzip:
            for zinfo in myzip.filelist:
                if zinfo.filename.endswith('deploy_snapshot.pt'):
                    candidates = [zinfo.filename]
                    break
                elif zinfo.filename.endswith('.pt'):
                    candidates.append(zinfo.filename)
        if len(candidates) == 0:
            raise OSError('Cannot find pretrained weights in {}'.format(
                self.fpath))
        elif len(candidates) > 1:
            raise OSError('Multiple weights files in {}'.format(
                self.fpath))
        else:
            fpath = join(self.fpath, candidates[0])
        return fpath

    def _rectify_fpath(self):
        """
        Resolves the `self.fpath`, which may be non-physical path (e.g.
        globstring or zipfile) to an existing physical path if possible.
        """
        if self.fpath is None:
            raise ValueError('Pretrained fpath is None!')
        # Handle torch deployment zipfiles
        if exists(self.fpath) and self.fpath.endswith('.zip'):
            fpath = self._rectify_deploy_zip_weights_path()
        else:
            fpath = self.fpath
            if not exists(fpath) and '*' in fpath:
                import glob
                cands = list(glob.glob(fpath))
                if len(cands) == 1:
                    fpath = cands[0]
                else:
                    raise Exception(
                        'Pattern fpath={!r} must resolve to exactly one file, '
                        'but got cands{!r}'.format(fpath, cands))
        return fpath

    def _load_model_state(self, xpu=None):
        """
        Load the model state from a path or from within a zipfile
        """
        from viame.pytorch import netharn as nh
        from viame.pytorch.netharn import XPU

        fpath = self._rectify_fpath()
        xpu = XPU.coerce('cpu')

        try:
            file = nh.util.zopen(fpath, 'rb', seekable=True)
            model_state_dict = xpu.load(file)
        except Exception:
            print('Failed to open fpath = {!r}'.format(fpath))
            raise
        return model_state_dict

    def forward(self, model, verbose=2):
        """
        Apply the pretrained weights to the model
        """
        from viame.pytorch.netharn import XPU
        xpu = XPU.from_data(model)

        model_state_dict = self._load_model_state(xpu=xpu)

        if 'model_state_dict' in model_state_dict:
            model_state_dict = model_state_dict['model_state_dict']
        elif 'state_dict' in model_state_dict:
            model_state_dict = model_state_dict['state_dict']
        elif 'weights' in model_state_dict:
            model_state_dict = model_state_dict['weights']
        else:
            # If the dictionary is flat (i.e. all values are tensors) then it
            # is safe to assume this file only contains weights.
            # Otherwise raise an exception.
            if not all(torch.is_tensor(v) for v in model_state_dict.values()):
                raise Exception(
                    'snapshot file is nested, but does not have expected keys: '
                    'model_state_dict or weights. Root keys are {}'.format(
                        sorted(model_state_dict.keys())
                    ))
        # Remove any DataParallel / DataSerial
        raw_model = xpu.raw(model)
        info = load_partial_state(raw_model, model_state_dict,
                                  leftover=self.leftover,
                                  mangle=self.mangle,
                                  association=self.association,
                                  verbose=verbose)
        return info

    def history(self):
        """
        if available return the history of the model as well
        """
        from viame.pytorch import netharn as nh
        if self.info is None:
            # TODO: check for train_info.json in a few different places
            fpath = self._rectify_fpath()
            snap_fpath = ub.expandpath(fpath)
            candidate_paths = [
                join(dirname(snap_fpath), 'train_info.json'),
                join(dirname(dirname(snap_fpath)), 'train_info.json'),
            ]
            info = None
            for info_fpath in candidate_paths:
                info_fpath = normpath(info_fpath)
                try:
                    # Info might be inside of a zipfile
                    info = nh.util.read_json(nh.util.zopen(info_fpath))
                    break
                except Exception:
                    pass
            if info is None:
                info = '__UNKNOWN__'
        else:
            info = self.info
        return info
