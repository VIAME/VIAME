# -*- coding: utf-8 -*-
from collections import OrderedDict
try:
    import imgaug
    _Augmenter = imgaug.augmenters.Augmenter
except Exception:
    # imgaug is deprecated, don't warn and dont use
    imgaug = None
    if 0:
        import warnings
        warnings.warn('imgaug is not availble', DeprecationWarning)
    _Augmenter = object


class ParamatarizedAugmenter(_Augmenter):
    """
    Helper that automatically registers stochastic parameters
    """

    def __init__(self, *args, **kwargs):
        if imgaug is None:
            raise Exception('imgaug is not available, but is needed to create an instance of ParamatarizedAugmenter. Moving away from imgaug is encouraged')
        super(ParamatarizedAugmenter, self).__setattr__('_initialized', True)
        super(ParamatarizedAugmenter, self).__setattr__('_registered_params', OrderedDict())
        super(ParamatarizedAugmenter, self).__init__(*args, **kwargs)

    def _setparam(self, name, value):
        self._registered_params[name] = value
        setattr(self, name, value)

    def get_parameters(self):
        return list(self._registered_params.values())

    def __setattr__(self, key, value):
        if not getattr(self, '_initialized', False) and key != '_initialized':
            raise Exception(
                ('Must call super().__init__ in {} that inherits '
                 'from Augmenter2').format(self.__class__))
        if not key.startswith('_'):
            if key in self._registered_params:
                self._registered_params[key] = value
            elif isinstance(value, imgaug.parameters.StochasticParameter):
                self._registered_params[key] = value
        super(ParamatarizedAugmenter, self).__setattr__(key, value)

    def _augment_heatmaps(self):
        raise NotImplementedError

    @staticmethod
    def _hack_get_named_params(self):
        """ hopefully imgaug will add get_named_params. Until then hack it. """
        named_params = OrderedDict()
        params = self.get_parameters()
        if params:
            # See if we can hack to gether what the param names are
            unused = OrderedDict(sorted(self.__dict__.items()))
            for p in params:
                found = False
                for key, value in list(unused.items()):
                    if p is value:
                        named_params[key] = p
                        unused.pop(key)
                        found = True
                if not found:
                    key = '__UNKNOWN_PARAM_NAME_{}__'.format(len(named_params))
                    named_params[key] = p
        return named_params

    @staticmethod
    def _json_id(aug):
        """
        TODO:
            - [ ] submit a PR to imgaug that registers parameters with classes

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> from .data.transforms.augmenter_base import *
            >>> import imgaug.augmenters as iaa
            >>> import imgaug
            >>> _PA = ParamatarizedAugmenter
            >>> augment = imgaug.augmenters.Affine()
            >>> info = _PA._json_id(augment)
            >>> assert info['__class__'] == 'Affine'
            >>> assert _PA._json_id('') == ''
            >>> #####
            >>> augmentors = [
            >>>     iaa.Fliplr(p=.5),
            >>>     iaa.Flipud(p=.5),
            >>>     iaa.Affine(
            >>>         scale={"x": (1.0, 1.01), "y": (1.0, 1.01)},
            >>>         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            >>>         rotate=(-15, 15),
            >>>         shear=(-7, 7),
            >>>         order=[0, 1, 3],
            >>>         cval=(0, 255),
            >>>         mode=imgaug.ALL,  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            >>>         # Note: currently requires imgaug master version
            >>>         backend='cv2',
            >>>     ),
            >>>     iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
            >>>     iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
            >>> ]
            >>> augment = iaa.Sequential(augmentors)
            >>> info = _PA._json_id(augment)
            >>> import ubelt as ub
            >>> print(ub.repr2(info, nl=2, precision=2))
        """
        _PA = ParamatarizedAugmenter
        if isinstance(aug, tuple):
            return [_PA._json_id(item) for item in aug]
        elif isinstance(aug, imgaug.parameters.StochasticParameter):
            return str(aug)
        elif isinstance(aug, imgaug.augmenters.Augmenter):
            info = OrderedDict()
            info['__class__'] = aug.__class__.__name__
            try:
                params = _PA._hack_get_named_params(aug)
                if params:
                    info['params'] = params
                if isinstance(aug, list):
                    children = aug[:]
                    children = [ParamatarizedAugmenter._json_id(c) for c in children]
                    info['children'] = children
                return info
            except Exception as ex:
                print(ex)
                # imgaug is weird and buggy
                info['__str__'] = str(aug)
        else:
            return str(aug)


def imgaug_json_id(aug):
    """
    Creates a json-like encoding that represents an imgaug augmentor

    Example:
        >>> # xdoctest: +REQUIRES(module:imgaug)
        >>> import imgaug.augmenters as iaa
        >>> import imgaug
        >>> from viame.pytorch import netharn as nh
        >>> augment = imgaug.augmenters.Affine()
        >>> info = nh.data.transforms.imgaug_json_id(augment)
        >>> import ubelt as ub
        >>> print(ub.repr2(info, nl=2, precision=2))
    """
    import imgaug
    if isinstance(aug, tuple):
        return [imgaug_json_id(item) for item in aug]
    elif isinstance(aug, imgaug.parameters.StochasticParameter):
        return str(aug)
    else:
        try:
            info = OrderedDict()
            info['__class__'] = aug.__class__.__name__
            params = aug.get_parameters()
            if params:
                info['params'] = [imgaug_json_id(p) for p in params]
            if isinstance(aug, list):
                children = aug[:]
                children = [imgaug_json_id(c) for c in children]
                info['children'] = children
            return info
        except Exception:
            # imgaug is weird and buggy
            return str(aug)
