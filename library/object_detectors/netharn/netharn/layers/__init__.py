"""
mkinit netharn.layers -w
"""
# flake8: noqa

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from . import common
    from . import conv_norm
    from . import gauss
    from . import norm
    from . import perceptron
    from . import rectify
    from . import reshape

    from .common import (AnalyticModule, Identity, Module,
                         ModuleMixin, Sequential,)
    from .conv_norm import (ConvNorm1d, ConvNorm2d, ConvNorm3d,
                            ConvNormNd,)
    from .gauss import (Conv1d_pad, Conv2d_pad, GaussianBlurNd,)
    from .norm import (InputNorm, L2Norm,)
    from .perceptron import (MultiLayerPerceptronNd,)
    from .rectify import (rectify_conv, rectify_dropout,
                          rectify_maxpool, rectify_nonlinearity,
                          rectify_normalizer,)
    from .reshape import (Permute, Reshape,)

    __all__ = ['AnalyticModule', 'Conv1d_pad', 'Conv2d_pad', 'ConvNorm1d',
               'ConvNorm2d', 'ConvNorm3d', 'ConvNormNd', 'GaussianBlurNd',
               'Identity', 'InputNorm', 'L2Norm', 'Module', 'ModuleMixin',
               'MultiLayerPerceptronNd', 'Permute', 'Reshape', 'Sequential',
               'common', 'conv_norm', 'gauss', 'norm', 'perceptron', 'rectify',
               'rectify_conv', 'rectify_dropout', 'rectify_maxpool',
               'rectify_nonlinearity', 'rectify_normalizer', 'reshape']
