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
    from netharn.layers import common
    from netharn.layers import conv_norm
    from netharn.layers import gauss
    from netharn.layers import norm
    from netharn.layers import perceptron
    from netharn.layers import rectify
    from netharn.layers import reshape

    from netharn.layers.common import (AnalyticModule, Identity, Module,
                                       ModuleMixin, Sequential,)
    from netharn.layers.conv_norm import (ConvNorm1d, ConvNorm2d, ConvNorm3d,
                                          ConvNormNd,)
    from netharn.layers.gauss import (Conv1d_pad, Conv2d_pad, GaussianBlurNd,)
    from netharn.layers.norm import (InputNorm, L2Norm,)
    from netharn.layers.perceptron import (MultiLayerPerceptronNd,)
    from netharn.layers.rectify import (rectify_conv, rectify_dropout,
                                        rectify_maxpool, rectify_nonlinearity,
                                        rectify_normalizer,)
    from netharn.layers.reshape import (Permute, Reshape,)

    __all__ = ['AnalyticModule', 'Conv1d_pad', 'Conv2d_pad', 'ConvNorm1d',
               'ConvNorm2d', 'ConvNorm3d', 'ConvNormNd', 'GaussianBlurNd',
               'Identity', 'InputNorm', 'L2Norm', 'Module', 'ModuleMixin',
               'MultiLayerPerceptronNd', 'Permute', 'Reshape', 'Sequential',
               'common', 'conv_norm', 'gauss', 'norm', 'perceptron', 'rectify',
               'rectify_conv', 'rectify_dropout', 'rectify_maxpool',
               'rectify_nonlinearity', 'rectify_normalizer', 'reshape']
