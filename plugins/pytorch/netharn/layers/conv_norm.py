# -*- coding: utf-8 -*-
from viame.pytorch.netharn.analytic.output_shape_for import OutputShapeFor
from viame.pytorch.netharn.layers import rectify
from viame.pytorch.netharn.layers import common
import ubelt as ub  # NOQA


__devnotes__ = """

There are plenty of resources discussing what the best ordering of
convolution, normalization, and non-linearities are. Our implementation
may not be the best. We may also consider incorporating dropout into
this block layer.

Li 2018 - "Understanding the disharmony between dropout and batch
normalization by variance shift" - https://arxiv.org/pdf/1801.05134.pdf

    * Advocates for applying Dropout after all BN layers

Chen 2019 - "Rethinking the Usage of Batch Normalization and Dropout in
the Training of Deep Neural Networks" -
https://arxiv.org/pdf/1905.05928.pdf

    * Places dropout right after each batch norm with a small p=0.05

    * Notes the "clasical" formulation is Conv-Norm-NoLI

    * Also advocates for Conv-NoLI-Norm-Drop but also uses the output
      of Conv (before the NoLi etc.) as the base residual values when
      using skip connections. (They refer to BN-Drop as IC) and phrase
      the ordering as ReLU-IC-Conv2D.

    * They claim their process is "more stable" but they never quantify
      it. Their curves are better than baseline, but maybe they just
      hyperoptimized until they got better curves.


There is SO discussion indicating that Szegedy now likes BN after the ReLU:

    * https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

"""


class ConvNormNd(common.Sequential):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        dim (int):
            dimensionality of the convolutional kernel (can be 0, 1, 2, or 3).

        in_channels (int):

        out_channels (int):

        kernel_size (int | Tuple):

        stride (int | Tuple):

        padding (int | Tuple):

        dilation (int | Tuple):

        groups (int):

        bias (bool):

        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.

        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

        standardize_weights (bool, default=False):
            Implements weight standardization as described in Qiao 2020 -
            "Micro-Batch Training with Batch-Channel Normalization and Weight
            Standardization"- https://arxiv.org/pdf/1903.10520.pdf

    Example:
        >>> from .layers.conv_norm import ConvNormNd
        >>> self = ConvNormNd(dim=2, in_channels=16, out_channels=64,
        >>>                    kernel_size=3)
        >>> print(self)
        ConvNormNd(
          (conv): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1))
          (norm): BatchNorm2d(64, ...)
          (noli): ReLU(...)
        )

    Example:
        >>> from .layers.conv_norm import ConvNormNd
        >>> self = ConvNormNd(dim=0, in_channels=16, out_channels=64)
        >>> print(self)
        ConvNormNd(
          (conv): Conv0d(in_features=16, out_features=64, bias=True)
          (norm): BatchNorm1d(64, ...)
          (noli): ReLU(...)
        )
        >>> input_shape = (None, 16)
        >>> print(ub.urepr(self.output_shape_for(input_shape).hidden, nl=1))
        {
            'conv': (None, 64),
            'norm': (None, 64),
            'noli': (None, 64),
        }
        >>> print(ub.urepr(self.receptive_field_for()))
        {
            'crop': np.array([0., 0.], dtype=np.float64),
            'shape': np.array([1., 1.], dtype=np.float64),
            'stride': np.array([1., 1.], dtype=np.float64),
        }
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noli='relu',
                 norm='batch', standardize_weights=False):
        super(ConvNormNd, self).__init__()

        conv_cls = rectify.rectify_conv(dim)
        conv = conv_cls(in_channels, out_channels, kernel_size=kernel_size,
                        padding=padding, stride=stride, groups=groups,
                        bias=bias, dilation=dilation,
                        standardize_weights=standardize_weights)

        norm = rectify.rectify_normalizer(out_channels, norm, dim=dim)
        noli = rectify.rectify_nonlinearity(noli, dim=dim)

        self.add_module('conv', conv)
        if norm:
            self.add_module('norm', norm)
        if noli:
            self.add_module('noli', noli)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.standardize_weights = standardize_weights
        self._dim = dim

    def output_shape_for(self, input_shape):
        return OutputShapeFor.sequential(self, input_shape)


class ConvNorm1d(ConvNormNd):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.
        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

    Example:
        >>> from .layers.conv_norm import *
        >>> input_shape = [2, 3, 5]
        >>> self = ConvNorm1d(input_shape[1], 7, kernel_size=3)
        >>> OutputShapeFor(self)._check_consistency(input_shape)
        >>> self.output_shape_for(input_shape)
        (2, 7, 3)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noli='relu',
                 norm='batch', standardize_weights=False):
        super(ConvNorm1d, self).__init__(dim=1, in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, bias=bias,
                                         padding=padding, noli=noli, norm=norm,
                                         dilation=dilation, groups=groups,
                                         standardize_weights=standardize_weights)


class ConvNorm2d(ConvNormNd):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.
        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

    Example:
        >>> input_shape = [2, 3, 5, 7]
        >>> self = ConvNorm2d(input_shape[1], 11, kernel_size=3)
        >>> OutputShapeFor(self)._check_consistency(input_shape)
        >>> self.output_shape_for(input_shape)
        (2, 11, 3, 5)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noli='relu',
                 norm='batch', standardize_weights=False):
        super(ConvNorm2d, self).__init__(dim=2, in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, bias=bias,
                                         padding=padding, noli=noli, norm=norm,
                                         dilation=dilation, groups=groups,
                                         standardize_weights=standardize_weights)


class ConvNorm3d(ConvNormNd):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.
        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

    Example:
        >>> input_shape = [2, 3, 5, 7, 11]
        >>> self = ConvNorm3d(input_shape[1], 13, kernel_size=3)
        >>> OutputShapeFor(self)._check_consistency(input_shape)
        >>> self.output_shape_for(input_shape)
        (2, 13, 3, 5, 9)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, padding=0, noli='relu', norm='batch',
                 groups=1, standardize_weights=False):
        super(ConvNorm3d, self).__init__(dim=3, in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, bias=bias,
                                         padding=padding, noli=noli, norm=norm,
                                         groups=groups,
                                         standardize_weights=standardize_weights)


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.layers.conv_norm all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
