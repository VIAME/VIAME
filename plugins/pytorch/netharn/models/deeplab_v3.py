"""
https://github.com/kazuto1011/deeplab-pytorch/blob/master/libs/models/deeplabv3.py

pip install torch-encoding
"""
import ubelt as ub  # NOQA
from collections import OrderedDict  # NOQA
import torch
import torch.nn as nn
import torch.nn.functional as F
from viame.pytorch.netharn import layers
# from .resnet import _ConvBnReLU, _ResLayer, _Stem

# from encoding.nn import SyncBatchNorm
# _BATCH_NORM = SyncBatchNorm
_BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4


class _ConvBnReLU(layers.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.

    Example:
        >>> from .models.deeplab_v3 import _ConvBnReLU
        >>> self = _ConvBnReLU(3, 5, 3, 1, 1, 1)
        >>> self.output_shape_for((1, 3, 32, 32))
        >>> self.receptive_field_for()
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(layers.AnalyticModule):
    """
    Bottleneck block of MSRA ResNet.

    Example:
        >>> from .models.deeplab_v3 import _Bottleneck
        >>> self = _Bottleneck(3, 3, 1, 1, 0)
        >>> self.output_shape_for((1, 3, 64, 64))
        >>> inputs = torch.rand(1, 3, 64, 64)
        >>> self.forward(inputs).shape
        >>> self.receptive_field_for().hidden.shallow(1)
        >>> self.receptive_field_for()
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = max(1, out_ch // _BOTTLENECK_EXPANSION)
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else ub.identity
        )

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> from .models.deeplab_v3 import *  # NOQA
            >>> from .models.deeplab_v3 import _Bottleneck
            >>> from viame.pytorch import netharn as nh
            >>> self = _Bottleneck(3, 16, 1, 1, 1)
            >>> kwargs = self._analytic_shape_kw()
            >>> globals().update(kwargs)
            >>> inputs = input_shape = (1, 3, 256, 256)
            >>> outputs = self._analytic_forward(inputs, **kwargs)
            >>> print('outputs = {}'.format(ub.repr2(outputs.hidden.shallow(1), nl=-1)))
        """
        x = inputs
        hidden = _Hidden()
        h = hidden['reduce'] = _OutputFor(self.reduce)(x)
        h = hidden['conv3x3'] = _OutputFor(self.conv3x3)(h)
        h = hidden['increase'] = _OutputFor(self.increase)(h)

        try:
            short = hidden['short'] = _OutputFor(self.shortcut)(h)
        except Exception:
            short = hidden['short'] = h

        try:
            short.shape
            h += short
        except Exception:
            pass
            # if isinstance(inputs, tuple):
            #     assert short == h, '{} {}'.format(short, h)

        out = hidden['relu'] = _OutputFor(F.relu)(h)
        try:
            out = _Output.coerce(out, hidden)
        except Exception:
            out = _Output(out)
        return out

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(layers.Sequential):
    """
    Residual layer with multi grids

    Example:
        >>> from .models.deeplab_v3 import *  # NOQA
        >>> from .models.deeplab_v3 import _ResLayer  # NOQA
        >>> self = _ResLayer(3, 3, 5, 1, 1)
        >>> self.output_shape_for((1, 3, 32, 32))
        >>> self.receptive_field_for()
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class _Stem(layers.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.

    Example:
        >>> from .models.deeplab_v3 import *  # NOQA
        >>> from .models.deeplab_v3 import _Stem  # NOQA
        >>> self = _Stem(3, 10).eval()
        >>> self.output_shape_for((1, 3, 32, 32))
        >>> self.receptive_field_for()
    """

    def __init__(self, in_ch, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(in_ch, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


class _Flatten(layers.AnalyticModule):
    """
    Example:
        >>> from .models.deeplab_v3 import *  # NOQA
        >>> from .models.deeplab_v3 import _Flatten  # NOQA
        >>> self = _Flatten().eval()
        >>> self.output_shape_for((1, 3, 32, 32))
    """
    def forward(self, x):
        return x.view(x.size(0), -1)

    def receptive_field_for(self, input_field):
        raise Exception('no rf for flatten')

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> from .models.deeplab_v3 import *  # NOQA
            >>> from .models.deeplab_v3 import _Flatten
            >>> from viame.pytorch import netharn as nh
            >>> self = _Flatten().eval()
            >>> # ---
            >>> kwargs = self._analytic_shape_kw()
            >>> globals().update(kwargs)
            >>> inputs = input_shape = (1, 3, 256, 256)
            >>> outputs = self._analytic_forward(inputs, **kwargs)
            >>> print('outputs = {!r}'.format(outputs))
            >>> # ---
            >>> kwargs = self._analytic_forward_kw()
            >>> globals().update(kwargs)
            >>> inputs = torch.rand(*input_shape)
            >>> outputs = self._analytic_forward(inputs, **kwargs)
            >>> # ---
            >>> outputs = self._analytic_forward(inputs, **kwargs)
        """
        hidden = _Hidden()
        try:
            input_shape = inputs.shape
        except Exception:
            input_shape = inputs

        import numpy as np
        bsize = input_shape[0]
        dim = int(np.prod(input_shape[1:]))

        try:
            output = hidden['out'] = _OutputFor(inputs.view, bsize, dim)
        except Exception:
            output = (bsize, dim)

        try:
            output = _Output.coerce(output, hidden)
        except Exception:
            pass
        return output


class ResNet(layers.Sequential):
    """
    Example:
        >>> from .models.deeplab_v3 import *  # NOQA
        >>> self = ResNet(3, 1000, [1, 1, 1, 1])
        >>> self.output_shape_for((1, 3, 32, 32))
    """

    def __init__(self, in_ch, n_classes, n_blocks):
        super(ResNet, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(in_ch, ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 2, 1))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 2, 1))
        self.add_module("pool5", nn.AdaptiveAvgPool2d(1))

        flatten = layers.reshape.Reshape(None, -1)
        flatten = _Flatten()

        self.add_module("flatten", flatten)
        self.add_module("fc", nn.Linear(ch[5], n_classes))


class _ImagePool(layers.AnalyticModule):
    """
    Example:
        >>> from .models.deeplab_v3 import _ImagePool  # NOQA
        >>> self = _ImagePool(3, 5).eval()
        >>> self.output_shape_for((1, 3, 32, 32))
    """
    def __init__(self, in_ch, out_ch):
        super(_ImagePool, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> from .models.deeplab_v3 import *  # NOQA
            >>> from .models.deeplab_v3 import _ImagePool
            >>> from viame.pytorch import netharn as nh
            >>> self = _ImagePool(3, 5).eval()
            >>> # ---
            >>> kwargs = self._analytic_shape_kw()
            >>> globals().update(kwargs)
            >>> inputs = input_shape = (1, 3, 256, 256)
            >>> outputs = self._analytic_forward(inputs, **kwargs)
            >>> print('outputs = {!r}'.format(outputs))
            >>> # ---
            >>> kwargs = self._analytic_forward_kw()
            >>> globals().update(kwargs)
            >>> inputs = torch.rand(*input_shape)
            >>> outputs = self._analytic_forward(inputs, **kwargs)
            >>> # ---
            >>> outputs = self._analytic_forward(inputs, **kwargs)
        """
        hidden = _Hidden()

        try:
            _, _, H, W = inputs.shape
        except Exception:
            _, _, H, W = inputs

        h = hidden['pool'] = _OutputFor(self.pool)(inputs)
        h = hidden['conv'] = _OutputFor(self.conv)(h)
        h = hidden['interpolate'] = _OutputFor(F.interpolate)(
            h, size=(H, W), mode="bilinear", align_corners=False)

        # output = hidden['out'] = _OutputFor(inputs.view, bsize, dim)
        try:
            output = _Output.coerce(h, hidden)
        except Exception:
            output = h
        return output


class _ASPP(layers.AnalyticModule):
    """
    Atrous spatial pyramid pooling with image-level feature

    Examples:
        >>> from .models.deeplab_v3 import _ASPP
        >>> self = _ASPP(3, 5, rates=[1, 2]).eval()
        >>> self.output_shape_for((1, 3, 32, 32))
        >>> self.forward(torch.rand(1, 3, 32, 32)).shape
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> from .models.deeplab_v3 import *  # NOQA
            >>> from .models.deeplab_v3 import _ASPP
            >>> from viame.pytorch import netharn as nh
            >>> self = _ASPP(3, 5, rates=[1, 2]).eval()
            >>> # ---
            >>> kwargs = self._analytic_shape_kw()
            >>> globals().update(kwargs)
            >>> inputs = input_shape = (1, 3, 256, 256)
            >>> outputs = self._analytic_forward(inputs, **kwargs)
            >>> print('outputs = {!r}'.format(outputs))
            >>> # ---
            >>> kwargs = self._analytic_forward_kw()
            >>> globals().update(kwargs)
            >>> inputs = torch.rand(*input_shape)
            >>> outputs = self._analytic_forward(inputs, **kwargs)
            >>> # ---
            >>> outputs = self._analytic_forward(inputs, **kwargs)
        """
        hidden = _Hidden()
        parts = []
        for i, stage in enumerate(self.stages.children()):
            p = hidden['stage_{}'.format(i)] = _OutputFor(stage)(inputs)
            parts.append(p)
        out = hidden['cat'] = _OutputFor(torch.cat)(parts, dim=1)
        try:
            out = _Output.coerce(out, hidden)
        except Exception:
            out = _Output(out)
        return out


class DeepLabV3(layers.Sequential):
    """
    DeepLab v3: Dilated ResNet with multi-grid + improved ASPP

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> from .models.deeplab_v3 import *  # NOQA
        >>> self = DeepLabV3(classes=21).eval()
        >>> ####
        >>> input_shape = (1, 3, 513, 513)
        >>> output_shape = self.output_shape_for(input_shape)
        >>> print('output_shape = {!r}'.format(output_shape))
        >>> #print(ub.repr2(output_shape.hidden.shallow(3), nl=-1))
        >>> print(ub.repr2(output_shape.hidden.shallow(2), nl=-1))
        >>> #print(ub.repr2(output_shape.hidden, nl=-1))
        >>> ####
        >>> image = torch.randn(*input_shape).to(ub.peek(self.devices()))
        >>> #print('self = {}'.format(self))
        >>> print("input:", image.shape)
        >>> print("output:", self(image).shape)
    """

    def __init__(self,
                 classes,
                 in_channels=3,
                 n_blocks=[3, 4, 23, 3],
                 atrous_rates=[6, 12, 18],
                 multi_grids=[1, 2, 4],
                 output_stride=8):
        super(DeepLabV3, self).__init__()

        try:
            import ndsampler
            self.classes = ndsampler.CategoryTree.coerce(classes)
            self.n_classes = n_classes = len(classes)
        except Exception:
            self.n_classes = n_classes = classes

        # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        self.in_channels = in_channels

        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(in_channels, ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0]))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1]))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2]))
        self.add_module(
            "layer5", _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        )
        self.add_module("aspp", _ASPP(ch[5], 256, atrous_rates))
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))
        self.add_module("fc2", nn.Conv2d(256, n_classes, kernel_size=1))


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m netharn.models.deeplab_v3
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
