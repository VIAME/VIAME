"""
Implementation slightly modified from [1]_.


References:
    ..[1] https://github.com/lukemelas/EfficientNet-PyTorch
"""
import torch
from torch import nn
from torch.nn import functional as F
from netharn import layers

import re
import math
import collections
from functools import partial
from torch.utils import model_zoo


class Conv2dDynamicSamePadding(nn.Conv2d, layers.AnalyticModule):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(Conv2dDynamicSamePadding, self).__init__(in_channels,
                                                       out_channels,
                                                       kernel_size, stride, 0,
                                                       dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @classmethod
    def forsize(cls, image_size=None):
        """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
            Static padding is necessary for ONNX exporting of models. """
        if image_size is None:
            return Conv2dDynamicSamePadding
        else:
            return partial(Conv2dStaticSamePadding, image_size=image_size)

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from .models.efficientnet import *  # NOQA
            >>> from viame.arrows.pytorch.netharn import core as nh
            >>> kwargs = layers.AnalyticModule._analytic_shape_kw()
            >>> globals().update(kwargs)
            >>> inputs = (1, 3, 224, 224)
            >>> self = Conv2dDynamicSamePadding(2, 3, 5)
            >>> outputs = self.output_shape_for(inputs)
            >>> import ubelt as ub
            >>> print(nh.util.align(ub.repr2(outputs.hidden, nl=-1), ':'))
        """
        hidden = _Hidden()
        x = inputs
        ih, iw = _OutputFor.shape(x)[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            pad = [pad_w // 2, pad_w - pad_w // 2,
                   pad_h // 2, pad_h - pad_h // 2]
            x = hidden['dynamic_padding'] = _OutputFor(F.pad)(x, pad)

        weight = self.weight
        bias = self.bias is not None
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        groups = self.groups

        y = hidden['conv'] = _OutputFor(F.conv2d)(x, weight, bias, stride,
                                                  padding, dilation, groups)
        outputs = _Output.coerce(y, hidden)
        return outputs


class Conv2dStaticSamePadding(nn.Conv2d, layers.AnalyticModule):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super(Conv2dStaticSamePadding, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = int(math.ceil(ih / sh)), int(math.ceil(iw / sw))
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        self.image_size = image_size
        self._pad = (pad_h, pad_w)
        self._pad_w = pad_w
        self._pad_h = pad_w
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = layers.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from .models.efficientnet import *  # NOQA
            >>> from viame.arrows.pytorch.netharn import core as nh
            >>> kwargs = layers.AnalyticModule._analytic_shape_kw()
            >>> globals().update(kwargs)
            >>> inputs = (1, 3, 224, 224)
            >>> self = Conv2dStaticSamePadding(2, 3, 5, image_size=[512, 512])
            >>> outputs = self.output_shape_for(inputs)
            >>> import ubelt as ub
            >>> print(nh.util.align(ub.repr2(outputs.hidden, nl=-1), ':'))
        """
        hidden = _Hidden()
        x = inputs
        x = hidden['static_padding'] = _OutputFor(self.static_padding)(x)
        y = hidden['conv'] = _OutputFor(F.conv2d)(
            x, self.weight, self.bias is not None, self.stride, self.padding,
            self.dilation, self.groups)
        outputs = _Output.coerce(y, hidden)
        return outputs


##################
# Model definition
##################

class MBConvBlock(layers.AnalyticModule):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (BlockArgs): see :class:`Details`
        global_params (GlobalParam): see :class:`Details`

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = Conv2dDynamicSamePadding.forsize(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        # Note: it is important to set the weight decay to be very low for the
        # depthwise convolutions
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        # Note that the bn2 layer before the residual add, should be
        # initailized with gamma=0
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._bn2._residual_bn = True
        noli = global_params.noli
        self._noli = layers.rectify_nonlinearity(noli, dim=2)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._noli(self._bn0(self._expand_conv(inputs)))
        x = self._noli(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._noli(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = self.drop_connect(x, p=drop_connect_rate)
            x = x + inputs  # skip connection
        return x

    @classmethod
    def demo(MBConvBlock):
        layer_block_args, global_params = Details.build_efficientnet_params()
        block_args = layer_block_args[0]
        self = MBConvBlock(block_args, global_params)
        return self

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from .models.efficientnet import *  # NOQA
            >>> from viame.arrows.pytorch.netharn import core as nh
            >>> self = MBConvBlock.demo()
            >>> kwargs = self._analytic_shape_kw()
            >>> globals().update(kwargs)
            >>> input_shape = inputs = (1, 32, 224, 224)
            >>> outputs = self.output_shape_for(input_shape)
            >>> import ubelt as ub
            >>> print(nh.util.align(ub.repr2(outputs.hidden, nl=-1), ':'))
        """
        hidden = _Hidden()

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = hidden['expand_conv'] = _OutputFor(self._expand_conv)(inputs)
            x = hidden['_bn0'] = _OutputFor(self._bn0)(x)
            x = hidden['_noli0'] = _OutputFor(self._noli)(x)

        x = hidden['depthwise_conv'] = _OutputFor(self._depthwise_conv)(x)
        x = hidden['_bn1'] = _OutputFor(self._bn1)(x)
        x = hidden['_noli1'] = _OutputFor(self._noli)(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = hidden['_se_pool'] = _OutputFor(F.adaptive_avg_pool2d)(x, 1)
            x_squeezed = hidden['_se_reduce'] =  _OutputFor(self._se_reduce)(x_squeezed)
            x_squeezed = hidden['_se_noli'] = _OutputFor(self._noli)(x_squeezed)
            x_squeezed = hidden['_se_expand'] = _OutputFor(self._se_expand)(x_squeezed)
            x_squeezed = hidden['_se_sigmoid'] = _OutputFor(torch.sigmoid)(x_squeezed)
            x = hidden['_se_mul'] = _OutputFor.mul(x_squeezed, x)

        x = hidden['_project'] = _OutputFor(self._project_conv)(x)
        x = hidden['_bn2'] = _OutputFor(self._bn2)(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            drop_connect_rate = kwargs.get('drop_connect_rate', 0)
            if drop_connect_rate:
                try:
                    x = self.drop_connect(x, p=drop_connect_rate)
                except Exception:
                    pass
                hidden['drop_connect'] = x

            # skip connection
            x = hidden['skip'] = _OutputFor.add(x, inputs)
        outputs = _Output.coerce(x, hidden)
        return outputs

    def drop_connect(self, inputs, p):
        """ Drop connect. """
        if not self.training:
            return inputs
        batch_size = inputs.shape[0]
        keep_prob = 1 - p
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output


class EfficientNet(layers.AnalyticModule):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (List[BlockArgs]): arguments for each block
        global_params (GlobalParams): shared between blocks

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> model = EfficientNet.from_name('efficientnet-b0')
        >>> print(model.number_of_parameters())
        >>> print(model.get_image_size())
        5288548
        224

        >>> model = EfficientNet.from_name('efficientnet-b8')
        >>> print(model.number_of_parameters())
        >>> print(model.get_image_size())
        87413142
        672
    """

    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientNet, self).__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Handle class specification
        import ndsampler
        import ubelt as ub
        classes = self._global_params.classes
        if classes is None:
            classes = self._global_params.num_classes
        self.classes = ndsampler.CategoryTree.coerce(classes)

        keys = self._global_params._fields
        vals = list(self._global_params)
        tmp = ub.dzip(keys, vals, cls=ub.odict)
        tmp['num_classes'] = len(self.classes)
        tmp['classes'] = self.classes.__json__()
        self._global_params = type(global_params)(**tmp)

        self.image_size = self._global_params._asdict()['image_size']

        # import ubelt as ub
        # print(ub.repr2(self._global_params._asdict(), nl=-4))
        # print(ub.repr2(self._global_params._asdict()))

        self._initkw = {
            'blocks_args': self._blocks_args,
            'global_params': self._global_params,
        }

        self.model_name = None

        # Get static or dynamic convolution depending on image size
        Conv2d = Conv2dDynamicSamePadding.forsize(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = self.round_filters(32)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        multiplier = global_params.depth_coefficient

        def round_repeats(repeats, multiplier):
            """ Round number of filters based on depth multiplier. """
            if not multiplier:
                return repeats
            return int(math.ceil(multiplier * repeats))

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=self.round_filters(block_args.input_filters),
                output_filters=self.round_filters(block_args.output_filters),
                num_repeat=round_repeats(block_args.num_repeat, multiplier)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = self.round_filters(1280)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        noli = global_params.noli
        self._noli = layers.rectify_nonlinearity(noli, dim=2)

    def round_filters(self, filters):
        """ Calculate and round number of filters based on depth multiplier. """
        multiplier = self._global_params.width_coefficient
        if not multiplier:
            return filters
        divisor = self._global_params.depth_divisor
        min_depth = self._global_params.min_depth
        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
            new_filters += divisor
        return int(new_filters)

    def extract_features(self, inputs):
        """
        Returns output of the final convolution layer

        Note that predefined arches downsample by a factor of about 32x

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from .models.efficientnet import *  # NOQA
            >>> self = EfficientNet.from_name('efficientnet-b0')
            >>> self = self.train(False)
            >>> inputs = torch.rand(1, 3, 32, 32)
            >>> x = self.extract_features(inputs)
        """
        # Stem
        x = self._conv_stem(inputs)
        x = self._noli(self._bn0(x))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._noli(self._bn1(self._conv_head(x)))

        return x

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from viame.arrows.pytorch.netharn import core as nh
            >>> from .models.efficientnet import *  # NOQA
            >>> self = EfficientNet.from_name('efficientnet-b0')
            >>> kwargs = self._analytic_shape_kw()
            >>> globals().update(kwargs)
            >>> inputs = (1, 3, 224, 224)
            >>> inputs = (1, 3, 32, 32)
            >>> outputs = self.output_shape_for(inputs)
            >>> import ubelt as ub
            >>> print(nh.util.align(ub.repr2(outputs.hidden.shallow(1), nl=-1), ':'))

            >>> print(nh.util.align(ub.repr2(outputs.hidden['block_0'].shallow(2), nl=-1), ':'))
            >>> print(nh.util.align(ub.repr2(outputs.hidden['block_1'].shallow(2), nl=-1), ':'))
            >>> print(nh.util.align(ub.repr2(outputs.hidden['block_2'].shallow(2), nl=-1), ':'))
            >>> print(nh.util.align(ub.repr2(outputs.hidden['block_3'].shallow(2), nl=-1), ':'))
            >>> print(nh.util.align(ub.repr2(outputs.hidden['block_14'].shallow(2), nl=-1), ':'))
            >>> print(nh.util.align(ub.repr2(outputs.hidden['block_15'].shallow(2), nl=-1), ':'))

            >>> self = EfficientNet.from_name('efficientnet-b7')
            >>> print('self.image_size = {!r}'.format(self.image_size))
            >>> self = EfficientNet.from_name('efficientnet-b6')
            >>> print('self.image_size = {!r}'.format(self.image_size))
            >>> self = EfficientNet.from_name('efficientnet-b3')
            >>> print('self.image_size = {!r}'.format(self.image_size))
            >>> self = EfficientNet.from_name('efficientnet-b2')
            >>> print('self.image_size = {!r}'.format(self.image_size))
            >>> self = EfficientNet.from_name('efficientnet-b1')
            >>> print('self.image_size = {!r}'.format(self.image_size))
            >>> self = EfficientNet.from_name('efficientnet-b0')
            >>> print('self.image_size = {!r}'.format(self.image_size))

            >>> inputs = (1, 3, 224, 224)
            >>> self = EfficientNet.from_name('efficientnet-b7')
            >>> print('self.image_size = {!r}'.format(self.image_size))
            >>> outputs = self.output_shape_for(inputs)
            >>> print(nh.util.align(ub.repr2(outputs.hidden.shallow(1), nl=-1), ':'))

            for name, layer in nh.util.trainable_layers(self, names=1):
                if hasattr(layer, 'image_size'):
                    print('name = {!r}'.format(name))
                    print('layer = {!r}'.format(layer))
                    print('layer.image_size = {!r}'.format(layer.image_size))

            >>> inputs = (1, 3, 224, 224)
            >>> self = EfficientNet.from_name('efficientnet-b0')
            >>> outputs = self.output_shape_for(inputs)
            >>> print(nh.util.align(ub.repr2(outputs.hidden.shallow(1), nl=-1), ':'))

            for name, layer in nh.util.trainable_layers(self, names=1):
                if hasattr(layer, 'image_size'):
                    print('name = {!r}'.format(name))
                    print('layer = {!r}'.format(layer))
                    print('layer.image_size = {!r}'.format(layer.image_size))
        """
        hidden = _Hidden()

        # NEEDS MORE BACKEND WORK

        bs = _OutputFor.shape(inputs)[0]

        x = inputs
        x = hidden['_conv_stem'] = _OutputFor(self._conv_stem)(x)
        x = hidden['_noli1'] = _OutputFor(self._noli)(x)

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = hidden['block_{}'.format(idx)] = _OutputFor(block)(
                x, drop_connect_rate=drop_connect_rate)

        x = hidden['_noli2'] = _OutputFor(self._noli)(x)

        # Pooling and final linear layer
        x = _OutputFor(self._avg_pooling)(x)
        x = _OutputFor.view(x, bs, -1)
        x = _OutputFor(self._dropout)(x)
        x = _OutputFor(self._fc)(x)
        outputs = _Output.coerce(x, hidden)
        return outputs

    def forward(self, inputs):
        """
        Calls extract_features to extract features,
        applies final linear layer, and returns logits.
        """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    # TODO: Analytic forward

    @classmethod
    def from_params(cls, width, depth, size, dropout, **override_params):
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = Details.build_efficientnet_params(
            width_coefficient=width, depth_coefficient=depth,
            dropout_rate=dropout, image_size=size)
        global_params = global_params._replace(**override_params)

    @classmethod
    def from_name(EfficientNet, model_name, override_params=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> model_name = 'efficientnet-b0'
            >>> override_params = None
        """
        Details._check_model_name_is_valid(model_name)
        blocks_args, global_params = Details._get_model_params(model_name, override_params)
        self = EfficientNet(blocks_args, global_params)
        self.model_name = model_name
        return self

    @classmethod
    def from_pretrained(EfficientNet, model_name, advprop=False,
                        override_params=None, in_channels=3):
        """
        Initialize the model from a pretrained state

        Example:
            >>> # xdoctest: +REQUIRES(--download)
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from .models.efficientnet import *  # NOQA
            >>> model = EfficientNet.from_pretrained('efficientnet-b0')
            >>> inputs = torch.rand(1, 3, 224, 224)
            >>> outputs = model.forward(inputs)

            >>> from .models.efficientnet import *  # NOQA
            >>> model = EfficientNet.from_pretrained('efficientnet-b0', override_params={'noli': 'mish'}, advprop=True)
            >>> model = EfficientNet.from_pretrained('efficientnet-b1', override_params={'noli': 'mish'}, advprop=True)
            >>> model = EfficientNet.from_pretrained('efficientnet-b2', override_params={'noli': 'mish'}, advprop=True)
            >>> model = EfficientNet.from_pretrained('efficientnet-b3', override_params={'noli': 'mish'}, advprop=True)
            >>> model = EfficientNet.from_pretrained('efficientnet-b4', override_params={'noli': 'mish'}, advprop=True)
            >>> model = EfficientNet.from_pretrained('efficientnet-b5', override_params={'noli': 'mish'}, advprop=True)
            >>> model = EfficientNet.from_pretrained('efficientnet-b6', override_params={'noli': 'mish'}, advprop=True)
            >>> model = EfficientNet.from_pretrained('efficientnet-b7', override_params={'noli': 'mish'}, advprop=True)
        """
        if override_params is None:
            override_params = {}
        self = EfficientNet.from_name(model_name, override_params=override_params)
        num_classes = len(self.classes)
        Details.load_pretrained_weights(
            self, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = Conv2dDynamicSamePadding.forsize(image_size=self._global_params.image_size)
            out_channels = self.round_filters(32)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return self

    def get_image_size(self):
        if self.model_name is not None:
            return Details.get_image_size(self.model_name)
        else:
            raise NotImplementedError('only know size for predefined models')


class Details(object):
    """
    Definition of models

    Helpers functions for loading model params
    """

    # Parameters for the entire model (stem, all blocks, and head)
    # url_map = {
    #     'efficientnet-b0': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b0-355c32eb.pth',
    #     'efficientnet-b1': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b1-f1951068.pth',
    #     'efficientnet-b2': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b2-8bb594d6.pth',
    #     'efficientnet-b3': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b3-5fb5a3c3.pth',
    #     'efficientnet-b4': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b4-6ed6700e.pth',
    #     'efficientnet-b5': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b5-b6417697.pth',

    #     # 'efficientnet-b1': 'https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1',
    #     # 'efficientnet-b2': 'https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1',
    #     # 'efficientnet-b3': 'https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1',
    #     # 'efficientnet-b4': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',
    #     # 'efficientnet-b5': 'https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1',

    #     'efficientnet-b6': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b6-c76e70fd.pth',
    #     'efficientnet-b7': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b7-dcc49843.pth',
    # }

    # url_map_advprop = {
    #     'efficientnet-b0': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b0-b64d5a18.pth',
    #     'efficientnet-b1': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b1-0f3ce85a.pth',
    #     'efficientnet-b2': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b2-6e9d97e5.pth',
    #     'efficientnet-b3': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b3-cdd7c0f4.pth',
    #     'efficientnet-b4': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b4-44fb3a87.pth',
    #     'efficientnet-b5': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b5-86493f6b.pth',
    #     'efficientnet-b6': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b6-ac80338e.pth',
    #     'efficientnet-b7': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b7-4652b6dd.pth',
    #     'efficientnet-b8': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b8-22a8fe65.pth',
    # }

    url_map = {
        'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
        'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
        'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
        'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
        'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
        'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
        'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
        'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
    }

    url_map_advprop = {
        'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',
        'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',
        'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',
        'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',
        'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth',
        'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth',
        'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth',
        'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth',
        'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth',
    }

    @classmethod
    def get_image_size(cls, model_name):
        Details._check_model_name_is_valid(model_name)
        _, _, res, _ = Details._named_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b{}'.format(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

    @classmethod
    def _get_model_params(cls, model_name, override_params=None):
        """
        Get the block args and global params for a given model

        Example:
            Details._get_model_params('efficientnet-b0')
        """
        if model_name.startswith('efficientnet'):
            w, d, s, p = Details._named_params(model_name)
            # note: all models have drop connect rate = 0.2
            blocks_args, global_params = Details.build_efficientnet_params(
                width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
        else:
            raise NotImplementedError('model name is not pre-defined: %s' % model_name)
        if override_params:
            # ValueError will be raised here if override_params has fields not included in global_params.
            global_params = global_params._replace(**override_params)
        return blocks_args, global_params

    @classmethod
    def _named_params(cls, model_name):
        """ Map EfficientNet model name to parameter coefficients. """
        params_dict = {
            # Coefficients:   width,depth,res,dropout
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
            'efficientnet-b8': (2.2, 3.6, 672, 0.5),
            'efficientnet-l2': (4.3, 5.3, 800, 0.5),
        }
        return params_dict[model_name]

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.
        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(Details._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return Details.BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    # Parameters for an individual model block
    BlockArgs = collections.namedtuple('BlockArgs', [
        'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
        'expand_ratio', 'id_skip', 'stride', 'se_ratio'])
    BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.
        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(Details._encode_block_string(block))
        return block_strings

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    GlobalParams = collections.namedtuple('GlobalParams', [
        'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
        'num_classes', 'width_coefficient', 'depth_coefficient',
        'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size',
        'classes', 'noli'])

    # Change namedtuple defaults
    GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

    @staticmethod
    def build_efficientnet_params(width_coefficient=None,
                                  depth_coefficient=None, dropout_rate=0.2,
                                  drop_connect_rate=0.2, image_size=None,
                                  num_classes=1000):
        """
        Creates a efficientnet parameters

        Example:
            Details.build_efficientnet_params(None, None, image_size=512)
        """

        blocks_args = [
            'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
            'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
            'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
            'r1_k3_s11_e6_i192_o320_se0.25',
        ]
        blocks_args = Details.decode(blocks_args)

        global_params = Details.GlobalParams(
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            dropout_rate=dropout_rate,
            drop_connect_rate=drop_connect_rate,
            num_classes=num_classes,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            depth_divisor=8,
            min_depth=None,
            image_size=image_size,
            classes=None,
            noli='swish'
        )
        return blocks_args, global_params

    @staticmethod
    def load_pretrained_weights(model, model_name, load_fc=True, advprop=False):
        """ Loads pretrained weights, and downloads if loading for the first time. """
        # AutoAugment or Advprop (different preprocessing)
        url_map_ = Details.url_map_advprop if advprop else Details.url_map
        state_dict = model_zoo.load_url(url_map_[model_name])
        if load_fc:
            model.load_state_dict(state_dict)
        else:
            state_dict.pop('_fc.weight')
            state_dict.pop('_fc.bias')
            res = model.load_state_dict(state_dict, strict=False)
            assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), (
                'issue loading pretrained weights')
        print('Loaded pretrained weights for {}'.format(model_name))
