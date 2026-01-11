# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import functools
import math
import torch
import torch.nn.functional as F

# __all__ = ['UNet']


class UNetConvNd(nn.Module):
    """
    Example:
        >>> from .models.unet import UNetConvNd  # NOQA
        >>> self = UNetConvNd(1, 1, False, dim=2)
        >>> inputs = torch.rand(1, 1, 8, 8)
        >>> self.forward(inputs)

    Example:
        >>> from .models.unet import UNetConvNd  # NOQA
        >>> self = UNetConvNd(1, 1, False, dim=3)
        >>> inputs = torch.rand(1, 1, 1, 8, 8)
        >>> self.forward(inputs)
    """
    def __init__(self, in_size, out_size, is_batchnorm, nonlinearity='relu', dim=2):
        # from viame.pytorch import netharn as nh
        from .layers import rectify
        super(UNetConvNd, self).__init__()

        if nonlinearity == 'relu':
            nonlinearity = functools.partial(nn.ReLU, inplace=False)
        elif nonlinearity == 'leaky_relu':
            nonlinearity = functools.partial(nn.LeakyReLU, inplace=False)

        ConvNd = rectify.rectify_conv(dim=dim)

        if dim == 2:
            kernel_size = (3, 3)
        elif dim == 3:
            kernel_size = (1, 3, 3)
        else:
            raise NotImplementedError

        conv_1 = ConvNd(in_size, out_size, kernel_size=kernel_size, stride=1,
                        padding=0)
        conv_2 = ConvNd(out_size, out_size, kernel_size=kernel_size, stride=1,
                        padding=0)

        if is_batchnorm:
            self.conv1 = nn.Sequential(conv_1, nn.BatchNorm2d(out_size),
                                       nonlinearity(),)
            self.conv2 = nn.Sequential(conv_2, nn.BatchNorm2d(out_size),
                                       nonlinearity(),)
        else:
            self.conv1 = nn.Sequential(conv_1, nonlinearity(),)
            self.conv2 = nn.Sequential(conv_2, nonlinearity(),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

    def output_shape_for(self, input_shape, math=math):
        from .analytic.output_shape_for import HiddenShapes, OutputShape, OutputShapeFor
        hidden = HiddenShapes()
        hidden['conv1'] = shape = OutputShapeFor(self.conv1[0])(input_shape)
        hidden['conv2'] = shape = OutputShapeFor(self.conv2[0])(shape)
        output_shape = OutputShape.coerce(shape, hidden)
        return output_shape


# class PadToAgree(nn.Module):
#     def __init__(self):
#         super(PadToAgree, self).__init__()

#     def forward(self, inputs1, inputs2):
#         want_w, want_h = inputs2.size()[-2:]
#         have_w, have_h = inputs1.size()[-2:]

#         half_offw = (want_w - have_w) / 2
#         half_offh = (want_h - have_h) / 2
#         # padding = 2 * [offw // 2, offh // 2]

#         padding = [
#             # Padding starts from the final dimension and then move backwards.
#             math.floor(half_offh),
#             math.ceil(half_offh),
#             math.floor(half_offw),
#             math.ceil(half_offw),
#         ]
#         outputs1 = F.pad(inputs1, padding)
#         return outputs1

#     def output_shape_for(self, input_shape1, input_shape2):
#         N1, C1, W1, H1 = input_shape1
#         N2, C2, W2, H2 = input_shape2
#         output_shape = (N1, C1, W2, H2)
#         return output_shape


class PadToAgree(nn.Module):
    """
    Example:
        >>> from .models.unet import *  # NOQA
        >>> from .models.unet import PadToAgree  # NOQA
        >>> self = PadToAgree()
        >>> input_shape1 = (2, 3, 5, 6, 8)
        >>> input_shape2 = (2, 3, 5, 7, 11)
        >>> self.padding(input_shape1, input_shape2)
        >>> inputs1 = torch.rand(*input_shape1)
        >>> inputs2 = torch.rand(*input_shape2)
        >>> self(inputs1, inputs2)

    """
    def __init__(self):
        super(PadToAgree, self).__init__()

    def padding(self, input_shape1, input_shape2):
        """
        CommandLine:
            xdoctest -m ~/code/netharn/netharn/models/unet.py PadToAgree.padding

        Example:
            >>> from .models.unet import *  # NOQA
            >>> self = PadToAgree()
            >>> input_shape1 = (1, 32, 37, 52)
            >>> input_shape2 = (1, 32, 28, 44)
            >>> self.padding(input_shape1, input_shape2)
            (-4, -4, -5, -4)
        """

        if len(input_shape1) == 4:
            # padding = 2 * [offw // 2, offh // 2]
            have_w, have_h = input_shape1[-2:]
            want_w, want_h = input_shape2[-2:]
            half_offw = (want_w - have_w) / 2
            half_offh = (want_h - have_h) / 2
            padding = tuple([
                # Padding starts from the final dimension and then move backwards.
                int(math.floor(half_offh)),
                int(math.ceil(half_offh)),
                int(math.floor(half_offw)),
                int(math.ceil(half_offw)),
            ])
        elif len(input_shape1) == 5:
            # padding = 2 * [offw // 2, offh // 2]
            have_t, have_w, have_h = input_shape1[-3:]
            want_t, want_w, want_h = input_shape2[-3:]
            half_offw = (want_w - have_w) / 2
            half_offh = (want_h - have_h) / 2
            half_offt = (want_t - have_t) / 2
            padding = tuple([
                # Padding starts from the final dimension and then move backwards.
                int(math.floor(half_offw)),
                int(math.ceil(half_offw)),
                int(math.floor(half_offh)),
                int(math.ceil(half_offh)),
                int(math.floor(half_offt)),
                int(math.ceil(half_offt)),
            ])
        else:
            raise NotImplementedError

        return padding

    def forward(self, inputs1, inputs2):
        input_shape1 = inputs1.shape
        input_shape2 = inputs2.shape
        padding = self.padding(input_shape1, input_shape2)

        outputs1 = F.pad(inputs1, padding)
        return outputs1

    def output_shape_for(self, input_shape1, input_shape2):
        N1, C1, *DIMS1 = input_shape1
        N2, C2, *DIMS2 = input_shape2
        output_shape = (N1, C1, *DIMS2)
        return output_shape


class UNetUp(nn.Module):
    """
    Example:
        >>> from .models.unet import UNetUp
        >>> self = UNetUp(6, 3, is_deconv=False, dim=3)
        >>> input_shape1 = (B, C, T, H, W) = 1, 3, 2, 8, 8
        >>> input_shape2 = (B, C, T, H *  2, W * 2)
        >>> inputs1 = torch.rand(*input_shape1)
        >>> inputs2 = torch.rand(*input_shape2)
        >>> out = self.forward(inputs1, inputs2)
        >>> out.shape

    """
    def __init__(self, in_size, out_size, is_deconv=True, nonlinearity='relu',
                 dim=2):
        super(UNetUp, self).__init__()
        if dim == 2:
            kernel_size = 2
            stride = 2
            ConvTransposeNd = nn.ConvTranspose2d
        elif dim == 3:
            ConvTransposeNd = nn.ConvTranspose3d
            kernel_size = (2, 2, 2)
            stride = (2, 2, 1)
        else:
            raise NotImplementedError
        if is_deconv:
            self.up = ConvTransposeNd(in_size, out_size, kernel_size=kernel_size, stride=stride)
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.pad = PadToAgree()
        self.dim = dim
        self.conv = UNetConvNd(in_size, out_size, is_batchnorm=False,
                               nonlinearity=nonlinearity, dim=dim)

    def output_shape_for(self, input1_shape, input2_shape):
        """
        Example:
            >>> import ubelt as ub
            >>> from .models.unet import *  # NOQA
            >>> self = UNetUp(256, 128)
            >>> input1_shape = [4, 128, 24, 24]
            >>> input2_shape = [4, 256, 8, 8]
            >>> output_shape = self.output_shape_for(input1_shape, input2_shape)
            >>> print('hidden_shapes = ' + ub.repr2(output_shape.hidden.shallow(100), nl=-1))
            ...
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (4, 128, 12, 12)
            >>> inputs1 = (torch.rand(input1_shape))
            >>> inputs2 = (torch.rand(input2_shape))
            >>> out = self.forward(inputs1, inputs2)
            >>> assert out.shape == output_shape
        """
        from .analytic.output_shape_for import HiddenShapes, OutputShape, OutputShapeFor
        hidden = HiddenShapes()
        hidden['up'] = output2_shape = OutputShapeFor(self.up)(input2_shape)
        hidden['pad'] = output1_shape = OutputShapeFor(self.pad)(input1_shape, output2_shape)
        hidden['cat'] = cat_shape = OutputShapeFor(torch.cat)([output1_shape, output2_shape], 1)
        hidden['conv'] = final  = OutputShapeFor(self.conv)(cat_shape)
        output_shape = OutputShape.coerce(final, hidden)
        return output_shape

    def forward(self, inputs1, inputs2):
        """
        inputs1 = (37 x 52)
        inputs2 = (14 x 22) -> up -> (28 x 44)
        self.up = self.up_concat4.up

        want_w, want_h = (28, 44)  # outputs2
        have_w, have_h = (37, 52)  # inputs1

        offw = -9
        offh = -8

        padding [-5, -4, -4, -4]
        """
        outputs2 = self.up(inputs2)
        outputs1 = self.pad(inputs1, outputs2)
        outputs_cat = torch.cat([outputs1, outputs2], 1)
        out = self.conv(outputs_cat)
        return out

from viame.pytorch import netharn as nh  # NOQA


class UNet(nh.layers.Module):
    """
    Note input shapes should be a power of 2.

    In this case there will be a ~188 pixel difference between input and output
    dims, so the input should be mirrored with

    Example:
        >>> # xdoctest: +REQUIRES(--slow,module:kwcoco)
        >>> import numpy as np
        >>> B, C, W, H = (4, 3, 256, 256)
        >>> B, C, W, H = (4, 3, 572, 572)
        >>> n_classes = 11
        >>> inputs = (torch.rand(B, C, W, H))
        >>> labels = ((torch.rand(B, W, H) * n_classes).long())
        >>> self = UNet(in_channels=C, classes=n_classes)
        >>> outputs = self.forward(inputs)['class_energy']
        >>> print('inputs.size() = {!r}'.format(inputs.size()))
        >>> print('outputs.size() = {!r}'.format(outputs.size()))
        >>> print(np.array(inputs.size()) - np.array(outputs.size()))

    Example:
        >>> # xdoctest: +REQUIRES(--slow,module:kwcoco)
        >>> import numpy as np
        >>> B, C, W, H = (4, 5, 480, 360)
        >>> n_classes = 11
        >>> inputs = (torch.rand(B, C, W, H))
        >>> labels = ((torch.rand(B, W, H) * n_classes).long())
        >>> self = UNet(in_channels=C, classes=n_classes)
        >>> outputs = self.forward(inputs)['class_energy']
        >>> print('inputs.size() = {!r}'.format(inputs.size()))
        >>> print('outputs.size() = {!r}'.format(outputs.size()))
        >>> print(np.array(inputs.size()) - np.array(outputs.size()))

    Example:
        >>> # xdoctest: +REQUIRES(--slow,module:kwcoco)
        >>> from .models.unet import *  # NOQA
        >>> import numpy as np
        >>> B, C, T, W, H = (4, 3, 2, 480, 360)
        >>> n_classes = 11
        >>> inputs = (torch.rand(B, C, T, W, H))
        >>> labels = ((torch.rand(B, T, W, H) * n_classes).long())
        >>> self = UNet(in_channels=C, classes=n_classes, dim=3)
        >>> outputs = self.forward(inputs)['class_energy']
        >>> print('inputs.size() = {!r}'.format(inputs.size()))
        >>> print('outputs.size() = {!r}'.format(outputs.size()))
        >>> print(np.array(inputs.size()) - np.array(outputs.size()))
    """
    def __init__(self, feature_scale=4, classes=21, is_deconv=True,
                 in_channels=3, is_batchnorm=True, nonlinearity='relu', dim=2):
        super(UNet, self).__init__()
        import kwcoco
        from .layers import rectify

        MaxPoolNd = rectify.rectify_maxpool(dim=dim)
        ConvNd = rectify.rectify_conv(dim=dim)

        self.classes = kwcoco.CategoryTree.coerce(classes)
        n_classes = len(self.classes)

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.nonlinearity = nonlinearity

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x // self.feature_scale) for x in filters]

        # downsampling

        if dim == 2:
            kernel_size = (2, 2)
        elif dim == 3:
            kernel_size = (2, 2, 1)
        else:
            raise NotImplementedError

        self.conv1 = UNetConvNd(self.in_channels, filters[0], self.is_batchnorm, self.nonlinearity, dim=dim)
        self.maxpool1 = MaxPoolNd(kernel_size=kernel_size)

        self.conv2 = UNetConvNd(filters[0], filters[1], self.is_batchnorm, self.nonlinearity, dim=dim)
        self.maxpool2 = MaxPoolNd(kernel_size=kernel_size)

        self.conv3 = UNetConvNd(filters[1], filters[2], self.is_batchnorm, self.nonlinearity, dim=dim)
        self.maxpool3 = MaxPoolNd(kernel_size=kernel_size)

        self.conv4 = UNetConvNd(filters[2], filters[3], self.is_batchnorm, self.nonlinearity, dim=dim)
        self.maxpool4 = MaxPoolNd(kernel_size=kernel_size)

        self.center = UNetConvNd(filters[3], filters[4], self.is_batchnorm, self.nonlinearity, dim=dim)

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3], self.is_deconv, self.nonlinearity, dim=dim)
        self.up_concat3 = UNetUp(filters[3], filters[2], self.is_deconv, self.nonlinearity, dim=dim)
        self.up_concat2 = UNetUp(filters[2], filters[1], self.is_deconv, self.nonlinearity, dim=dim)
        self.up_concat1 = UNetUp(filters[1], filters[0], self.is_deconv, self.nonlinearity, dim=dim)

        # final conv (without any concat)
        self.final = ConvNd(filters[0], n_classes, 1)
        self._cache = {}

    def output_shape_for(self, input_shape):
        from .. import OutputShapeFor
        # N1, C1, W1, H1 = input_shape
        # output_shape = (N1, self.n_classes, W1, H1)
        shape = input_shape
        shape = conv1 = OutputShapeFor(self.conv1)(shape)
        shape = OutputShapeFor(self.maxpool1)(shape)

        shape = conv2 = OutputShapeFor(self.conv2)(shape)
        shape = OutputShapeFor(self.maxpool2)(shape)

        shape = conv3 = OutputShapeFor(self.conv3)(shape)
        shape = OutputShapeFor(self.maxpool3)(shape)

        shape = conv4 = OutputShapeFor(self.conv4)(shape)
        shape = OutputShapeFor(self.maxpool4)(shape)

        shape = OutputShapeFor(self.center)(shape)

        shape = OutputShapeFor(self.up_concat4)(conv4, shape)
        shape = OutputShapeFor(self.up_concat3)(conv3, shape)
        shape = OutputShapeFor(self.up_concat2)(conv2, shape)
        shape = OutputShapeFor(self.up_concat1)(conv1, shape)

        shape = OutputShapeFor(self.final)(shape)
        output_shape = shape
        return output_shape

    def raw_output_shape_for(self, input_shape):
        from .. import OutputShapeFor
        # output shape without fancy prepad mirrors and post crops
        shape = conv1 = OutputShapeFor(self.conv1)(input_shape)
        shape = OutputShapeFor(self.maxpool1)(shape)

        shape = conv2 = OutputShapeFor(self.conv2)(shape)
        shape = OutputShapeFor(self.maxpool2)(shape)

        shape = conv3 = OutputShapeFor(self.conv3)(shape)
        shape = OutputShapeFor(self.maxpool3)(shape)

        shape = conv4 = OutputShapeFor(self.conv4)(shape)
        shape = OutputShapeFor(self.maxpool4)(shape)

        shape = OutputShapeFor(self.center)(shape)

        shape = OutputShapeFor(self.up_concat4)(conv4, shape)
        shape = OutputShapeFor(self.up_concat3)(conv3, shape)
        shape = OutputShapeFor(self.up_concat2)(conv2, shape)
        shape = OutputShapeFor(self.up_concat1)(conv1, shape)

        shape = OutputShapeFor(self.final)(shape)
        output_shape = shape
        return output_shape

    def find_padding_and_crop_for(self, input_shape):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--slow, module:kwcoco)
            >>> B, C, W, H = (4, 3, 572, 572)
            >>> B, C, W, H = (4, 3, 372, 400)
            >>> n_classes = 11
            >>> input_shape = (B, C, W, H)
            >>> self = UNet(in_channels=C, classes=n_classes)
            >>> self.raw_output_shape_for(input_shape)
            >>> prepad, postcrop = self.find_padding_and_crop_for(input_shape)
        """
        B, rest = input_shape[0], input_shape[1:]
        # B, *rest = input_shape
        # change batch to something arbitrary because it doesnt matter
        B = 8
        input_shape = tuple([B]) + tuple(rest)
        if input_shape in self._cache:
            return self._cache[input_shape]

        import sympy as sym
        from .. import OutputShapeFor
        shape = input_shape

        raw_output = self.raw_output_shape_for(input_shape)
        assert raw_output[2] > 0, 'input is too small'
        assert raw_output[3] > 0, 'input is too small'

        input_shape_ = sym.symbols('N, C, W, H', integer=True, positive=True)
        orig = OutputShapeFor.math
        OutputShapeFor.math = sym
        shape = input_shape_
        # hack OutputShapeFor with sympy to do some symbolic math
        output_shape = self.raw_output_shape_for(shape)
        OutputShapeFor.math = orig

        W1, H1 = input_shape_[-2:]
        W2_raw = output_shape[2]
        H2_raw = output_shape[3]

        padw, padh = sym.symbols('padw, padh', integer=True, positive=True)

        def find_padding(D_in, D_out, pad_in, want):
            """
            Find a padding where
            want = numeric in dimension
            out_dimension = forward(in_dimension + padding)
            """
            D_out_pad = D_out.subs({D_in: D_in + pad_in})

            expr = D_out_pad
            target = want
            fixed = {D_in: want}
            solve_for = pad_in

            fixed_expr = expr.subs(fixed).simplify()
            def func(a1):
                expr_value = float(fixed_expr.subs({solve_for: a1}).evalf())
                return expr_value - target

            def integer_step_linear_zero(func):
                value = 0
                hi, lo = 10000, 0
                while lo <= hi:
                    guess = (lo + hi) // 2
                    result = func(guess)
                    if result < value:
                        lo = guess + 1
                    elif result > value:
                        hi = guess - 1
                    else:
                        break

                # force a positive padding
                while result < value:
                    guess += 1
                    result = func(guess)

                # always choose the lowest level value of guess
                i = guess
                i -= 1
                result = func(i)
                while result == value:
                    i -= 1
                    result = func(i)
                low_level = i + 1

                # always choose the lowest level value of guess
                j = guess
                j += 1
                result = func(j)
                while result == value:
                    j += 1
                    result = func(j)
                high_level = j - 1

                return low_level, high_level

            # The correct pad is non-unique when it exists
            low_level, high_level = integer_step_linear_zero(func)
            got = low_level
            deltaf = fixed_expr.subs({solve_for: got}).evalf() - target
            delta = math.ceil(deltaf)

            # We return how much you need to pad and how much you need to crop
            # in order to get an output-size = input-size. pads and crops ard
            # in total, so do the propper floor / ceiling.
            crop = delta
            pad = got
            return pad, crop

        want_w, want_h = input_shape[2:4]
        prepad_w, postcrop_w = find_padding(W1, W2_raw, padw, want_w)
        prepad_h, postcrop_h = find_padding(H1, H2_raw, padh, want_h)

        prepad = (prepad_w, prepad_h)
        postcrop = (postcrop_w, postcrop_h)

        self._cache[input_shape] = (prepad, postcrop)

        # import tqdm
        # print = tqdm.tqdm.write
        print('prepad = {!r}'.format(prepad))
        print('postcrop = {!r}'.format(postcrop))
        return prepad, postcrop

    def prepad(self, inputs):
        # do appropriate mirroring so final.size()[-2:] >= input.size()[:-2]
        input_shape = inputs.size()
        pad_wh, crop_wh = self.find_padding_and_crop_for(input_shape)
        padw, padh = pad_wh
        halfw, halfh = padw / 2, padh / 2
        padding = tuple([
            # Padding starts from the final dimension and then move backwards.
            int(math.floor(halfh)),
            int(math.ceil(halfh)),
            int(math.floor(halfw)),
            int(math.ceil(halfw)),
        ])
        mirrored = F.pad(inputs, padding, mode='reflect')
        return mirrored, crop_wh

    def postcrop(self, final, crop_wh):
        # do appropriate mirroring so final.size()[-2:] >= input.size()[:-2]
        w, h = crop_wh

        halfw, halfh = w / 2, h / 2
        # Padding starts from the final dimension and then move backwards.
        y1 = int(math.floor(halfh))
        y2 = int(final.size()[-1] - math.ceil(halfh))
        x1 = int(math.floor(halfw))
        x2 = int(final.size()[-2] - math.ceil(halfw))

        cropped = final[:, :, x1:x2, y1:y2]
        return cropped

    def raw_forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final

    def forward(self, inputs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwcoco)
            >>> # xdoctest: +REQUIRES(module:sympy)
            >>> import torch
            >>> B, C, W, H = (1, 1, 256, 256)
            >>> n_classes = 2
            >>> inputs = (torch.rand(B, C, W, H))
            >>> labels = ((torch.rand(B, W, H) * n_classes).long())
            >>> self = UNet(in_channels=C, classes=n_classes)
            >>> outputs = self.forward(inputs)
        """
        # Is there a way to miror so that we have enough input pixels?
        # so we can crop off extras after?
        # if isinstance(inputs, (tuple, list)):
        #     assert len(inputs) == 1
        #     inputs = inputs[0]

        mirrored = inputs
        mirrored, crop_wh = self.prepad(inputs)

        final = self.raw_forward(mirrored)

        cropped = self.postcrop(final, crop_wh)
        class_energy = cropped

        outputs = {
            'class_energy': class_energy
        }
        return outputs
