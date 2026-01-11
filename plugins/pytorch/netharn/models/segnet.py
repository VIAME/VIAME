"""
Implementation of the "Segnet" network toplogy from [2].  Adapated from [1].

References:
    [1] https://github.com/meetshah1995/pytorch-semseg
    [2] https://arxiv.org/abs/1511.00561
"""
import torch.nn as nn
from viame.pytorch.netharn import layers

__all__ = ['Segnet']


class SegnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetDown2, self).__init__()
        self.conv1 = layers.ConvNorm2d(in_size, out_size, 3, 1, 1,
                                       norm='batch', noli='relu')
        self.conv2 = layers.ConvNorm2d(out_size, out_size, 3, 1, 1,
                                       norm='batch', noli='relu')
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetDown3, self).__init__()
        self.conv1 = layers.ConvNorm2d(in_size, out_size, 3, 1, 1,
                                       norm='batch', noli='relu')
        self.conv2 = layers.ConvNorm2d(out_size, out_size, 3, 1, 1,
                                       norm='batch', noli='relu')
        self.conv3 = layers.ConvNorm2d(out_size, out_size, 3, 1, 1,
                                       norm='batch', noli='relu')
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = layers.ConvNorm2d(in_size, out_size, 3, 1, 1,
                                       norm='batch', noli='relu')
        self.conv2 = layers.ConvNorm2d(out_size, out_size, 3, 1, 1,
                                       norm='batch', noli='relu')

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices,
                              output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class SegnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = layers.ConvNorm2d(in_size, out_size, 3, 1, 1,
                                       norm='batch', noli='relu')
        self.conv2 = layers.ConvNorm2d(out_size, out_size, 3, 1, 1,
                                       norm='batch', noli='relu')
        self.conv3 = layers.ConvNorm2d(out_size, out_size, 3, 1, 1,
                                       norm='batch', noli='relu')

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices,
                              output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class Segnet(layers.Module):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> from .models.segnet import *  # NOQA
        >>> self = Segnet(classes=12, in_channels=5)
    """
    def __init__(self, classes, in_channels=3, is_unpooling=True):
        super(Segnet, self).__init__()

        import ndsampler
        self.classes = ndsampler.CategoryTree.coerce(classes)
        n_classes = len(self.classes)
        self.n_classes = n_classes

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = SegnetDown2(self.in_channels, 64)
        self.down2 = SegnetDown2(64, 128)
        self.down3 = SegnetDown3(128, 256)
        self.down4 = SegnetDown3(256, 512)
        self.down5 = SegnetDown3(512, 512)

        self.up5 = SegnetUp3(512, 512)
        self.up4 = SegnetUp3(512, 256)
        self.up3 = SegnetUp3(256, 128)
        self.up2 = SegnetUp2(128, 64)
        self.up1 = SegnetUp2(64, n_classes)

    def forward(self, inputs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from .models.segnet import *  # NOQA
            >>> import torch
            >>> B, C, W, H = (4, 5, 256, 256)
            >>> n_classes = 11
            >>> inputs = torch.rand(B, C, W, H)
            >>> labels = (torch.rand(B, W, H) * n_classes).long()
            >>> self = Segnet(in_channels=C, classes=n_classes)
            >>> outputs = self.forward(inputs)
        """

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        outputs = {
            'class_energy': up1,
        }
        return outputs

    def init_vgg16_params(self):
        import torchvision
        print('initializing using VGG params')
        vgg16 = torchvision.models.vgg16(pretrained=True)

        # ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        vgg_layers = [_layer for _layer in vgg16.features.children()
                      if isinstance(_layer, nn.Conv2d)]

        blocks = [self.down1,
                  self.down2,
                  self.down3,
                  self.down4,
                  self.down5]

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.conv,
                         conv_block.conv2.conv]
            else:
                units = [conv_block.conv1.conv,
                         conv_block.conv2.conv,
                         conv_block.conv3.conv]
            for _layer in units:
                if isinstance(_layer, nn.Conv2d):
                    merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        import torch
        with torch.no_grad():
            for internal, other in zip(merged_layers, vgg_layers):
                assert isinstance(other, nn.Conv2d) and isinstance(internal, nn.Conv2d)
                assert other.bias.size() == internal.bias.size()
                ob, oc, ow, oh = other.weight.shape
                ib, ic, iw, ih = internal.weight.shape
                assert ob == ib and ow == iw and oh == ih
                assert oc <= ic
                # hack, when inputs have more channels try pulling in only the
                # first parts
                if oc == ic:
                    internal.weight.data[:] = other.weight.data[:]
                else:
                    internal.weight.data[:, 0:oc, :, :] = other.weight.data[:]
                internal.bias[:] = other.bias.data[:]

    @classmethod
    def _initializer_cls(cls):
        """
        Specify a custom initializer class for a FitHarness

        Netharn doesn't have a great way to specify custom initialization
        without creating a special `Initializer` class. The `_initializer_cls`
        pattern is the current best way around this where a classmethod returns
        an initializer class that simply calls the model's initializer
        function. In the future we may come up with a better way of doing this.
        """
        from viame.pytorch import netharn as nh
        class InitVgg16Features(nh.initializers.Initializer):
            def forward(self, model):
                basic = nh.initializers.KaimingNormal()
                model = nh.XPU.raw(model)
                basic(model)
                model.init_vgg16_params()
        return InitVgg16Features
