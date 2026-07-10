# -*- coding: utf-8 -*-
"""
Implementation taken from [1] and [2]

References:
    [1] https://raw.githubusercontent.com/warmspringwinds/pytorch-segmentation-detection/master/pytorch_segmentation_detection/models/psp.py
    [2] https://raw.githubusercontent.com/warmspringwinds/vision/eb6c13d3972662c55e752ce7a376ab26a1546fb5/torchvision/models/resnet.py
    https://paperswithcode.com/sota/semantic-segmentation-on-camvid
    https://github.com/warmspringwinds/pytorch-segmentation-detection/tree/master/pytorch_segmentation_detection/models
"""
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from viame.pytorch.netharn import layers


# __all__ = ['PSPNet_Resnet50_8s']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    kernel_size = np.asarray((3, 3))
    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2
    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)


class ResidualJoin(layers.Module):
    """
    Joins the main channel with a residual skip connection
    """

    def __init__(self, out_planes, inplace=True):
        super(ResidualJoin, self).__init__()
        self.inplace = inplace
        self.out_planes = out_planes

    def forward(self, main, skip):
        if self.inplace:
            out = main
            out += skip
        else:
            out = main + skip
        return out

    def output_shape_for(self, main_shape, skip_shape):
        """
        note: two inputs is not a standard yet
        """
        bsize, dims = main_shape[0], main_shape[2:]
        skip_bsize, skip_dims = skip_shape[0], skip_shape[2:]
        if bsize != skip_bsize:
            raise AssertionError('inconsistent bsize {} -vs- {}'.format(dims, skip_bsize))
        if dims != skip_dims:
            raise AssertionError('inconsistent dims {} -vs- {}'.format(dims, skip_dims))
        output_shape = tuple([bsize, self.out_planes] + list(dims))
        return output_shape

    def receptive_field_for(self, main_field, skip_field):
        """
        note: two inputs is not a standard yet
        """
        from viame.pytorch import netharn as nh
        # Handle skip connection
        # Unsure if it is ok for these values to not agree
        assert np.all(skip_field['crop'] == main_field['crop']), 'main_field={}, skip_field={}'.format(main_field, skip_field)
        assert np.all(skip_field['stride'] == main_field['stride']), 'main_field={}, skip_field={}'.format(main_field, skip_field)
        # It is definately true that maximum should be used to rectify
        # theoretical RF sizes, because RF-window of the smaller one (if they
        # are different) will be a subset of the larger one.
        field = nh.ReceptiveField({
            'shape': np.maximum(skip_field['shape'], main_field['shape']),
            'crop': np.maximum(skip_field['crop'], main_field['crop']),
            'stride': np.maximum(skip_field['stride'], main_field['stride']),
        })
        return field


class BasicBlock(layers.AnalyticModule):
    """
    Example:
        >>> from .models.psp import *  # NOQA
        >>> self = BasicBlock(8, 8)
        >>> inputs = torch.rand(2, 8, 7, 7)
        >>> out = self(inputs)
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        if downsample is None:
            downsample = layers.Identity()
            pass

        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.join = ResidualJoin(planes, inplace=True)
        self.stride = stride

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is None:
            residual = inputs
        else:
            residual = self.downsample(inputs)

        out = self.join(out, residual)
        out = self.relu(out)
        return out

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        analytic computation of forward / shape / receptive field.
        """
        hidden = _Hidden()
        out = inputs
        out = hidden['conv1'] = _OutputFor(self.conv1)(out)
        out = hidden['bn1'] = _OutputFor(self.bn1)(out)
        out = hidden['relu1'] = _OutputFor(self.relu)(out)

        out = hidden['conv2'] = _OutputFor(self.conv2)(out)
        out = hidden['bn2'] = _OutputFor(self.bn2)(out)

        if self.downsample is None:
            residual = inputs
        else:
            residual = hidden['residual'] = _OutputFor(self.downsample)(inputs)

        out = hidden['join'] = _OutputFor(self.join)(out, residual)
        out = hidden['relu3'] = _OutputFor(self.relu)(out)
        return _Output.coerce(out, hidden)


class Bottleneck(layers.AnalyticModule):
    """
    Example:
        >>> from .models.psp import *  # NOQA
        >>> self = Bottleneck(8, 2)
        >>> inputs = torch.rand(2, 8, 7, 7)
        >>> out = self(inputs)
        >>> self.output_shape_for((2, 8, 7, 7))
        >>> from viame.pytorch import netharn as nh
        >>> self.receptive_field_for()
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        if downsample is None:
            # downsample = layers.Identity()
            downsample = None

        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.join = ResidualJoin(planes * 4, inplace=True)
        self.stride = stride

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is None:
            residual = inputs
        else:
            residual = self.downsample(inputs)
        out = self.join(out, residual)

        out = self.relu(out)
        return out

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        analytic computation of forward / shape / receptive field.
        """
        hidden = _Hidden()
        out = inputs
        out = hidden['conv1'] = _OutputFor(self.conv1)(out)
        out = hidden['bn1'] = _OutputFor(self.bn1)(out)
        out = hidden['relu1'] = _OutputFor(self.relu)(out)

        out = hidden['conv2'] = _OutputFor(self.conv2)(out)
        out = hidden['bn2'] = _OutputFor(self.bn2)(out)
        out = hidden['relu2'] = _OutputFor(self.relu)(out)

        out = hidden['conv3'] = _OutputFor(self.conv3)(out)
        out = hidden['bn3'] = _OutputFor(self.bn3)(out)

        if self.downsample is None:
            residual = inputs
        else:
            residual = hidden['residual'] = _OutputFor(self.downsample)(inputs)

        out = hidden['join'] = _OutputFor(self.join)(out, residual)
        out = hidden['relu3'] = _OutputFor(self.relu)(out)
        return _Output.coerce(out, hidden)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, fully_conv=False,
                 remove_avg_pool_layer=False, output_stride=32,
                 additional_blocks=0, multi_grid=(1, 1, 1), in_channels=3):

        # Add additional variables to track
        # output stride. Necessary to achieve
        # specified output stride.
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1

        self.remove_avg_pool_layer = remove_avg_pool_layer

        self.inplanes = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, multi_grid=multi_grid)

        self.additional_blocks = additional_blocks

        if additional_blocks == 1:

            self.layer5 = self._make_layer(
                block, 512, layers[3], stride=2, multi_grid=multi_grid)

        if additional_blocks == 2:

            self.layer5 = self._make_layer(
                block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer6 = self._make_layer(
                block, 512, layers[3], stride=2, multi_grid=multi_grid)

        if additional_blocks == 3:

            self.layer5 = self._make_layer(
                block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer6 = self._make_layer(
                block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer7 = self._make_layer(
                block, 512, layers[3], stride=2, multi_grid=multi_grid)

        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
            # In the latest unstable torch 4.0 the tensor.copy_
            # method was changed and doesn't work as it used to be
            #self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, multi_grid=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:
                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride
            # We don't dilate 1x1 convolution.
            downsample = layers.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        parts = []

        dilation = (multi_grid[0] * self.current_dilation
                    if multi_grid else self.current_dilation)
        parts.append(
            block(self.inplanes, planes, stride, downsample,
                  dilation=dilation))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            dilation = multi_grid[i] * \
                self.current_dilation if multi_grid else self.current_dilation
            parts.append(block(self.inplanes, planes, dilation=dilation))

        return layers.Sequential(*parts)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.additional_blocks == 1:
            x = self.layer5(x)

        if self.additional_blocks == 2:
            x = self.layer5(x)
            x = self.layer6(x)

        if self.additional_blocks == 3:
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)

        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)

        if not self.fully_conv:
            x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        analytic computation of forward / shape / receptive field.
        """
        hidden = _Hidden()
        out = inputs
        # out = hidden['conv1'] = _OutputFor(self.conv1)(out)
        # out = hidden['bn1'] = _OutputFor(self.bn1)(out)
        # out = hidden['relu1'] = _OutputFor(self.relu)(out)

        # out = hidden['conv2'] = _OutputFor(self.conv2)(out)
        # out = hidden['bn2'] = _OutputFor(self.bn2)(out)
        # out = hidden['relu2'] = _OutputFor(self.relu)(out)

        # out = hidden['conv3'] = _OutputFor(self.conv3)(out)
        # out = hidden['bn3'] = _OutputFor(self.bn3)(out)

        # residual = hidden['residual'] = _OutputFor(self.downsample)(inputs)

        # out = hidden['join'] = _OutputFor(self.join)(out, residual)
        # out = hidden['relu3'] = _OutputFor(self.relu)(out)
        return _Output.coerce(out, hidden)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        if model.additional_blocks:
            model.load_state_dict(
                model_zoo.load_url(
                    model_urls['resnet18']),
                strict=False)
            return model
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if model.additional_blocks:
            model.load_state_dict(
                model_zoo.load_url(
                    model_urls['resnet34']),
                strict=False)
            return model
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    import torch_liberator
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if model.additional_blocks:
            torch_liberator.load_partial_state(
                model, model_zoo.load_url(model_urls['resnet50']), strict=False)
            return model
        torch_liberator.load_partial_state(
            model, model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        if model.additional_blocks:
            model.load_state_dict(
                model_zoo.load_url(
                    model_urls['resnet101']),
                strict=False)
            return model
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


# Above is their custom resnet code


class PSP_head(layers.AnalyticModule):
    """
    Example:
        >>> from .models.psp import *  # NOQA
        >>> self = PSP_head(8).eval()
        >>> inputs = torch.rand(1, 8, 10, 10)
        >>> outputs = self(inputs)
    """

    def __init__(self, in_channels, dropout=0.1):

        super(PSP_head, self).__init__()

        out_channels = int(in_channels / 4)

        Sequential = layers.Sequential

        def conv1x1_norm_noli():
            return Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True))

        self.conv1 = conv1x1_norm_noli()
        self.conv2 = conv1x1_norm_noli()
        self.conv3 = conv1x1_norm_noli()
        self.conv4 = conv1x1_norm_noli()

        self.fusion_bottleneck = Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(dropout, False)
        )

        self.adaptive_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.adaptive_pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.adaptive_pool4 = nn.AdaptiveAvgPool2d((6, 6))

        self._reset_weights()

    def _reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        fcn_features_spatial_dim = x.size()[2:]

        interpkw = dict(size=fcn_features_spatial_dim, mode='bilinear',
                        align_corners=True)
        interpolate = nn.functional.interpolate

        pooled_1 = self.adaptive_pool1(x)
        pooled_1 = self.conv1(pooled_1)
        pooled_1 = interpolate(pooled_1, **interpkw)

        pooled_2 = self.adaptive_pool2(x)
        pooled_2 = self.conv2(pooled_2)
        pooled_2 = interpolate(pooled_2, **interpkw)

        pooled_3 = self.adaptive_pool3(x)
        pooled_3 = self.conv3(pooled_3)
        pooled_3 = interpolate(pooled_3, **interpkw)

        pooled_4 = self.adaptive_pool4(x)
        pooled_4 = self.conv4(pooled_4)
        pooled_4 = interpolate(pooled_4, **interpkw)

        x = torch.cat([x, pooled_1, pooled_2, pooled_3, pooled_4], dim=1)
        x = self.fusion_bottleneck(x)
        return x

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        analytic computation of forward / shape / receptive field.

        Ignore:
            >>> from .models.psp import *  # NOQA
            >>> from viame.pytorch import netharn as nh
            >>> self = PSP_head(8).eval()
            >>> inputs = (1, 128, 64, 64)
            >>> _OutputFor = nh.OutputShapeFor
            >>> _Output = nh.OutputShape
            >>> _Hidden = nh.HiddenShapes
            >>> out = self.output_shape_for(inputs)
            >>> hidden = out.hidden
            >>> print(ub.urepr(hidden.shallow(1), nl=-1))

            >>> # out = self.receptive_field_for(inputs)
        """
        hidden = _Hidden()
        x = inputs

        # hack for output shape with upsample
        fcn_features_spatial_dim = inputs[2:]
        interpkw = dict(size=fcn_features_spatial_dim, mode='bilinear',
                        align_corners=True)
        interpolate = nn.functional.interpolate

        pooled_1 = x
        pooled_1 = hidden['adaptive_pool1'] = _OutputFor(self.adaptive_pool1)(pooled_1)
        pooled_1 = hidden['conv1'] = _OutputFor(self.conv1)(pooled_1)
        pooled_1 = hidden['upsample1'] = _OutputFor(interpolate)(pooled_1, **interpkw)

        pooled_2 = x
        pooled_2 = hidden['adaptive_pool2'] = _OutputFor(self.adaptive_pool2)(pooled_2)
        pooled_2 = hidden['conv2'] = _OutputFor(self.conv2)(pooled_2)
        pooled_2 = hidden['upsample2'] = _OutputFor(interpolate)(pooled_2, **interpkw)

        pooled_3 = x
        pooled_3 = hidden['adaptive_pool3'] = _OutputFor(self.adaptive_pool3)(pooled_3)
        pooled_3 = hidden['conv3'] = _OutputFor(self.conv3)(pooled_3)
        pooled_3 = hidden['upsample3'] = _OutputFor(interpolate)(pooled_3, **interpkw)

        pooled_4 = x
        pooled_4 = hidden['adaptive_pool4'] = _OutputFor(self.adaptive_pool4)(pooled_4)
        pooled_4 = hidden['conv4'] = _OutputFor(self.conv4)(pooled_4)
        pooled_4 = hidden['upsample4'] = _OutputFor(interpolate)(pooled_4, **interpkw)

        out = hidden['cat'] = _OutputFor(torch.cat)([x, pooled_1, pooled_2, pooled_3, pooled_4], dim=1)
        return _Output.coerce(out, hidden)


class PSPNet_Resnet50_8s(layers.AnalyticModule):
    """
    References:
        [1] https://arxiv.org/pdf/1612.01105.pdf

    Notes:
        # Excerpts from Section 3 in.

        The pyramid pooling module fuses features under four different pyramid
        scales.

        The coarsest level highlighted in red is global pooling to generate a single bin output.
        The following pyramid level separates the feature map into different sub-regions and forms pooled representation for different locations.

        The output of different levels in the pyramid pooling module contains the feature map with varied sizes.
        To maintain the weight of global feature, we use 1×1 convolution layer after each pyramid level to reduce the dimension of context representation to 1/N of the original one if the level size of pyramid is N.
        Then we directly upsample the low-dimension feature maps to get the same size feature as the original feature map via bilinear interpolation.
        Finally, different levels of features are concatenated as the final pyramid pooling global feature.
        Noted that the number of pyramid levels and size of each level can be modified.
        They are related to the size of feature map that is fed into the pyramid pooling layer.
        The structure abstracts different sub-regions by adopting varying-size pooling kernels in a few strides.
        Thus the multi-stage kernels should maintain a reasonable gap in representation.
        Our pyramid pooling module is a four-level one with bin sizes of 1×1, 2×2, 3×3 and 6×6 respectively.
        For the type of pooling operation between max and average, we perform extensive experiments to show the difference in Section 5.2

        The final feature map size is 1/8 of the input image,

        Next, we use the pyramid pooling module (4-levels).
        There are 4-level the pooling kernels
        cover the whole, half of, and small portions of the image. They are
        fused as the global prior. Then we concatenate the prior with the
        original feature map in the final part of (c). It is followed by a
        convolution layer to generate the final prediction map in (d).

        To explain our structure, PSPNet provides an effective global
        contextual prior for pixel-level scene parsing. The pyramid pooling
        module can collect levels of information, more representative than
        global pooling [24]. In terms of computational cost, our PSPNet does
        not much increase it compared to the original dilated FCN network. In
        end-toend learning, the global pyramid pooling module and the local FCN
        feature can be optimized simultaneously

    Ignore:
        >>> from .models.psp import *  # NOQA
        >>> import torch
        >>> self = PSPNet_Resnet50_8s(classes=2).eval()
        >>> inputs = torch.rand(1, 3, 64, 64)
        >>> outputs = self(inputs)
        >>> outputs['class_energy'].shape
    """

    def __init__(self, classes=1000, in_channels=3):
        super(PSPNet_Resnet50_8s, self).__init__()

        import ndsampler
        self.classes = ndsampler.CategoryTree.coerce(classes)

        num_classes = len(self.classes)

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet50_8s = resnet50(fully_conv=True, pretrained=True,
                               output_stride=8, remove_avg_pool_layer=True,
                               in_channels=in_channels)

        self.psp_head = PSP_head(resnet50_8s.inplanes)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes // 4, num_classes, 1)

        self.resnet50_8s = resnet50_8s

    def forward(self, inputs):
        input_spatial_dim = inputs.size()[2:]

        x = inputs
        x = self.resnet50_8s.conv1(x)
        x = self.resnet50_8s.bn1(x)
        x = self.resnet50_8s.relu(x)
        x = self.resnet50_8s.maxpool(x)

        x = self.resnet50_8s.layer1(x)
        x = self.resnet50_8s.layer2(x)
        x = self.resnet50_8s.layer3(x)
        resnet_out = self.resnet50_8s.layer4(x)

        psp_out = self.psp_head(resnet_out)
        fc_out = self.resnet50_8s.fc(psp_out)

        class_energy = upsample(input=fc_out, size=input_spatial_dim)

        outputs = {
            'class_energy': class_energy,
        }
        return outputs

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Ignore:
            >>> from .models.psp import *  # NOQA
            >>> from viame.pytorch import netharn as nh
            >>> self = PSPNet_Resnet50_8s(classes=211).eval()
            >>> kwargs = self._analytic_shape_kw()
            >>> globals().update(kwargs)
            >>> inputs = input_shape = (1, 3, 256, 256)
            >>> outputs = self._analytic_forward(inputs, **kwargs)
            >>> print('outputs = {}'.format(ub.urepr(outputs.hidden.shallow(1), nl=-1)))
        """
        try:
            input_spatial_dim = inputs.size()[2:]
        except Exception:
            input_spatial_dim = inputs[2:]

        hidden = _Hidden()

        x = inputs
        hidden['inputs'] = inputs
        x = hidden['conv1'] = _OutputFor(self.resnet50_8s.conv1)(x)
        x = _OutputFor(self.resnet50_8s.bn1)(x)
        x = _OutputFor(self.resnet50_8s.relu)(x)
        x = hidden['maxpool'] = _OutputFor(self.resnet50_8s.maxpool)(x)

        x = hidden['layer1'] = _OutputFor(self.resnet50_8s.layer1)(x)
        x = hidden['layer2'] = _OutputFor(self.resnet50_8s.layer2)(x)
        x = hidden['layer3'] = _OutputFor(self.resnet50_8s.layer3)(x)
        resnet_out = hidden['resnet_out'] = _OutputFor(self.resnet50_8s.layer4)(x)

        psp_out = hidden['psp_out'] = _OutputFor(self.psp_head)(resnet_out)
        fc_out = hidden['fc_out'] = _OutputFor(self.resnet50_8s.fc)(psp_out)

        class_energy = hidden['energy'] = _OutputFor(nn.functional.interpolate)(
            fc_out,
            size=input_spatial_dim,
            mode='bilinear',
            align_corners=True)

        outputs = {
            'class_energy': class_energy,
        }
        outputs = _Output.coerce(outputs, hidden)
        return outputs


def upsample(input, size):
    return nn.functional.interpolate(input=input, size=size, mode='bilinear', align_corners=True)
