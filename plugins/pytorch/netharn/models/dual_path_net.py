"""
Dual Path Networks in PyTorch.

TODO: move to backbones

References:
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/dpn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .analytic.output_shape_for import OutputShapeFor
from viame.pytorch.netharn.layers import ConvNorm2d

# __all__ = ['DPN']


class Bottleneck(nn.Module):
    """
    Maintains a dense channel and a residual channel.
    First


    Multiple Bottleneck "Blocks" make a Layer

    Args:
        last_planes (int): total number of input channels
        in_planes (int): number of bottleneck features

        out_planes (int): width of the residual channel
        dense_depth (int): increment on the width of the dense channel
        stride (int):

        groups (int): number of groups for grouped convolutions

    Example:
        >>> last_planes = 64
        >>> in_planes = 96
        >>> out_planes = 256
        >>> dense_depth = 16
        >>> stride = 1
        >>> self = Bottleneck(last_planes, in_planes, out_planes, dense_depth, stride, first_layer=True)
        >>> x = inputs = torch.ones(1, last_planes, 5, 7)

    """
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride,
                 first_layer, groups=32):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        # self.conv1 = nn.Conv2d(last_planes, in_planes,
        #                        kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(in_planes)

        # self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3,
        #                        stride=stride, padding=1, groups=groups,
        #                        bias=False)

        # self.bn2 = nn.BatchNorm2d(in_planes)

        # self.conv3 = nn.Conv2d(in_planes, out_planes +
        #                        dense_depth, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_planes + dense_depth)

        self.conv1 = ConvNorm2d(last_planes, in_planes, kernel_size=1,
                                bias=False, norm='batch', noli='relu')

        self.conv2 = ConvNorm2d(in_planes, in_planes, kernel_size=3,
                                stride=stride, padding=1, groups=groups,
                                bias=False, norm='batch', noli='relu')

        # Last conv is not given a ReLU
        self.conv3 = ConvNorm2d(in_planes, out_planes + dense_depth,
                                kernel_size=1, bias=False, norm='batch',
                                noli=None)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes + dense_depth,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes + dense_depth)
            )

    def forward(self, x):
        out = x
        # 1x1 conv + bn + relu
        out = self.conv1(out)
        # 3x3 strided grouped conv + bn + relu
        out = self.conv2(out)
        # 1x1 conv + bn
        out = self.conv3(out)

        # out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        # out = self.bn3(self.conv3(out))

        # Identity on all layers (within the block) but the first
        # On the first layer in the block, do a strided 1x1 to map to the
        # appropriate feature size.
        x_ = self.shortcut(x)
        d = self.out_planes

        residuals = out[:, :d, ...]
        new_depth = out[:, d:, ...]

        out = torch.cat([
            # Residual channel
            x_[:, :d, ...] + residuals,
            # Dense Channel
            x_[:, d:, ...], new_depth], dim=1)
        out = F.relu(out)
        return out


class DPN(nn.Module):
    """
    Dual Path Network

    References:
        https://arxiv.org/abs/1707.01629

    Example:
        >>> import torch
        >>> cfg26 = {
        >>>     'in_planes': (96, 192, 384, 768),
        >>>     'out_planes': (256, 512, 1024, 2048),
        >>>     'num_blocks': (2, 2, 2, 2),
        >>>     'dense_depth': (16, 32, 24, 128)
        >>> }
        >>> net = DPN(cfg26)
        >>> x = torch.randn(1, 3, 32, 32)
        >>> y = net(x)
        >>> print(tuple(y.shape))
        (1, 10)

    """
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']
        num_classes = cfg.get('num_classes', 10)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(
            in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(
            in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(
            in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(
            in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(
            out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            first_layer = i == 0
            layers.append(Bottleneck(self.last_planes, in_planes,
                                     out_planes, dense_depth, stride,
                                     first_layer))
            self.last_planes = out_planes + (i + 2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DPN26():
    cfg = {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (2, 2, 2, 2),
        'dense_depth': (16, 32, 24, 128)
    }
    return DPN(cfg)


def DPN92():
    cfg = {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (3, 4, 20, 3),
        'dense_depth': (16, 32, 24, 128)
    }
    return DPN(cfg)


# test()

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.models.dual_path_net all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
