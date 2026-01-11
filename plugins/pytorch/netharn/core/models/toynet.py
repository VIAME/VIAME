import torch
import numpy as np
from netharn import layers


class ToyNet1d(layers.Module):
    """
    Demo model for a simple 2 class learning problem

    Example:
        >>> self = ToyNet1d()
        >>> loader = self.demodata().make_loader(batch_size=16, shuffle=True)
        >>> inputs, labels = next(iter(loader))
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> prob = self(nh.XPU().move(inputs))
        >>> conf, pred = prob.max(dim=1)
    """
    def __init__(self, input_channels=2, num_classes=2):
        super(ToyNet1d, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers = torch.nn.Sequential(*[
            torch.nn.Linear(input_channels, 8),

            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8, 8),

            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8, num_classes),

            torch.nn.Softmax(dim=1)
        ])

    def forward(self, inputs):
        return self.layers(inputs)

    @classmethod
    def demodata(ToyNet1d, *args, **kwargs):
        import netharn.data
        dset = netharn.data.ToyData1d(*args, **kwargs)
        return dset


class ToyNet2d(layers.Module):
    """
    Demo model for a simple 2 class learning problem

    Example:
        >>> self = ToyNet2d()
        >>> loader = self.demodata().make_loader(batch_size=16, shuffle=True)
        >>> inputs, labels = next(iter(loader))
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> prob = self(nh.XPU().move(inputs))
        >>> conf, pred = prob.max(dim=1)
    """
    def __init__(self, input_channels=1, num_classes=2):
        super(ToyNet2d, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers = torch.nn.Sequential(*[
            torch.nn.Conv2d(input_channels, 8, kernel_size=3, padding=1, bias=False),

            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),

            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, num_classes, kernel_size=3, padding=1, bias=False),
        ])

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs):
        spatial_out = self.layers(inputs)
        num = float(np.prod(spatial_out.shape[-2:]))
        averaged = spatial_out.sum(dim=2).sum(dim=2) / num
        probs = self.softmax(averaged)
        return probs

    @classmethod
    def demodata(ToyNet2d, *args, **kwargs):
        import netharn.data
        dset = netharn.data.ToyData2d(*args, **kwargs)
        return dset


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.models.toynet all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
