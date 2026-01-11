# coding=utf8
import math
import numpy as np
import torch
from .layers import common
import torch.nn.functional as F


class Conv1d_pad(torch.nn.Conv1d, common.ModuleMixin):
    """
    Extends convolutions to include the type of mode used in padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, mode='reflect'):
        super(Conv1d_pad, self).__init__(in_channels, out_channels,
                                         kernel_size, stride=stride, padding=0,
                                         dilation=dilation, groups=groups,
                                         bias=bias)
        self._custom_padding = padding
        self._pad_mode = mode
        self.ndim = 1

    def forward(self, input):
        # Custom pad, then convolve
        x = F.pad(input, [self._custom_padding] * (2 * self.ndim), mode=self._pad_mode)
        return super(Conv1d_pad, self).forward(x)


class Conv2d_pad(torch.nn.Conv2d, common.ModuleMixin):
    """
    Extends convolutions to include the type of mode used in padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, mode='reflect'):
        super(Conv2d_pad, self).__init__(in_channels, out_channels,
                                         kernel_size, stride=stride, padding=0,
                                         dilation=dilation, groups=groups,
                                         bias=bias)
        self._custom_padding = padding
        self._pad_mode = mode
        self.ndim = 2

    def forward(self, input):
        # Custom pad, then convolve
        x = F.pad(input, [self._custom_padding] * (2 * self.ndim), mode=self._pad_mode)
        return super(Conv2d_pad, self).forward(x)


class GaussianBlurNd(common.Module):
    """
    Convolves a signal with a Gaussian kernel.

    Args:
        dim (int): number of input space-time dims (ignore batch / channel)

        num_features (int): number of incoming channels

        sigma (float): standard deviation of Gaussian kernel.

        truncate (float): ignored if kernel_size is not None, otherwise
            computes kernel size that cuts off the filter at this many standard
            deviations. (See scipy.ndimage.gaussian_filter1d)

        kernel_size (int): discrete window size (if not given, it is computed
            based on sigma and truncate)

        separable (bool): if True uses n seperable 1d convolutions instead of
            a single Nd convolution. This usually results in a speedup.
            Defaults to True. Note: the False case is only implemented for
            `dim=2`.

    Example:
        >>> # xdoc: +REQUIRES(module:kwimage)
        >>> from .layers.gauss import *  # NOQA
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> from torchvision.transforms.functional import to_tensor
        >>> import kwimage
        >>> input = to_tensor(kwimage.grab_test_image())[None, :]
        >>> self = GaussianBlurNd(2, num_features=input.shape[1], sigma=1.4)
        >>> assert self.number_of_parameters() == 0
        >>> output = self(input)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(input[0], colorspace='rgb', pnum=(1, 2, 1))
        >>> kwplot.imshow(output[0], colorspace='rgb', pnum=(1, 2, 2))
        >>> kwplot.show_if_requested()

    References:
        https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351

    Benchmark:
        from viame.arrows.pytorch.netharn import core as nh
        import kwimage
        import cv2
        import ubelt as ub
        from torchvision.transforms.functional import to_tensor

        image = kwimage.grab_test_image('carl')
        image = cv2.resize(image, (256, 256))
        im = to_tensor(image)
        input = torch.stack([im] * 16, dim=0)

        self1 = GaussianBlurNd(2, 3, sigma=math.sqrt(2), separable=True)
        self2 = GaussianBlurNd(2, 3, sigma=math.sqrt(2), separable=False)

        xpu = nh.XPU(0)
        self1, self2, input = map(xpu.move, [self1, self2, input])

        # Applying the 1d Gaussian multiple times has a clear time advantage
        ti = ub.Timerit(100, bestof=10, verbose=1)
        for timer in ti.reset('time-1d'):
            with timer:
                self1(input)
                torch.cuda.synchronize()
        # Timed best=2.162 ms, mean=2.167 ± 0.0 ms for time-1d

        for timer in ti.reset('time-2d'):
            with timer:
                self2(input)
                torch.cuda.synchronize()
        # Timed best=5.998 ms, mean=6.082 ± 0.1 ms for time-2d
    """

    def __init__(self, dim, num_features, sigma, kernel_size=None,
                 truncate=4.0, separable=False):
        super(GaussianBlurNd, self).__init__()
        self.separable = separable

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        # truncate = 4
        if kernel_size is None:
            lw = int(truncate * sigma + 0.5)
            kernel_size = lw * 2 + 1
        else:
            assert kernel_size % 2 == 1
            lw = (kernel_size - 1) // 2

        if self.separable:
            # Calculate the 1d Gaussian kernel
            # Follow scipy.ndimage method closely
            import scipy
            import scipy.ndimage
            kernel1d = scipy.ndimage.filters._gaussian_kernel1d(sigma, order=0, radius=lw)[::-1]
            kernel1d = torch.from_numpy(np.ascontiguousarray(kernel1d))

            # Reshape to 1d depthwise convolutional weight
            kernel1d = kernel1d.view(1, kernel_size)
            kernel1d = kernel1d.repeat(num_features, 1, 1)

            self.padding = lw

            gauss1d = Conv1d_pad(in_channels=num_features, out_channels=num_features,
                                 kernel_size=kernel_size, groups=num_features,
                                 bias=False, padding=self.padding)

            gauss1d.weight.data[:] = kernel1d
            gauss1d.weight.requires_grad = False
            self.gauss1d = gauss1d
        else:
            kernel_size = lw * 2 + 1
            mean = (kernel_size - 1) / 2.0
            x_cord = torch.arange(kernel_size)
            x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

            coeff = (1.0 / (2.0 * math.pi * sigma ** 2.0))
            exponent = -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * sigma ** 2.0)
            kernel2d = coeff * torch.exp(exponent)
            kernel2d /= torch.sum(kernel2d)

            # Reshape to 2d depthwise convolutional weight
            kernel2d = kernel2d.view(1, 1, kernel_size, kernel_size)
            kernel2d = kernel2d.repeat(num_features, 1, 1, 1)

            gauss2d = Conv2d_pad(in_channels=num_features, out_channels=num_features,
                                 kernel_size=kernel_size, groups=num_features,
                                 bias=False, padding=kernel_size // 2)

            gauss2d.weight.data[:] = kernel2d
            gauss2d.weight.requires_grad = False
            self.gauss2d = gauss2d

    def forward(self, input):
        if self.separable:
            # Apply the 1d gaussian filter in each dimension
            b, c = input.shape[0:2]
            dims = input.shape[2:]
            x = input
            major_dim = 2  # tensor dim corresponding to major spatial axis
            for axis, d in enumerate(dims):
                if axis > 0:
                    # Transpose to put the axis of interest in front
                    x = x.transpose(major_dim, major_dim + axis).contiguous()
                # Flatten so current axis is stacked
                shape = x.shape
                axis_view = x.view(b, c, -1)
                axis_view = self.gauss1d.forward(axis_view)
                x = axis_view.view(*shape)
                if axis > 0:
                    # Restore original axis order
                    x = x.transpose(major_dim, major_dim + axis)
            output = x.contiguous()
            return output
        else:
            return self.gauss2d.forward(input)

    def output_shape_for(self, input_shape):
        return input_shape

    def receptive_field_for(self, input_field=None):
        # Even though this does change the receptive feild a little bit
        # lets pretend that it doesnt
        return input_field
