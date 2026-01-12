import torch.nn.functional as F
import torch
from torch import nn
import ubelt as ub
from viame.pytorch.netharn.layers import common
from viame.pytorch.netharn.analytic import output_shape_for
from viame.pytorch.netharn.analytic import receptive_field_for


class L2Norm(common.Module):
    """
    L2Norm layer across all channels

    Notes:
        "The L2 normalization technique introduced in [12] to scale the feature
        norm at each location in the feature map to `scale` and learn the scale
        during back propagation."

        * In my experience (author of netharn), using this has tended to hurt
        more than help.

    References:
        [12] Liu, Rabinovich, Berg - ParseNet: Looking wider to see better (ILCR) (2016)

    Example:
        >>> import numpy as np
        >>> import ubelt as ub
        >>> in_features = 7
        >>> self = L2Norm(in_features, scale=20)
        >>> x = torch.rand(1, in_features, 2, 2)
        >>> y = self(x)
        >>> norm = np.linalg.norm(y.data.cpu().numpy(), axis=1)
        >>> print(ub.repr2(norm, precision=2))
        np.array([[[20., 20.],
                   [20., 20.]]], dtype=np.float32)

    Example:
        >>> from .analytic.output_shape_for import OutputShapeFor
        >>> self = L2Norm(in_features=7, scale=20)
        >>> OutputShapeFor(self)._check_consistency((1, 7, 2, 2))
        (1, 7, 2, 2)
    """

    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self._initial_scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self._initial_scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        return scale * x

    def output_shape_for(self, input_shape):
        return output_shape_for.OutputShape.coerce(input_shape)

    def receptive_field_for(self, input_field=None):
        return input_field


class InputNorm(common.Module):
    """
    Normalizes the input by shifting and dividing by a scale factor.

    This allows for the network to take care of 0-mean 1-std normalization.
    The developer explicitly specifies what these shift and scale values are.
    By specifying this as a layer (instead of a data preprocessing step), the
    netharn exporter will remember and associated this information with any
    deployed model. This means that a user does not need to remember what these
    shit/scale arguments were before passing inputs to a network.

    If the mean and std arguments are unspecified, this layer becomes a noop.

    References:
        https://towardsdatascience.com/why-data-should-be-normalized-before-training-a-neural-network-c626b7f66c7d

    Example:
        >>> self = InputNorm(mean=50.0, std=29.0)
        >>> inputs = torch.rand(2, 3, 5, 7) * 100
        >>> outputs = self(inputs)
        >>> # If mean and std are unspecified, this becomes a noop.
        >>> assert torch.all(InputNorm()(inputs) == inputs)
        >>> # Specifying either the mean or the std is ok.
        >>> partial1 = InputNorm(mean=50)(inputs)
        >>> partial2 = InputNorm(std=29)(inputs)

        import torch

        model = torch.nn.Sequential(*[
            InputNorm(mean=10, std=0.2),
            torch.nn.Conv2d(3, 3, 3),
        ])
        inputs = torch.rand(2, 3, 5, 7) * 100
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)

        for i in range(100):
            optim.zero_grad()
            x = model(inputs).sum()
            x.backward()
            optim.step()

            std = model[0].mean
            mean = model[0].std
            print('std = {!r}'.format(std))
            print('mean = {!r}'.format(mean))
    """

    def __init__(self, mean=None, std=None):
        super(InputNorm, self).__init__()
        if mean is not None:
            mean = mean if ub.iterable(mean) else [mean]
            mean = torch.FloatTensor(mean)
        if std is not None:
            std = std if ub.iterable(std) else [std]
            std = torch.FloatTensor(std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, inputs):
        outputs = inputs
        if self.mean is not None:
            mean = self.mean
            # Reshape for broadcasting with (B, C, H, W) inputs if needed
            if mean.dim() == 1:
                mean = mean.view(1, -1, 1, 1)
            # Ensure mean is on the same device as inputs
            if mean.device != inputs.device:
                mean = mean.to(inputs.device)
            outputs = outputs - mean
        if self.std is not None:
            std = self.std
            # Reshape for broadcasting with (B, C, H, W) inputs if needed
            if std.dim() == 1:
                std = std.view(1, -1, 1, 1)
            # Ensure std is on the same device as inputs
            if std.device != inputs.device:
                std = std.to(inputs.device)
            outputs = outputs / std
        return outputs

    def output_shape_for(self, input_shape):
        return output_shape_for.OutputShape.coerce(input_shape)

    def receptive_field_for(self, input_field=None):
        return receptive_field_for.ReceptiveField.coerce(input_field)
