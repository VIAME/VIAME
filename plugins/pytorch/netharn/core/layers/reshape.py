import torch
from netharn import util
from .analytic import output_shape_for


class Reshape(torch.nn.Module, util.ModuleMixin):
    """
    Wrapper class around `torch.view` that implements `output_shape_for`

    TODO:
        [ ] - Can we implement receptive_feild_for for this layer?

    Args:
        *shape: same ars that would be passed to view.
            if an item in shape is None it means that the output
            shape should keep the input shape value in that dimension

    Example:
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> nh.OutputShapeFor(Reshape(-1, 3))._check_consistency((20, 6, 20))
        (800, 3)
        >>> nh.OutputShapeFor(Reshape(100, -1, 5))._check_consistency((10, 10, 15))
        (100, 3, 5)
        >>> Reshape(7, -1, 3).output_shape_for((None, 1))  # weird case
        (7, None, 3)
        >>> nh.OutputShapeFor(Reshape(None, -1, 4))._check_consistency((10, 32, 32, 16))
        (10, 4096, 4)
        >>> Reshape(None, -1, 4).output_shape_for((None, 32, 32, 16))
        (None, 4096, 4)
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> nh.OutputShapeFor(Reshape(-1, 3))._check_consistency((20, 6, 20))

    Ignore:
        >>> from .layers.reshape import *
        >>> self = Reshape(None, 1600)
        >>> input_shape = (10, 64, 5, 5)
        >>> nh.OutputShapeFor(self)._check_consistency(input_shape)
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
        if len(shape) == 0:
            raise ValueError('Reshape dims cannot be empty')

        self._none_dims = []
        self._neg_dims = []
        for i, d in enumerate(shape):
            if d is None:
                self._none_dims.append(i)
            elif d < 0:
                self._neg_dims.append(i)
        if len(self._neg_dims) > 1:
            raise ValueError('Can only specify one negative dimension')

    def forward(self, input):
        """
        Example:
            >>> from viame.arrows.pytorch.netharn import core as nh
            >>> self = Reshape(None, -1, 4)
            >>> input_shape = (10, 32, 32, 16)
            >>> input = torch.rand(input_shape)
            >>> output = self.forward(input)
            >>> print(tuple(output.shape))
            (10, 4096, 4)
            >>> print(tuple(self.output_shape_for(input_shape)))
            >>> nh.OutputShapeFor(self)._check_consistency(input_shape)
        """
        if not self._none_dims:
            output_shape = self.shape
        else:
            output_shape = list(self.shape)
            input_shape = input.shape
            for i in self._none_dims:
                if i >= len(input_shape):
                    raise ValueError('input shape does not correspond')
                output_shape[i] = input_shape[i]
        return input.view(*output_shape)

    def extra_repr(self):
        """
        Example:
            >>> print(Reshape(-1, 10))
            Reshape(-1, 10)
            >>> print(Reshape(5, 5, 5))
            Reshape(5, 5, 5)
        """
        return '{}'.format(', '.join(str(s) for s in self.shape))

    def output_shape_for(self, input_shape):
        # Not sure if this works in all cases
        # TODO: find a cleaner (and correct) implementation

        if len(input_shape) == 0:
            raise ValueError('input shape cannot be empty')

        # If any dim in output_shape is set to None, it should use whatever the
        # corresonding value in input shape is. This feature is not part of
        # standard torch.view
        output_shape = list(self.shape)
        for i in self._none_dims:
            if i >= len(input_shape):
                raise ValueError('input shape does not correspond')
            output_shape[i] = input_shape[i]

        # Check how many total numbers are in the input / if the input
        # has an unspecified batch dimension.
        input_has_none = input_shape[0] is None
        input_total = 1 if input_has_none else input_shape[0]

        for i, d in enumerate(input_shape[1:], start=1):
            if d is None:
                raise ValueError(
                    'Invalid input shape: input_shape[{}] = None, '
                    'but only the first item can be None'.format(i))
            input_total *= d

        # Check the total numbers that the output shape wants

        can_check_fit = not input_has_none or (self._none_dims == [0] and input_has_none)

        if can_check_fit:
            unused = input_total
            for j, s in enumerate(output_shape):
                if j not in self._neg_dims and not (j in self._none_dims and input_has_none):
                    # if not input_has_none:
                    if s > input_total or input_total % s != 0:
                        raise ValueError('does not fit: input_shape={} -> {}'.format(input_shape, self))
                    unused = unused // s

        if self._neg_dims:
            if len(self._neg_dims) > 1:
                raise ValueError('Can only specify -1 in reshape dim once')
            j = self._neg_dims[0]
            if can_check_fit:
                output_shape[j] = unused
            else:
                output_shape[j] = None
        elif can_check_fit:
            assert unused == 1

        return output_shape_for.OutputShape.coerce(tuple(output_shape))


class Permute(torch.nn.Module, util.ModuleMixin):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def output_shape_for(self, input_shape):
        output_shape = tuple([input_shape[i] for i in self.dims])
        output_shape_for.OutputShape.coerce(output_shape)
        return output_shape
