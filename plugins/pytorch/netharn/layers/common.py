from viame.pytorch.netharn import util
from torch import nn
import torch


# TODO: move module mixin from util to here
ModuleMixin = util.ModuleMixin


class Module(nn.Module, util.ModuleMixin):
    """
    Like torch.nn.Module but contains ModuleMixin (for number_of_parameters)
    """

    def output_shape_for(self, input_shape):
        raise NotImplementedError

    def receptive_field_for(self, input_field=None):
        raise NotImplementedError


class Sequential(nn.Sequential, util.ModuleMixin):
    """
    Like torch.nn.Sequential but implements output_shape_for

    Example:
        >>> import torch
        >>> from viame.pytorch import netharn as nh
        >>> import ubelt as ub
        >>> self = nh.layers.Sequential(
        >>>     torch.nn.Conv2d(2, 3, kernel_size=3),
        >>>     torch.nn.Conv2d(3, 5, kernel_size=3),
        >>>     torch.nn.Conv2d(5, 7, kernel_size=3),
        >>> )
        >>> shape = self.output_shape_for([1, 1, 7, 11])
        >>> print('shape = {}'.format(shape))
        >>> print('shape.hidden = {}'.format(ub.urepr(shape.hidden, nl=1)))
        shape = (1, 7, 1, 5)
        shape.hidden = {
            '0': (1, 3, 5, 9),
            '1': (1, 5, 3, 7),
            '2': (1, 7, 1, 5),
        }
    """
    def output_shape_for(self, input_shape):
        from .analytic import output_shape_for
        return output_shape_for.OutputShapeFor.sequential(self, input_shape)

    def receptive_field_for(self, input_field=None):
        from .analytic import receptive_field_for
        return receptive_field_for.ReceptiveFieldFor.sequential(self, input_field)


class Identity(Sequential):
    """
    A identity-function layer.

    Example:
        >>> import torch
        >>> self = Identity()
        >>> a = torch.rand(3, 3)
        >>> b = self(a)
        >>> assert torch.all(a == b)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def output_shape_for(self, input_shape):
        return input_shape

    def receptive_field_for(self, input_field=None):
        if input_field is None:
            from .analytic import receptive_field_for
            input_field = receptive_field_for.ReceptiveFieldFor.input()
        return input_field


class AnalyticModule(Module):
    """
    Provides default implementations for output_shape_for and
    receptive_field_for if _analytic_forward is defined
    """

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Defines a symbolic representation of forward, output shape, and receptive field.

        Notes:
            The basic formula for implementing this function is
            x = inputs

            # Create a hidden object to store intermediate computations
            hidden = _Hidden()

            # Wrap any torch module callable with _OutputFor
            x = hidden['mylayer1'] = _OutputFor(self.mylayer1)(x)
            x = hidden['mylayer2'] = _OutputFor(self.mylayer2)(x)

            # Create an output object with the output data and hidden info
            out = _Output.coerce(x, hidden)
            return out

        Example:
            >>> from viame.pytorch import netharn as nh
            >>> globals().update(nh.layers.AnalyticModule._analytic_shape_kw())
        """
        # _Output.coerce(output, hidden)
        raise NotImplementedError

    @classmethod
    def _analytic_shape_kw(self):
        from .analytic import output_shape_for
        return {
            '_OutputFor': output_shape_for.OutputShapeFor,
            '_Output': output_shape_for.OutputShape,
            '_Hidden': output_shape_for.HiddenShapes
        }

    @classmethod
    def _analytic_field_kw(self):
        from .analytic import receptive_field_for
        # from viame.pytorch import netharn as nh
        return {
            '_OutputFor': receptive_field_for.ReceptiveFieldFor,
            '_Output': receptive_field_for.ReceptiveField,
            '_Hidden': receptive_field_for.HiddenFields
        }

    @classmethod
    def _analytic_forward_kw(self):
        # from viame.pytorch import netharn as nh
        from .analytic import analytic_for
        return {
            '_OutputFor': analytic_for.ForwardFor,
            '_Output': analytic_for.Output,
            '_Hidden': analytic_for.Hidden,
        }

    def output_shape_for(self, input_shape, **kwargs):
        """
        Uses custom _analytic_forward to compute output shape
        """
        kw = self._analytic_shape_kw()
        if kwargs:
            kw = kw.copy()
            kw.update(kwargs)
        return self._analytic_forward(input_shape, **kw)

    def receptive_field_for(self, input_field=None, **kwargs):
        """
        Uses custom _analytic_forward to compute receptive field
        """
        # from viame.pytorch import netharn as nh
        from .analytic import receptive_field_for
        if input_field is None:
            input_field = receptive_field_for.ReceptiveFieldFor.input()
        kw = self._analytic_field_kw()
        if kwargs:
            kw = kw.copy()
            kw.update(kwargs)
        return self._analytic_forward(input_field, **kw)

    def forward(self, inputs, **kwargs):
        """
        Uses custom _analytic_forward to compute receptive field
        """
        kw = self._analytic_forward_kw()
        if kwargs:
            kw = kw.copy()
            kw.update(kwargs)
        return self._analytic_forward(inputs, **kw)


class Loss(nn.modules.loss._Loss):
    """
    Helper to keep track of if a loss module is in cpu or gpu mod
    """

    def __init__(self):
        super(Loss, self).__init__()
        self._iscuda = False
        self._device_num = None

    def cuda(self, device_num=None, **kwargs):
        self._iscuda = True
        self._device_num = device_num
        return super(Loss, self).cuda(device_num, **kwargs)

    def cpu(self):
        self._iscuda = False
        self._device_num = None
        return super(Loss, self).cpu()

    @property
    def is_cuda(self):
        return self._iscuda

    def get_device(self):
        if self._device_num is None:
            return torch.device('cpu')
        return self._device_num
