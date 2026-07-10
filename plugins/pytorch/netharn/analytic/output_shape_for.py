# -*- coding: utf-8 -*-
import ubelt as ub
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from collections import OrderedDict
import builtins
from viame.pytorch.netharn.analytic import analytic_for
from viame.pytorch.netharn.device import DataSerial

REGISTERED_TYPES = []


SHAPE_CLS = tuple  # We exepct shapes to be specified as this class


def compute_type(*types):
    def _wrap(func):
        for type in types:
            if type is not None:
                REGISTERED_TYPES.append((type, func))
        return func
    return _wrap


def output_shape_of(outputs):
    """
    Given a network output, try and find the shape. Works in most standard
    cases, but not all cases.

    Args:
        outputs (Tensor | Dict | Tuple): some typical torch network output

    Example:
        >>> output_shape_of(torch.empty(3, 2))
        (3, 2)
        >>> output_shape_of({'a': torch.empty(3, 2)})
        {'a': (3, 2)}
        >>> output_shape_of(((torch.empty(3, 2),),))
        [[(3, 2)]]
    """
    if torch.is_tensor(outputs):
        computed_output_shape = SHAPE_CLS(outputs.shape)
    elif isinstance(outputs, dict):
        dict_cls = outputs.__class__  # handle odict
        computed_output_shape = dict_cls([
            (k, output_shape_of(v)) for k, v in outputs.items()])
    elif isinstance(outputs, tuple):
        # Allow outputs to be a tuple of tensors
        computed_output_shape = [output_shape_of(o) for o in outputs]
    else:
        raise TypeError('Cannot find shape of {!r}'.format(type(outputs)))
    return computed_output_shape


def _brute_force_output_shape_for(self, input_shape):
    """
    Computes output shape by actually running the network. Works in most
    standard cases, but not all cases. If the batch size is None, we attempt to
    be smart about ensuring that that None is propogated in the output.

    Example:
        >>> module = nn.Conv2d(3, 11, 3, 1, 0)
        >>> _brute_force_output_shape_for(module, (None, 3, 256, 256))
        (None, 11, 254, 254)
    """
    _input_shape = list(input_shape)
    unknown_bsize = _input_shape[0] is None
    if unknown_bsize:
        bsize = 2
        _input_shape[0] = bsize
    device = next(iter(self.state_dict().values())).device
    dummy_input = torch.rand(*_input_shape).to(device)
    dummy_output = self(dummy_input)
    output_shape = output_shape_of(dummy_output)
    if torch.is_tensor(dummy_output):
        if unknown_bsize:
            if output_shape[0] == bsize:
                output_shape = list(output_shape)
                output_shape[0] = None
        output_shape = SHAPE_CLS(output_shape)
    else:
        raise NotImplementedError('other output types')
    return output_shape


def _simplify(shape):
    import sympy
    if isinstance(shape, (tuple, list)):
        shape = shape.__class__([_simplify(v) for v in shape])
    elif isinstance(shape, dict):
        shape = shape.__class__([(k, _simplify(v)) for k, v in shape.items()])
    elif isinstance(shape, sympy.Expr):
        shape = sympy.simplify(shape)
    return shape


class HiddenShapes(analytic_for.Hidden):
    """
    Augments normal hidden shape dicts with a convinience setitem

    Doctest:
        >>> from .analytic.output_shape_for import *
        >>> shape = OutputShape.coerce([None, 3, 32, 32], 'foo')
        >>> print(HiddenShapes({'e': shape}))
        <HiddenShapes({'e': 'foo'})>
        >>> hidden = HiddenShapes({'a': 1})
        >>> hidden['b'] = 2
        >>> hidden['c'] = shape
        >>> print(hidden)
        <HiddenShapes({'a': 1, 'b': 2, 'c': 'foo'})>
    """
    pass


# class HiddenShapes(OrderedDict, ub.NiceRepr):
#     """
#     Augments normal hidden shape dicts with a convinience setitem

#     Doctest:
#         >>> from .analytic.output_shape_for import *
#         >>> shape = OutputShape.coerce([None, 3, 32, 32], 'foo')
#         >>> print(HiddenShapes({'e': shape}))
#         <HiddenShapes({'e': 'foo'})>
#         >>> hidden = HiddenShapes({'a': 1})
#         >>> hidden['b'] = 2
#         >>> hidden['c'] = shape
#         >>> print(hidden)
#         <HiddenShapes({'a': 1, 'b': 2, 'c': 'foo'})>
#     """
#     def __nice__(self):
#         return ub.urepr(self, nl=0)

#     def __str__(self):
#         return ub.NiceRepr.__str__(self)

#     def __repr__(self):
#         return ub.NiceRepr.__repr__(self)

#     def __setitem__(self, key, value):
#         if getattr(value, 'hidden', None) is not None:
#             # When setting a value to an OutputShape object, if that object has
#             # a hidden shape, then use that instead.
#             value = value.hidden
#         return OrderedDict.__setitem__(self, key, value)

#     def shallow(self, n=1):
#         """
#         Grabs only the shallowest n layers of hidden shapes
#         """
#         if n == 0:
#             last = self
#             while isinstance(last, HiddenShapes):
#                 values = list(last.values())
#                 if len(values):
#                     last = values[-1]
#                 else:
#                     break
#             return last
#         else:
#             output = OrderedDict()
#             for key, value in self.items():
#                 # if isinstance(value, HiddenShapes):
#                 if hasattr(value, 'shallow'):
#                     value = value.shallow(n - 1)
#                 output[key] = value
#             return output


class OutputShape(analytic_for.Output):
    """
    Mixin class to extend output shapes with extra information

    Doctest:
        >>> from .analytic.output_shape_for import *
        >>> shape = OutputShape.coerce([None, 3, 32, 32], 'foo')
        >>> print('shape = {!r}'.format(shape))
        shape = (None, 3, 32, 32)
        >>> print('shape.hidden = {!r}'.format(shape.hidden))
        shape.hidden = 'foo'
    """
    def __init__(self, data=None, hidden=None):
        self.data = data
        self.hidden = hidden

    @classmethod
    def template(cls, type):
        """ Get a specific template for a subclass type """
        if issubclass(type, tuple):
            return OutputShapeTuple
        elif issubclass(type, OrderedDict):
            return OutputShapeDict
        elif issubclass(type, dict):
            return OutputShapeDict
        else:
            raise TypeError(type)

    @classmethod
    def coerce(cls, data=None, hidden=None):
        """
        Create an OutputShape instance of the approriate subclass given the
        type of input data.
        """
        if isinstance(data, cls):
            if hidden is None:
                self = data
            else:
                self = data.__class__(data, hidden)
        elif isinstance(data, (tuple, list)):
            self = cls.template(tuple)(data, hidden)
        elif isinstance(data, dict):
            self = cls.template(dict)(data, hidden)
        else:
            raise TypeError(type(data))
        return self


class OutputShapeTuple(tuple, OutputShape):
    """
    OutputShape templated as a tuple

    Example:
        >>> from .analytic.output_shape_for import *  # NOQA
        >>> self = OutputShapeTuple((1, 2, 3))
        >>> print(self)
        (1, 2, 3)
    """
    def __new__(cls, data=None, hidden=None):
        # tuple subclass is a bit weird
        if data is None:
            data = tuple()
        self = tuple.__new__(OutputShapeTuple, data)
        OutputShape.__init__(self, data, hidden)
        return self


class OutputShapeDict(OrderedDict, OutputShape):
    """ OutputShape templated as a dictionary """
    def __init__(self, data=None, hidden=None):
        if data is None:
            data = OrderedDict()
        OrderedDict.__init__(self, data)
        OutputShape.__init__(self, data, hidden)


class OutputShapeFor(analytic_for.OutputFor):
    """
    Compute the output shape for standard torch modules as well as
    any custom modules that follow the OutputShapeFor protocol.

    Notes:
        The OutputShapeFor protocol is simple. For any custom torch module
        define the method `output_shape_for(self, input_shape)`, which is
        typically written to mirror the `forward` function. Instead of calling
        forward on the custom module's torch members use `OutputShapeFor`. See
        netharn.layers for more examples of custom layers that implement this
        protocol. A simple example is shown below.

    Example:
        >>> # Example showing how to implement the OutputShapeFor protocol
        >>> class MyCustomNet(nn.Module):
        >>>     def __init__(self):
        >>>         super(MyCustomNet, self).__init__()
        >>>         self.conv1 = nn.Conv2d(1, 5, 3)
        >>>         self.pool1 = nn.MaxPool2d(2)
        >>>         self.conv2 = nn.Conv2d(5, 7, 3)
        >>>     def forward(self, input):
        >>>         x = input
        >>>         x = self.conv1(x)
        >>>         x = self.pool1(x)
        >>>         x = self.conv2(x)
        >>>         return x
        >>>     def output_shape_for(self, input_shape):
        >>>         x = input_shape
        >>>         # Note using hidden shapes is optional, but sometimes useful
        >>>         hidden = HiddenShapes()
        >>>         # The basic idea is to simply mirror the forward func
        >>>         # but instead of calling the modules use output shape for
        >>>         hidden['conv1'] = x = OutputShapeFor(self.conv1)(x)
        >>>         hidden['pool1'] = x = OutputShapeFor(self.pool1)(x)
        >>>         hidden['conv2'] = x = OutputShapeFor(self.conv2)(x)
        >>>         shape = OutputShape.coerce(x, hidden)
        >>>         return shape
        >>> net = MyCustomNet()
        >>> # Now it is very easy and efficient to infer the output shape
        >>> input_shape = (None, 1, 9, 9)
        >>> net.output_shape_for(input_shape)
        (None, 7, 1, 1)
        >>> # The OutputShapeFor class now recognizes your module as well
        >>> # so it can be used to constuct more complex modules while
        >>> # still maintaining the ability fo infer the output shape.
        >>> OutputShapeFor(net)(input_shape)
        (None, 7, 1, 1)
        >>> # Note that if you did return an true OutputShape object with
        >>> # a populated hidden shape attribute, then you can access it
        >>> # to inspect how the shape changes in the hidden layer of the net
        >>> print(OutputShapeFor(net)(input_shape).hidden)
        <HiddenShapes({'conv1': (None, 5, 7, 7), 'pool1': (None, 5, 3, 3), 'conv2': (None, 7, 1, 1)})>

    Example:
        >>> # Example showing how this class is used on basic torch Modules
        >>> module = nn.Conv2d(3, 11, 3, 1, 0)
        >>> OutputShapeFor(module)((1, 3, 256, 256))
        (1, 11, 254, 254)
    """
    math = math  # for hacking in sympy

    def __init__(self, module, force=False):
        """
        Args:
            module (nn.Module) : module with output_shape_for func or
                with some known registered type (e.g. torch.nn.Conv2d).

            force (bool): if True and no implicit computation is known
                try to create a dummy input with input_shape and simply
                run it through the network to see what shape it produces.
                (Defaults to False).
        """
        self._requires_force = False
        self.module = module
        # First try to lookup the output_shape_for func
        self._func = getattr(module, 'output_shape_for', None)

        if self._func is None:
            # Lookup shape func if we can't find it
            found = []
            for type, _func in REGISTERED_TYPES:
                try:
                    if module is type or isinstance(module, type):
                        found.append(_func)
                except TypeError:
                    pass
            if len(set(found)) == 1:
                self._func = found[0]
            elif len(found) == 0:
                raise TypeError('Unknown (output_shape) module type {}'.format(module))
            else:
                raise AssertionError('Ambiguous (output_shape) module {}. Found {}'.format(module, found))

    def __call__(self, *args, **kwargs):
        if isinstance(self.module, nn.Module):
            # bound methods dont need module
            is_bound  = hasattr(self._func, '__func__') and getattr(self._func, '__func__', None) is not None
            is_bound |= hasattr(self._func, 'im_func') and getattr(self._func, 'im_func', None) is not None
            if is_bound:
                output_shape = self._func(*args, **kwargs)
            else:
                # nn.Module with state
                output_shape = self._func(self.module, *args, **kwargs)
        else:
            # a simple pytorch func
            output_shape = self._func(*args, **kwargs)

        # Package the output shape up in the appropriate wrapper class
        output_shape = OutputShape.coerce(output_shape)
        # if self.math.__name__ == 'sympy':
        #     output_shape = _simplify(output_shape)
        # debug = True
        # if debug:
        #     print('{}.output_shape = {}'.format(str(self._func.__name__), output_shape))
        return output_shape

    def _check_consistency(self, input_shape, **kwargs):
        """
        Test function to check that expected shape is equal to computed shape.
        The kwargs are passed to both output_shape_for and forward, so ensure
        that both functions accept the same arguments.
        """
        # Run the output shape computation
        expected = self(input_shape, **kwargs)

        if isinstance(expected, OutputShape):
            expected_output_shape = expected.data
        else:
            expected_output_shape = expected

        # Create dummy inputs and send them through the network
        inputs = torch.randn(input_shape)
        with torch.no_grad():
            self.module.eval()
            outputs = self.module(inputs, **kwargs)

        if isinstance(outputs, dict):
            if not isinstance(expected_output_shape, dict):
                raise AssertionError((
                    'if outputs is a dict, then output_shape must also be '
                    'a corresponding dict. Instead we got: '
                    'type(outputs)={} '
                    'type(expected_output_shape)={} '
                ).format(type(outputs), type(expected_output_shape)))
        computed_output_shape = output_shape_of(outputs)

        if computed_output_shape != expected_output_shape:
            print('expected_output_shape = {}'.format(ub.urepr(expected_output_shape, nl=0)))
            print('computed_output_shape = {}'.format(ub.urepr(computed_output_shape, nl=0)))
            raise AssertionError(
                'computed shape {!r} != expected shape {!r}'.format(
                    computed_output_shape,
                    expected_output_shape,
                )
            )
        return expected_output_shape

    @staticmethod
    @compute_type(nn.Upsample)
    def Upsample(module, input_shape):
        r"""
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
            :math:`H_{out} = floor(H_{in} * scale\_factor)`
            :math:`W_{out} = floor(W_{in}  * scale\_factor)`

        Example:
            >>> # xdoctest: +SKIP
            >>> # There is a torch bug in 1.1.0 that prevents this from working
            >>> from .analytic.output_shape_for import *
            >>> input_shape = (1, 3, 256, 256, 256)
            >>> module = nn.Upsample(scale_factor=(2, 3, 4))
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 3, 512, 768, 1024)
            >>> module = nn.Upsample(size=100)
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 3, 100, 100, 100)
            >>> input_shape = (1, 3, 256, 256)
            >>> module = nn.UpsamplingBilinear2d(scale_factor=2)
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 3, 512, 512)
        """
        math = OutputShapeFor.math
        # N, C, *DIMS_in = input_shape
        N, C = input_shape[0:2]
        DIMS_in = input_shape[2:]

        if module.size is None:
            scale_factor = ensure_iterablen(module.scale_factor, len(DIMS_in))
            int = builtins.int if math.__name__ == 'math' else ub.identity
            DIMS_out = [
                int(math.floor(D_in * scale_factor[i]))
                for i, D_in in enumerate(DIMS_in)
            ]
        else:
            DIMS_out = ensure_iterablen(module.size, len(DIMS_in))

        output_shape = SHAPE_CLS([N, C] + list(DIMS_out))
        if math.__name__ == 'sympy':
            output_shape = _simplify(output_shape)
        return output_shape

    @staticmethod
    @compute_type(torch.nn.functional.interpolate)
    def interpolate(input_shape, size=None, scale_factor=None, **kwargs):
        """
        Example:
            >>> from .analytic.output_shape_for import *
            >>> input_shape = (1, 3, 256, 256)
            >>> output_shape = OutputShapeFor(torch.nn.functional.interpolate)(input_shape, size=(32, 32))
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 3, 32, 32)
        """
        math = OutputShapeFor.math
        # N, C, *DIMS_in = input_shape
        N, C = input_shape[0:2]
        DIMS_in = input_shape[2:]

        if size is None:
            scale_factor = ensure_iterablen(scale_factor, len(DIMS_in))
            int = builtins.int if math.__name__ == 'math' else ub.identity
            DIMS_out = [
                int(math.floor(D_in * scale_factor[i]))
                for i, D_in in enumerate(DIMS_in)
            ]
        else:
            DIMS_out = ensure_iterablen(size, len(DIMS_in))

        output_shape = SHAPE_CLS([N, C] + list(DIMS_out))
        if math.__name__ == 'sympy':
            output_shape = _simplify(output_shape)
        return output_shape

    @staticmethod
    @compute_type(nn.ConvTranspose1d)
    def conv1dT(module, input_shape):
        return OutputShapeFor.convndT(module, input_shape, 1)

    @staticmethod
    @compute_type(nn.ConvTranspose2d)
    def conv2dT(module, input_shape):
        return OutputShapeFor.convndT(module, input_shape, 2)

    @staticmethod
    @compute_type(nn.ConvTranspose3d)
    def conv3dT(module, input_shape):
        return OutputShapeFor.convndT(module, input_shape, 3)

    @staticmethod
    @compute_type(nn.Conv1d)
    def conv1d(module, input_shape):
        return OutputShapeFor.convnd(module, input_shape, 1)

    @staticmethod
    @compute_type(nn.Conv2d)
    def conv2d(module, input_shape):
        return OutputShapeFor.convnd(module, input_shape, 2)

    @staticmethod
    @compute_type(nn.ZeroPad2d)
    def zeropad2d(module, input_shape):
        r"""
        Shape:
            - Input: :math:`(N, C, H_{in}, W_{in})`
            - Output: :math:`(N, C, H_{out}, W_{out})` where

              :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

              :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

        Example:
            >>> module = nn.ZeroPad2d([2, 3, 5, 7])
            >>> input_shape = (1, 3, 5, 7)
            >>> out = OutputShapeFor(module)(input_shape)
            >>> out_want = module(torch.zeros(*input_shape)).shape
            >>> assert out == tuple(out_want)
        """
        return OutputShapeFor.pad(input_shape, module.padding)

    @staticmethod
    @compute_type(F.conv2d)
    def f_conv2d(inputs, weight, bias=None, stride=1, padding=0,
                 dilation=1, groups=1):
        """
        Example:
            >>> x = inputs = (1, 124, 226, 226)
            >>> module = nn.Conv2d(128, 64, kernel_size=(3, 5), groups=8)
            >>> weight = module.weight
            >>> bias = module.bias is not None
            >>> stride = module.stride
            >>> padding = module.padding
            >>> dilation = module.dilation
            >>> groups = module.groups
            >>> y = OutputShapeFor(F.conv2d)(x, weight, bias, stride, padding,
            >>>                              dilation, groups)
            >>> print(y)
            >>> y2 = OutputShapeFor(module)(x)
            >>> assert y == y2

            >>> weight = torch.rand(3, 2, 5, 5)
            >>> OutputShapeFor(F.conv2d)((1, 3, 7, 7), weight)
        """
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        kernel = (kernel_h, kernel_w)
        module = nn.Conv2d(in_channels * groups, out_channels, kernel,
                           bias=bias, stride=stride, padding=padding,
                           dilation=dilation, groups=groups)
        return OutputShapeFor.convnd(module, inputs, 2)

    @staticmethod
    @compute_type(nn.Conv3d)
    def conv3d(module, input_shape):
        return OutputShapeFor.convnd(module, input_shape, 3)

    @staticmethod
    @compute_type(nn.MaxPool1d)
    def maxpool1d(module, input_shape):
        return OutputShapeFor.maxpoolnd(module, input_shape, 1)

    @staticmethod
    @compute_type(nn.MaxPool2d)
    def maxpool2d(module, input_shape):
        return OutputShapeFor.maxpoolnd(module, input_shape, 2)

    @staticmethod
    @compute_type(nn.MaxPool3d)
    def maxpool3d(module, input_shape):
        return OutputShapeFor.maxpoolnd(module, input_shape, 3)

    @staticmethod
    @compute_type(nn.AvgPool1d)
    def avepool1d(module, input_shape):
        return OutputShapeFor.avepoolnd(module, input_shape, 1)

    @staticmethod
    @compute_type(nn.AvgPool2d)
    def avepool2d(module, input_shape):
        return OutputShapeFor.avepoolnd(module, input_shape, 2)

    @staticmethod
    @compute_type(nn.AvgPool3d)
    def avepool3d(module, input_shape):
        return OutputShapeFor.avepoolnd(module, input_shape, 3)

    @staticmethod
    @compute_type(nn.modules.pooling._AdaptiveMaxPoolNd, nn.modules.pooling._AdaptiveAvgPoolNd)
    def adaptive_poolnd(module, input_shape):
        """
        Adaptive pooling is easy because the output-shape is known a-priori
        """
        B, C = input_shape[0:2]
        in_dims = input_shape[2:]

        n = len(in_dims)
        output_dims = ensure_iterablen(module.output_size, n)
        for i, d in enumerate(output_dims):
            if d is None:
                output_dims[i] = in_dims[i]

        output_shape = SHAPE_CLS([B, C] + list(output_dims))
        return output_shape

    @staticmethod
    def convndT(module, input_shape, n):
        r"""
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
            :math:`H_{out} = (H_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0] + output\_padding[0]`
            :math:`W_{out} = (W_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1] + output\_padding[1]`

        Example:
            >>> from .analytic.output_shape_for import *
            >>> input_shape = (1, 3, 256, 256)
            >>> module = nn.ConvTranspose2d(input_shape[1], 11, kernel_size=2, stride=2)
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 11, 512, 512)

        Example:
            >>> from .analytic.output_shape_for import *
            >>> input_shape = (1, 3, 25, 32, 32)
            >>> module = nn.Conv3d(in_channels=input_shape[1], out_channels=11,
            >>>                    kernel_size=(3, 3, 3), stride=1, padding=0,
            >>>                    dilation=1, groups=1, bias=True)
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 11, 23, 30, 30)
        """
        # N, C_in, *DIMS_in = input_shape
        N, C_in = input_shape[0:2]
        DIMS_in = input_shape[2:]

        if len(DIMS_in) != n:
            raise ValueError('must have {} dims, but got {} '.format(n, len(DIMS_in)))

        C_out = module.out_channels
        stride = module.stride
        kernel_size = module.kernel_size
        output_padding = module.output_padding
        dilation = module.dilation

        padding = module.padding
        DIMS_out = [
            # Fix the docs: https://github.com/pytorch/pytorch/issues/14099
            (D_in - 1) * stride[i] - 2 * padding[i] + (kernel_size[i] - 1) * dilation[i] + output_padding[i] + 1
            for i, D_in in enumerate(DIMS_in)
        ]
        output_shape = SHAPE_CLS([N, C_out] + DIMS_out)
        if math.__name__ == 'sympy':
            output_shape = _simplify(output_shape)
        return output_shape

    @staticmethod
    def convnd(module, input_shape, n):
        r"""
        Notes:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
                :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
                :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`

        Example:
            >>> from .analytic.output_shape_for import *
            >>> input_shape = (1, 3, 256, 256)
            >>> module = nn.Conv2d(input_shape[1], 11, 3, 1, 0)
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 11, 254, 254)
        """
        math = OutputShapeFor.math
        # N, C_in, *DIMS_in = input_shape
        N, C_in = input_shape[0:2]
        DIMS_in = input_shape[2:]

        if len(DIMS_in) != n:
            raise ValueError('must have {} dims, but got {} '.format(n, len(DIMS_in)))

        C_out = module.out_channels
        padding = module.padding
        stride = module.stride
        dilation = module.dilation
        kernel_size = module.kernel_size

        int = builtins.int if math.__name__ == 'math' else ub.identity
        DIMS_out = [
            int(math.floor(
                (D_in + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1
            ))
            for i, D_in in enumerate(DIMS_in)
        ]
        output_shape = SHAPE_CLS([N, C_out] + DIMS_out)
        if math.__name__ == 'sympy':
            output_shape = _simplify(output_shape)
        return output_shape

    @staticmethod
    def maxpoolnd(module, input_shape, n):
        r"""
        CommandLine:
            python -m xdoctest netharn.analytic.output_shape_for OutputShapeFor.maxpoolnd:0

        Example:
            >>> from .analytic.output_shape_for import *
            >>> input_shape = (1, 3, 256, 256)
            >>> module = nn.MaxPool2d(kernel_size=2, stride=2)
            >>> output_shape = tuple(OutputShapeFor(module)(input_shape))
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 3, 128, 128)

        Example:
            >>> from .analytic.output_shape_for import *
            >>> input_shape = (1, 512, 37, 37)
            >>> module = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
            >>> output_shape = tuple(OutputShapeFor(module)(input_shape))
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 512, 19, 19)

        Shape:
            2d Case:
            Same as conv2 forumla except C2 = C1
            - Input: :math:`(N, C, H_{in}, W_{in})`
            - Output: :math:`(N, C, H_{out}, W_{out})` where
            :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
            :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
        """
        math = OutputShapeFor.math
        # N, C, *DIMS_in = input_shape
        N, C = input_shape[0:2]
        DIMS_in = input_shape[2:]

        padding = ensure_iterablen(module.padding, n)
        stride = ensure_iterablen(module.stride, n)
        dilation = ensure_iterablen(module.dilation, n)
        kernel_size = ensure_iterablen(module.kernel_size, n)

        trunc = math.ceil if module.ceil_mode else math.floor

        int = builtins.int if math.__name__ == 'math' else ub.identity

        DIMS_out = [
            int(trunc((D_in  + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1))
            for i, D_in in enumerate(DIMS_in)
        ]
        output_shape = SHAPE_CLS([N, C] + DIMS_out)
        if math.__name__ == 'sympy':
            output_shape = _simplify(output_shape)
        return output_shape

    @staticmethod
    def avepoolnd(module, input_shape, n):
        r"""
        2D case:
          Shape:
              - Input: :math:`(N, C, H_{in}, W_{in})`
              - Output: :math:`(N, C, H_{out}, W_{out})` where
                :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - kernel\_size[0]) / stride[0] + 1)`
                :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - kernel\_size[1]) / stride[1] + 1)`
        """
        math = OutputShapeFor.math
        # N, C, *DIMS_in = input_shape
        N, C = input_shape[0:2]
        DIMS_in = input_shape[2:]

        padding = ensure_iterablen(module.padding, n)
        stride = ensure_iterablen(module.stride, n)
        kernel_size = ensure_iterablen(module.kernel_size, n)

        int = builtins.int if math.__name__ == 'math' else ub.identity

        DIMS_out = [
            int(math.floor((D_in + 2 * padding[i] - kernel_size[i]) / stride[i] + 1))
            for i, D_in in enumerate(DIMS_in)
        ]
        output_shape = SHAPE_CLS([N, C] + DIMS_out)
        if math.__name__ == 'sympy':
            output_shape = _simplify(output_shape)
        return output_shape

    @staticmethod
    @compute_type(nn.Linear)
    def linear(module, input_shape):
        r"""
           Shape:
               - Input: :math:`(N, *, in\_features)` where `*` means any number of
                 additional dimensions
               - Output: :math:`(N, *, out\_features)` where all but the last dimension
                 are the same shape as the input.
        """
        # N, *other, in_feat = input_shape
        N = input_shape[0]
        other = input_shape[1:-1]
        in_feat = input_shape[-1]  # NOQA

        output_shape = [N] + list(other) + [module.out_features]
        return SHAPE_CLS(output_shape)

    @staticmethod
    def identity(input_shape):
        return SHAPE_CLS(input_shape)

    @staticmethod
    @compute_type(nn.functional.relu)
    def relu_func(input_shape):
        return SHAPE_CLS(input_shape)

    @staticmethod
    @compute_type(nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.modules.normalization.GroupNorm,
                  nn.modules.normalization.LocalResponseNorm,
                  nn.modules.normalization.LayerNorm, nn.CrossMapLRN2d,
                  nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    def normalization(module, input_shape):
        """
            import redbaron
            import torch
            source = open(torch.nn.modules.instancenorm.__file__, 'r').read()
            baron = redbaron.RedBaron(source)
            classes = [item.name for item in baron if item.type == 'class']
            print(', '.join(['nn.{}'.format(c) for c in classes]))

            source = open(torch.nn.modules.normalization.__file__, 'r').read()
            baron = redbaron.RedBaron(source)
            classes = [item.name for item in baron if item.type == 'class']
            print(', '.join(['nn.{}'.format(c) for c in classes]))
        """
        return OutputShapeFor.identity(input_shape)

    @staticmethod
    @compute_type(nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout,
                  nn.FeatureAlphaDropout)
    def dropout(module, input_shape):
        return OutputShapeFor.identity(input_shape)

    @staticmethod
    @compute_type(nn.Threshold, nn.RReLU, nn.Hardtanh, nn.ReLU6, nn.ReLU,
                  nn.Sigmoid, nn.Tanh, nn.ELU, nn.CELU, nn.SELU, nn.GLU,
                  nn.Hardshrink, nn.LeakyReLU, nn.LogSigmoid, nn.Softplus,
                  nn.Softshrink, nn.PReLU, nn.Softsign, nn.Tanhshrink,
                  nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax)
    def nonlinearity(module, input_shape):
        r"""
        Ignore:
            import redbaron
            import torch
            source = open(torch.nn.modules.activation.__file__, 'r').read()
            baron = redbaron.RedBaron(source)
            classes = [item.name for item in baron if item.type == 'class']
            print(', '.join(['nn.{}'.format(c) for c in classes]))
        """
        return OutputShapeFor.identity(input_shape)

    @staticmethod
    @compute_type(nn.Sequential)
    def sequential(module, input_shape):
        """
        CommandLine:
            xdoctest -m netharn.analytic.output_shape_for OutputShapeFor.sequential

        Example:
            >>> from .analytic.output_shape_for import *
            >>> self = nn.Sequential(
            >>>     nn.Conv2d(2, 3, kernel_size=3),
            >>>     nn.Conv2d(3, 5, kernel_size=3),
            >>>     nn.Conv2d(5, 7, kernel_size=3),
            >>> )
            >>> shape = OutputShapeFor(self)([1, 1, 7, 11])
            >>> print('shape = {}'.format(ub.urepr(shape, nl=0)))
            >>> print('shape.hidden = {}'.format(ub.urepr(shape.hidden, nl=1)))
            shape = (1, 7, 1, 5)
            shape.hidden = {
                '0': (1, 3, 5, 9),
                '1': (1, 5, 3, 7),
                '2': (1, 7, 1, 5),
            }
        """
        hidden = HiddenShapes()
        shape = input_shape
        for key, child in module._modules.items():
            hidden[key] = shape = OutputShapeFor(child)(shape)
        shape = OutputShape.coerce(shape, hidden=hidden)
        return shape

    @staticmethod
    @compute_type(torchvision.models.resnet.BasicBlock)
    def resent_basic_block(module, input_shape):
        residual_shape = input_shape
        shape = input_shape

        hidden = HiddenShapes()
        hidden['conv1'] = shape = OutputShapeFor(module.conv1)(shape)
        hidden['bn1']   = shape = OutputShapeFor(module.bn1)(shape)
        hidden['relu1'] = shape = OutputShapeFor(module.relu)(shape)

        hidden['conv2'] = shape = OutputShapeFor(module.conv2)(shape)
        hidden['bn2']   = shape = OutputShapeFor(module.bn2)(shape)
        hidden['relu2'] = shape = OutputShapeFor(module.relu)(shape)

        if module.downsample is not None:
            residual_shape = OutputShapeFor(module.downsample)(residual_shape)
            hidden['residual'] = residual_shape

        hidden['join'] = shape
        assert residual_shape[-2:] == shape[-2:], (
            'cannot add residual {} {}'.format(residual_shape, shape))
        shape = OutputShapeFor(module.relu)(shape)
        hidden['relu3'] = shape
        shape = OutputShape.coerce(shape, hidden=hidden)
        return shape

    @staticmethod
    @compute_type(torchvision.models.resnet.Bottleneck)
    def resent_bottleneck(module, input_shape):
        residual_shape = input_shape
        shape = input_shape

        hidden = HiddenShapes()
        hidden['conv1'] = shape = OutputShapeFor(module.conv1)(shape)
        hidden['bn1']   = shape = OutputShapeFor(module.bn1)(shape)
        hidden['relu1'] = shape = OutputShapeFor(module.relu)(shape)

        hidden['conv2'] = shape = OutputShapeFor(module.conv2)(shape)
        hidden['bn2']   = shape = OutputShapeFor(module.bn2)(shape)
        hidden['relu2'] = shape = OutputShapeFor(module.relu)(shape)

        hidden['conv3'] = shape = OutputShapeFor(module.conv3)(shape)
        hidden['bn3']   = shape = OutputShapeFor(module.bn3)(shape)

        if module.downsample is not None:
            residual_shape = OutputShapeFor(module.downsample)(input_shape)
            hidden['residual'] = residual_shape

        assert residual_shape[-2:] == shape[-2:], (
            'cannot add residual {} {}'.format(residual_shape, shape))
        hidden['join'] = shape

        shape = OutputShapeFor(module.relu)(shape)
        hidden['relu3'] = shape

        shape = OutputShape.coerce(shape, hidden=hidden)
        return shape

    @staticmethod
    @compute_type(torchvision.models.resnet.ResNet)
    def resnet_model(module, input_shape):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> from .analytic.output_shape_for import *
            >>> module = torchvision.models.resnet50()
            >>> input_shape = (1, 3, 224, 224)
            >>> shape = OutputShapeFor(module)(input_shape=input_shape)
            >>> print(ub.urepr(shape.hidden, nl=-1))
        """
        shape = input_shape

        hidden = HiddenShapes()
        hidden['conv1'] = shape = OutputShapeFor(module.conv1)(shape)
        hidden['bn1'] = shape = OutputShapeFor(module.bn1)(shape)
        hidden['relu1'] = shape = OutputShapeFor(module.relu)(shape)
        hidden['maxpool'] = shape = OutputShapeFor(module.maxpool)(shape)

        hidden['layer1'] = shape = OutputShapeFor(module.layer1)(shape)
        hidden['layer2'] = shape = OutputShapeFor(module.layer2)(shape)
        hidden['layer3'] = shape = OutputShapeFor(module.layer3)(shape)
        hidden['layer4'] = shape = OutputShapeFor(module.layer4)(shape)

        hidden['avgpool'] = shape = OutputShapeFor(module.avgpool)(shape)

        def prod(args):
            result = args[0]
            for arg in args[1:]:
                result = result * arg
            return result
        shape = (shape[0], prod(shape[1:]))
        hidden['view'] = shape

        hidden['fc'] = shape = OutputShapeFor(module.fc)(shape)
        shape = OutputShape.coerce(shape, hidden=hidden)
        return shape

    @staticmethod
    @compute_type(nn.functional.adaptive_avg_pool2d)
    def adaptive_poolnd_func(input_shape, output_shape):
        """
        Adaptive pooling is easy because the output-shape is known a-priori

        Example:
            >>> from .analytic.output_shape_for import *
            >>> input_shape = (1, 3, 256, 256)
            >>> output_shape = (7, 7)
            >>> output_shape_ = OutputShapeFor(nn.functional.adaptive_avg_pool2d)(input_shape, output_shape)
            >>> print('output_shape = {!r}'.format(output_shape_))
            output_shape = (1, 3, 7, 7)
        """
        B, C = input_shape[0:2]
        in_dims = input_shape[2:]

        n = len(in_dims)
        output_dims = ensure_iterablen(output_shape, n)
        for i, d in enumerate(output_dims):
            if d is None:
                output_dims[i] = in_dims[i]

        output_shape_ = SHAPE_CLS([B, C] + list(output_dims))
        return output_shape_

    @staticmethod
    @compute_type(torch.sigmoid)
    def sigmoid(input_shape):
        return OutputShapeFor.identity(input_shape)

    @staticmethod
    @compute_type(F.pad)
    def pad(x, pad, mode='constant', value=0):
        """
        Example:
            >>> t4d = x = (3, 3, 4, 2)
            >>> pad = p1d = (1, 1)
            >>> out = OutputShapeFor(F.pad)(x, pad)
            >>> print(out)
            (3, 3, 4, 4)
            >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
            >>> out = OutputShapeFor.pad(t4d, p2d, "constant", 0)
            >>> print(out)
            (3, 3, 8, 4)
            >>> t4d = (3, 3, 4, 2)
            >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
            >>> out = OutputShapeFor.pad(t4d, p3d, "constant", 0)
            >>> print(out)
            (3, 9, 7, 3)
        """
        new_x = list(x)
        dim = len(new_x)
        for idx, dpad in enumerate(ub.chunks(pad, 2), start=1):
            dimx = dim - idx
            lpad, rpad = dpad
            new_x[dimx] = x[dimx] + lpad + rpad
        out = SHAPE_CLS(new_x)
        return out

    @staticmethod
    @compute_type(torch.cat)
    def cat(input_shapes, dim=0):
        """
        Example:
            >>> from .analytic.output_shape_for import *
            >>> input_shape1 = (1, 3, 256, 256)
            >>> input_shape2 = (1, 4, 256, 256)
            >>> input_shapes = [input_shape1, input_shape2]
            >>> output_shape = OutputShapeFor(torch.cat)(input_shapes, dim=1)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 7, 256, 256)
        """
        n_dims = max(map(len, input_shapes))
        assert n_dims == min(map(len, input_shapes))
        output_shape = [None] * n_dims
        for shape in input_shapes:
            for i, v in enumerate(shape):
                if output_shape[i] is None:
                    output_shape[i] = v
                else:
                    if i == dim:
                        output_shape[i] += v
                    else:
                        assert output_shape[i] == v, 'inconsistent dims {}'.format(input_shapes)
        return SHAPE_CLS(output_shape)

    @staticmethod
    @compute_type(DataSerial)
    def data_serial(module, *args, **kw):
        return OutputShapeFor(module.module)(*args, **kw)

    @staticmethod
    @compute_type(torch.nn.DataParallel)
    def data_parallel(module, *args, **kw):
        return OutputShapeFor(module.module)(*args, **kw)

    @staticmethod
    def getitem(arr):
        """
        Wraps getitem calls

        Example:
            >>> arr = (2, 32, 9, 9)
            >>> result = OutputShapeFor.getitem(arr)[:, 0:4]
            >>> assert result == [2, 4, 9, 9]
        """
        return _ShapeGetItem(arr)

    @staticmethod
    def view(arr, *args):
        """
        Wraps view calls

        Example:
            >>> arr = (2, 32, 9, 9)
            >>> result = OutputShapeFor.view(arr, -1)
            >>> assert result == (5184,)
        """
        from .. import layers
        reshape = layers.Reshape(*args)
        return reshape.output_shape_for(arr)

    @staticmethod
    def shape(arr):
        """
        Wraps shape calls

        Example:
            >>> arr = (2, 32, 9, 9)
            >>> result = OutputShapeFor.shape(arr)
            >>> assert result == arr
        """
        return arr

    @staticmethod
    def add(arr1, arr2):
        return _output_shape_broadcast(arr1, arr2)

    @staticmethod
    def mul(arr1, arr2):
        return _output_shape_broadcast(arr1, arr2)

    @staticmethod
    def sub(arr1, arr2):
        return _output_shape_broadcast(arr1, arr2)

    @staticmethod
    def div(arr1, arr2):
        return _output_shape_broadcast(arr1, arr2)


def _output_shape_broadcast(arr1, arr2):
    """
    Args:
        arr1 (Tuple | scalar): shape of arr1 or a scalar
        arr2 (Tuple | scalar): shape of arr2 or a scalar
    """
    if not ub.iterable(arr1):
        return arr2
    if not ub.iterable(arr2):
        return arr1
    if tuple(arr1) != tuple(arr2):

        if len(arr1) == len(arr2):
            arr3 = []
            for d1, d2 in zip(arr1, arr2):
                if d1 is None or d1 < 0:
                    raise NotImplementedError
                if d2 is None or d2 < 0:
                    raise NotImplementedError
                if d1 == d2:
                    arr3.append(d1)
                elif d1 == 1:
                    arr3.append(d2)
                elif d2 == 1:
                    arr3.append(d1)
                else:
                    raise ValueError('broadcast seems bad')
            arr3 = type(arr1)(arr3)
            return arr3

        # TODO: handle broadcast
        raise NotImplementedError('Full broadcast not implemented {} != {}'.format(arr1, arr2))
    return arr1


class _ShapeGetItem(object):
    def __init__(self, inp):
        self.inp = inp

    def __getitem__(self, slices):
        ellipsis_type = type(Ellipsis)
        oup = list(self.inp)
        if isinstance(slices, slice):
            slices = (slices,)

        if isinstance(slices, tuple):
            for i, sl in enumerate(slices):
                if isinstance(sl, ellipsis_type):
                    assert i == len(slices) - 1
                    break
                start, stop, step = sl.indices(oup[i])
                oup[i] = (stop - start) // step
        return oup


def ensure_iterablen(scalar, n):
    try:
        iter(scalar)
    except TypeError:
        return [scalar] * n
    return scalar
