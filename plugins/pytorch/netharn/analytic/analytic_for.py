"""
Code for commonalities between "X for" objects that compute analytic properties
of networks like OutputShapeFor and ReceptiveFieldFor


The purpose of analysic modules is to make it easy to introspect both the final
and intermediate tensor shapes and receptive fields. As long as the relevant
``output_shape_for`` ``receptive_field_for`` OR ``_analytic_forward`` methods
are defined the computation will be fully symbolic. SeeAlso
:class:`netharn.layers.AnalyticModule`.


Example:
    >>> import torch
    >>> from viame.pytorch import netharn as nh
    >>> # Inheriting from nh.layers.AnalyticModule lets us define _analytic_forward
    >>> class MyNetwork(nh.layers.AnalyticModule):
    >>>     def __init__(self, classes):
    >>>         super().__init__()
    >>>         self.classes = classes
    >>>         # Note we are just using regular torch layers here
    >>>         # No special tricks required as long as the computation for
    >>>         # receptive field / output shape is registered.
    >>>         self.backbone = torch.nn.Sequential(*[
    >>>             torch.nn.Conv2d(3, 32, kernel_size=3),
    >>>             torch.nn.BatchNorm2d(32),
    >>>             torch.nn.MaxPool2d(2, stride=2),
    >>>             torch.nn.ReLU(),
    >>>             torch.nn.Conv2d(32, 256, kernel_size=3, stride=2),
    >>>             torch.nn.BatchNorm2d(256),
    >>>         ])
    >>>         self.clf_head = torch.nn.Conv2d(256, len(self.classes), kernel_size=1)
    >>>     def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
    >>>                       **kwargs):
    >>>         # Defining the analytic forward function and using the _OutputFor
    >>>         # wrappers instead of calling each module directly will
    >>>         # automatically define the symbolic computation for
    >>>         # output_shape_for, receptive_field_for, and the real
    >>>         # computation for forward. Using Hidden will track any
    >>>         # intermediate states.
    >>>         x = inputs
    >>>         hidden = _Hidden()
    >>>         x = hidden['backbone'] = _OutputFor(self.backbone)(x)
    >>>         x = hidden['clf_head'] = _OutputFor(self.clf_head)(x)
    >>>         outputs = {
    >>>             'class_energy': x,
    >>>         }
    >>>         outputs = _Output.coerce(outputs, hidden)
    >>>         return outputs
    >>> # We can create an instance of our network
    >>> self = MyNetwork(['a', 'b'])
    >>> # Asking about the output shape for any input shape is computed
    >>> # without directly invoking any tensor operations.
    >>> output_shape = self.output_shape_for((None, 3, 32, 32))
    >>> print('output_shape = {!r}'.format(output_shape))
    >>> print(ub.urepr(output_shape.hidden, nl=-1))
    output_shape = OutputShapeDict([('class_energy', (None, 2, 7, 7))])
    {
        'backbone': {
            '0': (None, 32, 30, 30),
            '1': (None, 32, 30, 30),
            '2': (None, 32, 15, 15),
            '3': (None, 32, 15, 15),
            '4': (None, 256, 7, 7),
            '5': (None, 256, 7, 7)
        },
        'clf_head': (None, 2, 7, 7)
    }
    >>> # In most cases the receptive field does not need to know about the
    >>> # input shape (adaptive layers are the exception here)
    >>> rf = self.receptive_field_for()
    >>> print('rf = {}'.format(ub.urepr(rf, nl=2)))
    >>> print(ub.urepr(rf.hidden, nl=3))
    rf = {
        'class_energy': {
            'crop': np.array([3.5, 3.5], dtype=np.float64),
            'shape': np.array([8., 8.], dtype=np.float64),
            'stride': np.array([4., 4.], dtype=np.float64),
        },
    }
    {
        'backbone': {
            '0': {
                'crop': np.array([1., 1.], dtype=np.float64),
                'shape': np.array([3., 3.], dtype=np.float64),
                'stride': np.array([1., 1.], dtype=np.float64),
            },
            '1': {
                'crop': np.array([1., 1.], dtype=np.float64),
                'shape': np.array([3., 3.], dtype=np.float64),
                'stride': np.array([1., 1.], dtype=np.float64),
            },
            '2': {
                'crop': np.array([1.5, 1.5], dtype=np.float64),
                'shape': np.array([4., 4.], dtype=np.float64),
                'stride': np.array([2., 2.], dtype=np.float64),
            },
            '3': {
                'crop': np.array([1.5, 1.5], dtype=np.float64),
                'shape': np.array([4., 4.], dtype=np.float64),
                'stride': np.array([2., 2.], dtype=np.float64),
            },
            '4': {
                'crop': np.array([3.5, 3.5], dtype=np.float64),
                'shape': np.array([8., 8.], dtype=np.float64),
                'stride': np.array([4., 4.], dtype=np.float64),
            },
            '5': {
                'crop': np.array([3.5, 3.5], dtype=np.float64),
                'shape': np.array([8., 8.], dtype=np.float64),
                'stride': np.array([4., 4.], dtype=np.float64),
            },
        },
        'clf_head': {
            'crop': np.array([3.5, 3.5], dtype=np.float64),
            'shape': np.array([8., 8.], dtype=np.float64),
            'stride': np.array([4., 4.], dtype=np.float64),
        },
    }
    >>> # analytic forward ensures that your forward definition is consistent
    >>> # with output_shape_for and analytic_for
    >>> inputs = torch.rand(1, 3, 32, 32)
    >>> outputs = self.forward(inputs)
    >>> print('class_energy = {}'.format(outputs['class_energy'].shape))
    class_energy = torch.Size([1, 2, 7, 7])
"""
import ubelt as ub
from collections import OrderedDict


class Hidden(OrderedDict, ub.NiceRepr):
    """
    Object for storing hidden states of analystic computation

    Example:
        hidden0 = Hidden()
        hidden1 = Hidden()
        from .analytic.output_shape_for import OutputShape
        hidden0['a'] = (1, 2, 3)
        hidden0['b'] = (1, 2,)
        # Note that the **last** value _currently_ (may change)
        # needs to be the same as what ever the real output is
        hidden0['c'] = (1,)
        output0 = OutputShape.coerce((1,), hidden=hidden0)
        hidden1['c'] = (1, 2, 3)
        hidden1['d'] = output0
        print(ub.urepr(hidden1.shallow(2), nl=-1))
    """

    def __nice__(self):
        return ub.urepr(self, nl=0)

    def __str__(self):
        return ub.NiceRepr.__str__(self)

    def __repr__(self):
        return ub.NiceRepr.__repr__(self)

    def __setitem__(self, key, value):
        if getattr(value, 'hidden', None) is not None:
            # When setting a value to an OutputShape object, if that object has
            # a hidden shape, then use that instead.
            value = value.hidden
        return OrderedDict.__setitem__(self, key, value)

    def shallow(self, n=1):
        """
        Grabs only the shallowest n layers of hidden shapes
        """
        if n == 0:
            last = self
            while hasattr(last, 'shallow'):
                values = list(last.values())
                if len(values):
                    last = values[-1]
                else:
                    break
            return last
        else:
            output = OrderedDict()
            for key, value in self.items():
                # if isinstance(value, HiddenShapes):
                if hasattr(value, 'shallow'):
                    value = value.shallow(n - 1)
                output[key] = value
            return output


class OutputFor(object):
    """
    Analytic base / identity class
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)


class Output(object):
    """
    Analytic base / identity class
    """
    @classmethod
    def coerce(cls, data=None, hidden=None):
        return data


class ForwardFor(OutputFor):
    """
    Analytic version of forward functions
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)

    @staticmethod
    def getitem(arr):
        """
        Wraps getitem calls

        Example:
            >>> import torch
            >>> arr = torch.rand(2, 16, 2, 2)
            >>> result = ForwardFor.getitem(arr)[:, 0:4]
            >>> assert result.shape == (2, 4, 2, 2)
        """
        return _ForwardGetItem(arr)

    @staticmethod
    def view(arr, *args):
        """
        Wraps view calls

        Example:
            >>> import torch
            >>> arr = torch.rand(2, 16, 2, 2)
            >>> result = ForwardFor.view(arr, -1)
        """
        return arr.view(*args)

    @staticmethod
    def shape(arr):
        """
        Wraps shape calls

        Example:
            >>> import torch
            >>> arr = torch.rand(2, 16, 2, 2)
            >>> result = ForwardFor.shape(arr)
        """
        return arr.shape

    @staticmethod
    def add(arr1, arr2):
        return arr1 + arr2

    @staticmethod
    def mul(arr1, arr2):
        return arr1 * arr2

    @staticmethod
    def sub(arr1, arr2):
        return arr1 - arr2

    @staticmethod
    def div(arr1, arr2):
        return arr1 - arr2


class _ForwardGetItem(object):
    def __init__(self, inp):
        self.inp = inp

    def __getitem__(self, slices):
        return self.inp.__getitem__(slices)
