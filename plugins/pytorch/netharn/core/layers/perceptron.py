from .layers import common
from .layers import rectify
from .layers import conv_norm
import numpy as np


class MultiLayerPerceptronNd(common.AnalyticModule):
    """
    A multi-layer perceptron network for n dimensional data

    Choose the number and size of the hidden layers, number of output channels,
    wheather to user residual connections or not, nonlinearity, normalization,
    dropout, and more.

    Args:
        dim (int): specify if the data is 0, 1, 2, 3, or 4 dimensional.

        in_channels (int): number of input channels

        hidden_channels (List[int]): or an int specifying the number of hidden
            layers (we choose the channel size to linearly interpolate between
            input and output channels)

        out_channels (int): number of output channels

        dropout (float, default=None): amount of dropout to use between 0 and 1

        norm (str, default='batch'): type of normalization layer
            (e.g. batch or group), set to None for no normalization.

        noli (str, default='relu'): type of nonlinearity

        residual (bool, default=False):
            if true includes a resitual skip connection between inputs and
            outputs.

        norm_output (bool, default=True):
            if True, applies a final normalization layer to the output.

        noli_output (bool, default=True):
            if True, applies a final nonlineary to the output.

        standardize_weights (bool, default=False):
            Use weight standardization

    CommandLine:
        xdoctest -m netharn.layers.perceptron MultiLayerPerceptronNd:0

    Example:
        >>> from .layers.perceptron import *
        >>> kw = {'dim': 0, 'in_channels': 2, 'out_channels': 1}
        >>> model0 = MultiLayerPerceptronNd(hidden_channels=0, **kw)
        >>> model1 = MultiLayerPerceptronNd(hidden_channels=1, **kw)
        >>> model2 = MultiLayerPerceptronNd(hidden_channels=2, **kw)
        >>> print('model0 = {!r}'.format(model0))
        >>> print('model1 = {!r}'.format(model1))
        >>> print('model2 = {!r}'.format(model2))

        >>> from .layers.perceptron import *
        >>> kw = {'dim': 0, 'in_channels': 2, 'out_channels': 1, 'residual': True}
        >>> model0 = MultiLayerPerceptronNd(hidden_channels=0, **kw)
        >>> model1 = MultiLayerPerceptronNd(hidden_channels=1, **kw)
        >>> model2 = MultiLayerPerceptronNd(hidden_channels=2, **kw)
        >>> print('model0 = {!r}'.format(model0))
        >>> print('model1 = {!r}'.format(model1))
        >>> print('model2 = {!r}'.format(model2))

    Example:
        >>> from .layers.perceptron import *
        >>> import ubelt as ub
        >>> self = MultiLayerPerceptronNd(dim=1, in_channels=128, hidden_channels=3, out_channels=2)
        >>> print(self)
        MultiLayerPerceptronNd...
        >>> shape = self.output_shape_for([1, 128, 7])
        >>> print('shape = {!r}'.format(shape))
        >>> print('shape.hidden = {}'.format(ub.repr2(shape.hidden, nl=2)))
        shape.hidden = {
            'hidden': {
                'hidden0': {'conv': (1, 96, 7), 'norm': (1, 96, 7), 'noli': (1, 96, 7)},
                'hidden1': {'conv': (1, 65, 7), 'norm': (1, 65, 7), 'noli': (1, 65, 7)},
                'hidden2': {'conv': (1, 34, 7), 'norm': (1, 34, 7), 'noli': (1, 34, 7)},
                'output': (1, 2, 7),
            },
        }
        >>> from viame.arrows.pytorch.netharn import core as nh
        >>> nh.OutputShapeFor(self)._check_consistency([1, 128, 7])
        (1, 2, 7)
        >>> print('self._hidden_channels = {!r}'.format(self._hidden_channels))

    Example:
        >>> from .layers.perceptron import *
        >>> import ubelt as ub
        >>> self = MultiLayerPerceptronNd(0, 128, [256, 64], residual=True,
        >>>                               norm='group', out_channels=2)
        >>> print(self)
        >>> input_shape = (None, 128)
        >>> print(ub.repr2(self.output_shape_for(input_shape).hidden, nl=-1))

    Example:
        >>> from .layers.perceptron import *
        >>> import ubelt as ub
        >>> self = MultiLayerPerceptronNd(0, 128, [], residual=False,
        >>>                               norm='group', out_channels=2)
        >>> print(self)
        >>> input_shape = (None, 128)
        >>> print(ub.repr2(self.output_shape_for(input_shape).hidden, nl=-1))
    """
    def __init__(self, dim, in_channels, hidden_channels, out_channels,
                 bias=True, dropout=None, noli='relu', norm='batch',
                 residual=False, noli_output=False, norm_output=False,
                 standardize_weights=False):

        super(MultiLayerPerceptronNd, self).__init__()
        dropout_cls = rectify.rectify_dropout(dim)
        conv_cls = rectify.rectify_conv(dim=dim)
        curr_in = in_channels

        if isinstance(hidden_channels, int):
            n = hidden_channels
            hidden_channels = np.linspace(in_channels, out_channels, n + 1,
                                          endpoint=False)[1:]
            hidden_channels = hidden_channels.round().astype(int).tolist()
        self._hidden_channels = hidden_channels

        hidden = self.hidden = common.Sequential()
        for i, curr_out in enumerate(hidden_channels):
            layer = conv_norm.ConvNormNd(
                dim, curr_in, curr_out, kernel_size=1, bias=bias, noli=noli,
                norm=norm, standardize_weights=standardize_weights)
            hidden.add_module('hidden{}'.format(i), layer)
            if dropout is not None:
                hidden.add_module('dropout{}'.format(i), dropout_cls(p=dropout))
            curr_in = curr_out

        outkw = {'bias': bias, 'kernel_size': 1}
        self.hidden.add_module(
            'output', conv_cls(curr_in, out_channels, **outkw))

        if residual:
            if in_channels == out_channels:
                self.skip = common.Identity()
            else:
                self.skip = conv_cls(in_channels, out_channels, **outkw)
        else:
            self.skip = None

        if norm_output:
            self.final_norm = rectify.rectify_normalizer(out_channels, norm, dim=dim)
        else:
            self.final_norm = None

        if noli_output:
            self.final_noli = rectify.rectify_nonlinearity(noli, dim=dim)
        else:
            self.final_noli = None

        self.norm_output = norm_output
        self.noli_output = noli_output
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inputs):
        outputs = self.hidden(inputs)

        if self.skip is not None:
            projected = self.skip(inputs)
            outputs = projected + outputs

        if self.final_norm is not None:
            outputs = self.final_norm(outputs)

        if self.final_noli is not None:
            outputs = self.final_noli(outputs)

        return outputs

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Alternative implementation of forward using analytic functions
        for fast computation of ouptut shape / receptive field.

        The actual forward pass is written explictly above, but this should
        follow the same structure.

        Example:
            >>> from .layers.perceptron import *  # NOQA
            >>> self = MultiLayerPerceptronNd(
            >>>     1, 128, 3, 2, residual=True, noli_output=True, norm_output=True)
            >>> print('self = {!r}'.format(self))
            >>> output_shape = self.output_shape_for( (1, 128, 10))
            >>> print('output_shape = {}'.format(output_shape))
            >>> import ubelt as ub
            >>> print('{}'.format(ub.repr2(output_shape.hidden.shallow(4), nl=-1)))

            >>> receptive_field = self.receptive_field_for()
            >>> print('receptive_field = {}'.format(ub.repr2(receptive_field, nl=1)))
            >>> import ubelt as ub
            >>> print('{}'.format(ub.repr2(receptive_field.hidden.shallow(2), nl=3)))

        Ignore:
            >>> from .layers.perceptron import *  # NOQA
            >>> self = MultiLayerPerceptronNd(
            >>>     1, 128, 3, 2, residual=True, noli_output=True)
            >>> globals().update(self._analytic_shape_kw())
            >>> inputs = (1, 128, 10)
        """
        hidden = _Hidden()

        outputs = hidden['hidden'] = _OutputFor(self.hidden)(inputs)
        if self.skip is not None:
            projected = hidden['skip'] = _OutputFor(self.skip)(outputs)
            outputs = _OutputFor.add(outputs, projected)

        if self.final_norm is not None:
            outputs = hidden['final_norm'] = _OutputFor(self.final_norm)(outputs)

        if self.final_noli is not None:
            outputs = hidden['final_noli'] = _OutputFor(self.final_noli)(outputs)

        out = _Output.coerce(outputs, hidden)
        return out


# TODO:
# class PerceptronChainNd(common.AnalyticModule):
#     def __init__(self):
#         num_layers = 4
#         in_channels = 2
#         out_channels = 2
#         curr_in = in_channels
#         transition_channels = 128
#         dim = 2
#         layers = []
#         for layer_idx in range(num_layers):
#             layer = MultiLayerPerceptronNd(
#                 dim=dim,
#                 in_channels=curr_in,
#                 hidden_channels=[32],
#                 out_channels=transition_channels, residual=True)
#             curr_in = layer.out_channels
#             layers.append(layer)
