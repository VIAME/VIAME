import numpy as np
from viame.pytorch.netharn import layers
import torch
import torchvision
import ubelt as ub  # NOQA


class DescriptorNetwork(layers.Module):
    """
    Produces resnet50 + MLP descriptors

    Args:
        input_shape (Tuple): expected shape of inputs
        desc_size (int): output descriptor dimensionality
        hidden_channels (List[int]): number of channels in hidden layer
        out_channels (int): number of output channels / classes
        dropout (float, default=0): amount of dropout to use
        norm (str, default='batch'): type of normalization layer (e.g. batch or group)
        noli (str, default='relu'): type of nonlinearity
        residual (bool, default=False):
            if true includes a resitual skip connection between inputs and
            outputs.

    CommandLine:
        xdoctest -m ~/code/netharn/netharn/models/descriptor_network.py DescriptorNetwork --gpu
        xdoctest -m ~/code/netharn/netharn/models/descriptor_network.py DescriptorNetwork:0

    Example:
        >>> from .models.descriptor_network import *
        >>> input_shape = (4, 512, 32, 32)
        >>> self = DescriptorNetwork(input_shape=input_shape)
        >>> print(self)
        >>> shape = self.output_shape_for(input_shape)
        >>> print(ub.urepr(shape.hidden.shallow(2), nl=-1))

    Example:
        >>> # xdoctest: +REQUIRES(--gpu)
        >>> from .models.descriptor_network import *
        >>> input_shape = (4, 3, 32, 32)
        >>> self = DescriptorNetwork(input_shape=input_shape).to(0)
        >>> print(self)
        >>> shape = self.output_shape_for(input_shape)
        >>> print(ub.urepr(shape.hidden.shallow(2), nl=-1))
        {
            'dvecs': {
                'conv1': (4, 64, 16, 16),
                'bn1': (4, 64, 16, 16),
                'relu1': (4, 64, 16, 16),
                'maxpool': (4, 64, 8, 8),
                'layer1': (4, 256, 8, 8),
                'layer2': (4, 512, 4, 4),
                'layer3': (4, 1024, 2, 2),
                'layer4': (4, 2048, 1, 1),
                'avgpool': (4, 2048, 1, 1),
                'view': (4, 2048),
                'fc': (4, 128)
            }
        }
    """
    def __init__(self, branch='resnet50', input_shape=(1, 3, 416, 416),
                 norm_desc=False, desc_size=1024, hidden_channels=3, dropout=0,
                 norm='batch', noli='relu', residual=False, bias=False):
        """
        Note:
            * i have found norm_desc to be generally unhelpful.

        Example:
            >>> from .models.descriptor_network import *
            >>> from viame.pytorch import netharn as nh
            >>> input_shape = (4, 3, 32, 32)
            >>> self = DescriptorNetwork(input_shape=input_shape)
            >>> nh.OutputShapeFor(self)._check_consistency(input_shape)
            {'dvecs': (4, 1024)}
        """
        from viame.pytorch import netharn as nh
        super(DescriptorNetwork, self).__init__()

        pretrained = True

        if branch is None or branch == 'resnet50':
            self.branch = torchvision.models.resnet50(pretrained=pretrained)
        else:
            self.branch = branch
        if not isinstance(self.branch, torchvision.models.ResNet):
            raise ValueError('can only accept resnet at the moment')
        self.norm_desc = norm_desc

        self.in_channels = input_shape[1]
        self.out_channels = desc_size

        if self.branch.conv1.in_channels != self.in_channels:
            prev = self.branch.conv1
            cls = prev.__class__
            self.branch.conv1 = cls(
                in_channels=self.in_channels,
                out_channels=prev.out_channels,
                kernel_size=prev.kernel_size,
                stride=prev.stride,
                padding=prev.padding,
                dilation=prev.dilation,
                groups=prev.groups,
                bias=prev.bias,
            )
            if pretrained:
                nh.initializers.functional.load_partial_state(
                    self.branch.conv1,
                    prev.state_dict(),
                    leftover=nh.initializers.KaimingNormal(),
                    verbose=0,
                )

        # Note the advanced usage of output-shape-for
        if 0 and __debug__:
            # new torchvision broke this
            branch_field = nh.ReceptiveFieldFor(self.branch)(
                input_shape=input_shape)
            prepool_field = branch_field.hidden.shallow(1)['layer4']
            input_dims = np.array(input_shape[-2:])
            rf_stride = prepool_field['stride']
            if np.any(input_dims < rf_stride // 2):
                msg = ('Network is too deep OR input is to small. '
                       'rf_stride={} but input_dims={}'.format(
                           rf_stride, input_dims))

                self._debug_hidden(input_shape, n=2)
                print(msg)
                import warnings
                warnings.warn(msg)
                raise Exception(msg)

        branch_shape = nh.OutputShapeFor(self.branch)(input_shape)
        prepool_shape = branch_shape.hidden.shallow(1)['layer4']

        # replace the last layer of resnet with a linear embedding to learn the
        # LP distance between pairs of images.
        # Also need to replace the pooling layer in case the input has a
        # different size.
        self.prepool_shape = prepool_shape
        pool_channels = prepool_shape[1]
        pool_dims = prepool_shape[2:]
        if np.all(np.array(pool_dims) == 1):
            self.branch.avgpool = layers.Identity()
        else:
            self.branch.avgpool = torch.nn.AvgPool2d(pool_dims, stride=1)

        # Check that the modification to the layer fixed the size
        postbranch_shape = nh.OutputShapeFor(self.branch)(input_shape)
        postpool_shape = postbranch_shape.hidden.shallow(1)['layer4']

        assert np.all(np.array(prepool_shape[1:]) > 0)
        assert np.all(np.array(postpool_shape[1:]) > 0)

        # Replace the final linear layer with an MLP head
        self.branch.fc = layers.MultiLayerPerceptronNd(
            dim=0, in_channels=pool_channels, hidden_channels=hidden_channels,
            out_channels=desc_size, bias=bias, dropout=dropout, norm=norm,
            noli=noli, residual=residual)

    def _debug_hidden(self, input_shape, n=5):
        """
        Print internal shape and field info
        """
        from viame.pytorch import netharn as nh
        shape = nh.OutputShapeFor(self.branch)(input_shape=input_shape)
        print(ub.urepr(shape.hidden.shallow(n), nl=-1, dtype=False, si=True))
        field = nh.ReceptiveFieldFor(self.branch)(input_shape=input_shape)
        print(ub.urepr(field.hidden.shallow(n), nl=-1, dtype=False, si=True))

    def forward(self, inputs):
        """
        Compute a resnet50 vector for each input and look at the LP-distance
        between the vectors.

        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> from viame.pytorch import netharn as nh
            >>> inputs = nh.XPU(None).move(torch.rand(4, 21, 224, 224))
            >>> self = DescriptorNetwork(input_shape=inputs.shape)
            >>> output = self(inputs)

        Ignore:
            >>> from viame.pytorch import netharn as nh
            >>> input1 = nh.XPU(None).move(torch.rand(1, 3, 416, 416))
            >>> input2 = nh.XPU(None).move(torch.rand(1, 3, 416, 416))
            >>> input_shape1 = input1.shape
            >>> self = DescriptorNetwork(input_shape=input2.shape[1:])
            >>> self(input1, input2)
        """
        dvecs = self.branch(inputs)
        if self.norm_desc:
            # LP normalize the vectors
            dvecs = torch.nn.functional.normalize(dvecs, p=2)
        output = {
            'dvecs': dvecs
        }
        return output

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        TODO: work towards a system where we dont even have to miror the
        forward function, we can simply introspect and construct the analytic
        computation of shape / receptive field.
        """
        hidden = _Hidden()
        dvecs = hidden['dvecs'] = _OutputFor(self.branch)(inputs, **kwargs)
        if self.norm_desc:
            # LP normalize the vectors
            dvecs = hidden['norm_dvecs'] = _OutputFor(
                torch.nn.functional.normalize)(dvecs, p=2)
        output = {
            'dvecs': dvecs
        }
        return _Output.coerce(output, hidden)

    def output_shape_for(self, input_shape):
        from viame.pytorch import netharn as nh
        return self._analytic_forward(input_shape, nh.OutputShapeFor,
                                      nh.OutputShape, nh.HiddenShapes)

    def receptive_field_for(self, input_field=None):
        from viame.pytorch import netharn as nh
        return self._analytic_forward(input_field, nh.ReceptiveFieldFor,
                                      nh.ReceptiveField, nh.HiddenFields,
                                      input_shape=self.input_shape)

    def _initialize(self, verbose=0):
        """ Reinitialized with pretrained imagenet weights """
        import torch.utils.model_zoo as model_zoo
        from torchvision.models.resnet import model_urls
        from viame.pytorch import netharn as nh

        pretrained_state = model_zoo.load_url(model_urls['resnet50'])

        nh.initializers.nninit_base.load_partial_state(
            self.branch.conv1, pretrained_state,
            initializer=nh.initializers.KaimingNormal(), verbose=verbose,
        )
        return self
