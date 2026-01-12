import torch
import ubelt as ub
from math import gcd


def rectify_nonlinearity(key=ub.NoParam, dim=2):
    """
    Allows dictionary based specification of a nonlinearity

    Example:
        >>> rectify_nonlinearity('relu')
        ReLU(...)
        >>> rectify_nonlinearity('leaky_relu')
        LeakyReLU(negative_slope=0.01...)
        >>> rectify_nonlinearity(None)
        None
        >>> rectify_nonlinearity('swish')
    """
    if key is None:
        return None

    if key is ub.NoParam:
        key = 'relu'

    if isinstance(key, str):
        key = {'type': key}
    elif isinstance(key, dict):
        key = key.copy()
    else:
        raise TypeError(type(key))
    kw = key
    noli_type = kw.pop('type')
    if 'inplace' not in kw:
        kw['inplace'] = True

    if noli_type == 'leaky_relu':
        cls = torch.nn.LeakyReLU
    elif noli_type == 'relu':
        cls = torch.nn.ReLU
    elif noli_type == 'elu':
        cls = torch.nn.ELU
    elif noli_type == 'celu':
        cls = torch.nn.CELU
    elif noli_type == 'selu':
        cls = torch.nn.SELU
    elif noli_type == 'relu6':
        cls = torch.nn.ReLU6
    elif noli_type == 'swish':
        from .layers.swish import Swish
        kw.pop('inplace', None)
        cls = Swish
    elif noli_type == 'mish':
        from .layers.mish import Mish
        kw.pop('inplace', None)
        cls = Mish
    else:
        raise KeyError('unknown type: {}'.format(kw))
    return cls(**kw)


def rectify_normalizer(in_channels, key=ub.NoParam, dim=2, **kwargs):
    """
    Allows dictionary based specification of a normalizing layer

    Args:
        in_channels (int): number of input channels
        dim (int): dimensionality
        **kwargs: extra args

    Example:
        >>> rectify_normalizer(8)
        BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, 'batch')
        BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, {'type': 'batch'})
        BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, 'group')
        GroupNorm(4, 8, eps=1e-05, affine=True)
        >>> rectify_normalizer(8, {'type': 'group', 'num_groups': 2})
        GroupNorm(2, 8, eps=1e-05, affine=True)
        >>> rectify_normalizer(8, dim=3)
        BatchNorm3d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, None)
        None
        >>> rectify_normalizer(8, key={'type': 'syncbatch'})
        >>> from viame.pytorch import netharn as nh
        >>> nh.layers.rectify_normalizer(8, {'type': 'group', 'num_groups': 'auto'})
        >>> nh.layers.rectify_normalizer(1, {'type': 'group', 'num_groups': 'auto'})
        >>> nh.layers.rectify_normalizer(16, {'type': 'group', 'num_groups': 'auto'})
        >>> nh.layers.rectify_normalizer(32, {'type': 'group', 'num_groups': 'auto'})
        >>> nh.layers.rectify_normalizer(64, {'type': 'group', 'num_groups': 'auto'})
        >>> nh.layers.rectify_normalizer(1024, {'type': 'group', 'num_groups': 'auto'})
    """
    if key is None:
        return None

    if key is ub.NoParam:
        key = 'batch'

    if isinstance(key, str):
        key = {'type': key}
    elif isinstance(key, dict):
        key = key.copy()
    else:
        raise TypeError(type(key))

    norm_type = key.pop('type')
    if norm_type == 'batch':
        in_channels_key = 'num_features'

        if dim == 0:
            cls = torch.nn.BatchNorm1d
        elif dim == 1:
            cls = torch.nn.BatchNorm1d
        elif dim == 2:
            cls = torch.nn.BatchNorm2d
        elif dim == 3:
            cls = torch.nn.BatchNorm3d
        else:
            raise ValueError(dim)
    elif norm_type == 'syncbatch':
        in_channels_key = 'num_features'
        cls = torch.nn.SyncBatchNorm
    elif norm_type == 'group':
        in_channels_key = 'num_channels'
        if key.get('num_groups') is None:
            key['num_groups'] = 'auto'
            # key['num_groups'] = ('gcd', min(in_channels, 32))

        if key.get('num_groups') == 'auto':
            if in_channels == 1:
                # Warning: cant group norm this
                from viame.pytorch import netharn as nh
                return nh.layers.Identity()
            else:
                valid_num_groups = [
                    factor for factor in range(1, in_channels)
                    if in_channels % factor == 0
                ]
                if len(valid_num_groups) == 0:
                    raise Exception
                infos = [
                    {'ng': ng, 'nc': in_channels / ng}
                    for ng in valid_num_groups
                ]
                ideal = in_channels ** (0.5)
                for item in infos:
                    item['heuristic'] = abs(ideal - item['ng']) * abs(ideal - item['nc'])
                chosen = sorted(infos, key=lambda x: (x['heuristic'], 1 - x['ng']))[0]
                key['num_groups'] = chosen['ng']
                if key['num_groups'] == in_channels:
                    key['num_groups'] = 1

            if isinstance(key['num_groups'], tuple):
                if key['num_groups'][0] == 'gcd':
                    key['num_groups'] = gcd(
                        key['num_groups'][1], in_channels)

            if in_channels % key['num_groups'] != 0:
                raise AssertionError(
                    'Cannot divide n_inputs {} by num groups {}'.format(
                        in_channels, key['num_groups']))
        cls = torch.nn.GroupNorm

    elif norm_type == 'batch+group':
        return torch.nn.Sequential(
            rectify_normalizer(in_channels, 'batch', dim=dim),
            rectify_normalizer(in_channels, ub.dict_union({'type': 'group'}, key), dim=dim),
        )
    else:
        raise KeyError('unknown type: {}'.format(key))
    assert in_channels_key not in key
    key[in_channels_key] = in_channels

    try:
        import copy
        kw = copy.copy(key)
        kw.update(kwargs)
        return cls(**kw)
    except Exception:
        raise
        # Ignore kwargs
        import warnings
        warnings.warn('kwargs ignored in rectify normalizer')
        return cls(**key)


def _ws_extra_repr(self):
    s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
         ', stride={stride}')
    if self.padding != (0,) * len(self.padding):
        s += ', padding={padding}'
    if self.dilation != (1,) * len(self.dilation):
        s += ', dilation={dilation}'
    if self.output_padding != (0,) * len(self.output_padding):
        s += ', output_padding={output_padding}'
    if self.groups != 1:
        s += ', groups={groups}'
    if self.bias is None:
        s += ', bias=False'
    if self.padding_mode != 'zeros':
        s += ', padding_mode={padding_mode}'
    if self.standardize_weights:
        s += ', standardize_weights={standardize_weights}'
    return s.format(**self.__dict__)


def weight_standardization_nd(dim, weight, eps):
    """
    Note: input channels must be greater than 1!

    weight = torch.rand(3, 2, 1, 1)
    dim = 2
    eps = 1e-5
    weight_normed = weight_standardization_nd(dim, weight, eps)
    print('weight = {!r}'.format(weight))
    print('weight_normed = {!r}'.format(weight_normed))

    weight = torch.rand(1, 2)
    dim = 0
    eps = 1e-5
    weight_normed = weight_standardization_nd(dim, weight, eps)
    print('weight = {!r}'.format(weight))
    print('weight_normed = {!r}'.format(weight_normed))
    """
    # Note: In 2D Weight dimensions are [C_out, C_in, H, W]
    mean_dims = tuple(list(range(1, dim + 2)))
    weight_mean = weight.mean(dim=mean_dims, keepdim=True)
    weight = weight - weight_mean
    trailing = [1] * (dim + 1)
    std = weight.view(weight.shape[0], -1).std(dim=1).view(-1, *trailing) + eps
    weight = weight / std.expand_as(weight)
    return weight


class Conv0d(torch.nn.Linear):
    """
    self = Conv0d(2, 3, 1, standardize_weights=True)
    print('self = {!r}'.format(self))
    x = torch.rand(1, 2)
    y = self.forward(x)
    print('y = {!r}'.format(y))
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', standardize_weights=False):
        assert kernel_size == 1, 'Conv0D must have a kernel_size=1'
        assert padding == 0, 'Conv0D must have padding=1'
        assert stride == 1, 'Conv0D must have stride=1'
        assert groups == 1, 'Conv0D must have groups=1'
        assert dilation == 1, 'Conv0D must have a dilation=1'
        # assert padding_mode == 'zeros'
        super().__init__(in_features=in_channels, out_features=out_channels,
                         bias=bias)
        self.standardize_weights = standardize_weights
        if standardize_weights:
            assert in_channels > 1, 'must be greater than 1 to prevent nan'
        self.dim = 0
        self.eps = 1e-5

    def forward(self, x):
        if self.standardize_weights:
            weight = weight_standardization_nd(self.dim, self.weight, self.eps)
            return torch.nn.functional.linear(x, weight, self.bias)
        else:
            return super().forward(x)

    def extra_repr(self) -> str:
        s = 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        if self.standardize_weights:
            s += ', standardize_weights={standardize_weights}'
        return s


class Conv1d(torch.nn.Conv1d):
    """
    self = Conv1d(2, 3, 1, standardize_weights=True)
    print('self = {!r}'.format(self))
    x = torch.rand(1, 2, 1)
    y = self.forward(x)
    print('y = {!r}'.format(y))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 standardize_weights=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.standardize_weights = standardize_weights
        if standardize_weights:
            assert in_channels > 1, 'must be greater than 1 to prevent nan'
        self.eps = 1e-5
        self.dim = 1

    def forward(self, x):
        if self.standardize_weights:
            weight = weight_standardization_nd(self.dim, self.weight, self.eps)
            return torch.nn.functional.conv1d(
                x, weight, self.bias, self.stride, self.padding, self.dilation,
                self.groups)
        else:
            return super().forward(x)

    extra_repr = _ws_extra_repr


class Conv2d(torch.nn.Conv2d):
    """
    self = Conv2d(2, 3, 1, standardize_weights=True)
    print('self = {!r}'.format(self))
    x = torch.rand(1, 2, 3, 3)
    y = self.forward(x)
    print('y = {!r}'.format(y))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 standardize_weights=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.standardize_weights = standardize_weights
        if standardize_weights:
            assert in_channels > 1, 'must be greater than 1 to prevent nan'
        self.eps = 1e-5
        self.dim = 2

    def forward(self, x):
        if self.standardize_weights:
            weight = weight_standardization_nd(self.dim, self.weight, self.eps)
            return torch.nn.functional.conv2d(
                x, weight, self.bias, self.stride, self.padding, self.dilation,
                self.groups)
        else:
            return super().forward(x)

    extra_repr = _ws_extra_repr


class Conv3d(torch.nn.Conv3d):
    """
    self = Conv3d(2, 3, 1, standardize_weights=True)
    print('self = {!r}'.format(self))
    x = torch.rand(1, 2, 1, 1, 1)
    y = self.forward(x)
    print('y = {!r}'.format(y))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 standardize_weights=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.standardize_weights = standardize_weights
        if standardize_weights:
            assert in_channels > 1, 'must be greater than 1 to prevent nan'
        self.eps = 1e-5
        self.dim = 3

    def forward(self, x):
        if self.standardize_weights:
            weight = weight_standardization_nd(self.dim, self.weight, self.eps)
            return torch.nn.functional.conv3d(
                x, weight, self.bias, self.stride, self.padding, self.dilation,
                self.groups)
        else:
            return super().forward(x)

    extra_repr = _ws_extra_repr


def rectify_conv(dim=2):
    conv_cls = {
        0: Conv0d,
        # 1: torch.nn.Conv1d,
        # 2: torch.nn.Conv2d,
        # 3: torch.nn.Conv3d,
        1: Conv1d,
        2: Conv2d,
        3: Conv3d,
    }[dim]
    return conv_cls


def rectify_dropout(dim=2):
    conv_cls = {
        0: torch.nn.Dropout,
        1: torch.nn.Dropout,
        2: torch.nn.Dropout2d,
        3: torch.nn.Dropout3d,
    }[dim]
    return conv_cls


def rectify_maxpool(dim=2):
    conv_cls = {
        0: torch.nn.MaxPool1d,
        1: torch.nn.MaxPool1d,
        2: torch.nn.MaxPool2d,
        3: torch.nn.MaxPool3d,
    }[dim]
    return conv_cls
