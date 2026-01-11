"""
References:
    https://github.com/alykhantejani/initializers
"""
import torch
from viame.pytorch.netharn import api
from .functional import apply_initializer


class NoOp(api.Initializer):
    """
    An initializer that does nothing, which is useful when you have initialized
    the weights yourself.

    Example:
        >>> from .models import toynet
        >>> import copy
        >>> self = NoOp()
        >>> model = toynet.ToyNet2d()
        >>> old_state = sum(v.sum() for v in model.state_dict().values())
        >>> self(model)
        >>> new_state = sum(v.sum() for v in model.state_dict().values())
        >>> assert old_state == new_state
        >>> assert self.history() is None
    """
    def forward(self, model):
        return


class Orthogonal(api.Initializer):
    """
    Same as Orthogonal, but uses pytorch implementation

    Example:
        >>> from .models import toynet
        >>> self = Orthogonal()
        >>> model = toynet.ToyNet2d()
        >>> try:
        >>>     self(model)
        >>> except RuntimeError:
        >>>     import pytest
        >>>     pytest.skip('geqrf: Lapack probably not availble')
        >>> layer = torch.nn.modules.Conv2d(3, 3, 3)
        >>> self(layer)
    """
    def __init__(self, gain=1):
        self.gain = gain

    def forward(self, model):
        try:
            func = torch.nn.init.orthogonal_
        except AttributeError:
            func = torch.nn.init.orthogonal

        apply_initializer(model, func, self.__dict__)


class KaimingUniform(api.Initializer):
    """
    Same as HeUniform, but uses pytorch implementation

    Example:
        >>> from .models import toynet
        >>> self = KaimingUniform()
        >>> model = toynet.ToyNet2d()
        >>> self(model)
        >>> layer = torch.nn.modules.Conv2d(3, 3, 3)
        >>> self(layer)
    """
    def __init__(self, param=0, mode='fan_in'):
        self.a = param
        self.mode = mode

    def forward(self, model):
        try:
            func = torch.nn.init.kaiming_uniform_
        except AttributeError:
            func = torch.nn.init.kaiming_uniform
        apply_initializer(model, func, self.__dict__)


class KaimingNormal(api.Initializer):
    """
    Same as HeNormal, but uses pytorch implementation

    Example:
        >>> from .models import toynet
        >>> self = KaimingNormal()
        >>> model = toynet.ToyNet2d()
        >>> self(model)
        >>> layer = torch.nn.modules.Conv2d(3, 3, 3)
        >>> self(layer)
    """
    def __init__(self, param=0, mode='fan_in'):
        self.a = param
        self.mode = mode

    def forward(self, model):
        try:
            func = torch.nn.init.kaiming_normal_
        except AttributeError:
            func = torch.nn.init.kaiming_normal
        apply_initializer(model, func, self.__dict__)
