# -*- coding: utf-8 -*-
"""
mkinit netharn.initializers
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from ..api import Initializer  # NOQA

# backwards compatibility patch to support older deployed models
from . import core
nninit_base = core
nninit_base._BaseInitializer = Initializer

__explicit__ = ['Initializer', 'nninit_base']


# <AUTOGEN_INIT>
from . import core
from . import functional
from . import lsuv
from . import pretrained

from .core import (KaimingNormal, KaimingUniform, NoOp,
                   Orthogonal,)
from .functional import (apply_initializer,
                         load_partial_state,
                         trainable_layers,)
from .lsuv import (LSUV, Orthonormal, svd_orthonormal,)
from .pretrained import (Pretrained,)

__all__ = ['Initializer', 'KaimingNormal', 'KaimingUniform', 'LSUV', 'NoOp',
           'Orthogonal', 'Orthonormal', 'Pretrained', 'apply_initializer',
           'core', 'functional', 'load_partial_state', 'lsuv', 'pretrained',
           'svd_orthonormal', 'trainable_layers']
# </AUTOGEN_INIT>
