# -*- coding: utf-8 -*-
"""
mkinit netharn.initializers
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from netharn.api import Initializer  # NOQA

# backwards compatibility patch to support older deployed models
from netharn.initializers import core
nninit_base = core
nninit_base._BaseInitializer = Initializer

__explicit__ = ['Initializer', 'nninit_base']


# <AUTOGEN_INIT>
from netharn.initializers import core
from netharn.initializers import functional
from netharn.initializers import lsuv
from netharn.initializers import pretrained

from netharn.initializers.core import (KaimingNormal, KaimingUniform, NoOp,
                                       Orthogonal,)
from netharn.initializers.functional import (apply_initializer,
                                             load_partial_state,
                                             trainable_layers,)
from netharn.initializers.lsuv import (LSUV, Orthonormal, svd_orthonormal,)
from netharn.initializers.pretrained import (Pretrained,)

__all__ = ['Initializer', 'KaimingNormal', 'KaimingUniform', 'LSUV', 'NoOp',
           'Orthogonal', 'Orthonormal', 'Pretrained', 'apply_initializer',
           'core', 'functional', 'load_partial_state', 'lsuv', 'pretrained',
           'svd_orthonormal', 'trainable_layers']
# </AUTOGEN_INIT>
