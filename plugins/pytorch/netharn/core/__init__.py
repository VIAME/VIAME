# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Netharn Core Training Framework

This is the core training framework originally from the netharn package.
It provides FitHarn, XPU, and other training utilities.
"""

__version__ = '0.6.2'

try:
    # PIL 7.0.0 removed PIL_VERSION, which breaks torchvision, monkey patch it
    # back in.
    import PIL
    PIL.PILLOW_VERSION = PIL.__version__
except (AttributeError, Exception):
    pass


# patch for imgaug
try:
    import numpy as np
    np.random.bit_generator = np.random._bit_generator
except (AttributeError, Exception):
    pass


from .api import (
    Initializer, Optimizer, Criterion, Loaders, Scheduler, Dynamics,
    configure_hacks, configure_workdir,
)
from .device import (XPU,)
from .fit_harn import (FitHarn,)
from .hyperparams import (HyperParams,)
from .monitor import (Monitor,)
from .analytic.output_shape_for import (
    OutputShapeFor, OutputShape, HiddenShapes)
from .analytic.receptive_field_for import (
    ReceptiveFieldFor, ReceptiveField, HiddenFields)

__extra_all__ = [
    'Initializer',
    'Optimizer',
    'Criterion',
    'Loaders',
    'Scheduler',
    'Dynamics',
    'configure_hacks',
    'configure_workdir',

    'XPU',
    'FitHarn',
    'HyperParams',
    'Monitor',
    'Initializer',

    'OutputShapeFor',
    'OutputShape',
    'HiddenShapes',

    'ReceptiveFieldFor',
    'ReceptiveField',
    'HiddenFields',
]

# Import submodules
from . import api
from . import criterions
from . import data
from . import device
from . import exceptions
from . import fit_harn
from . import hyperparams
from . import initializers
from . import layers
from . import mixins
from . import models
from . import monitor
from . import optimizers
from . import prefit
from . import schedulers
from . import util
from .analytic import analytic_for
from .analytic import output_shape_for
from .analytic import receptive_field_for

__all__ = ['Criterion', 'Dynamics', 'FitHarn', 'HiddenFields', 'HiddenShapes',
           'HyperParams', 'Initializer', 'Initializer', 'Loaders', 'Monitor',
           'Optimizer', 'OutputShape', 'OutputShapeFor', 'ReceptiveField',
           'ReceptiveFieldFor', 'Scheduler', 'XPU', 'analytic_for', 'api',
           'configure_hacks', 'configure_workdir', 'criterions', 'data',
           'device', 'exceptions', 'fit_harn', 'hyperparams',
           'initializers', 'layers', 'mixins', 'models', 'monitor',
           'optimizers', 'output_shape_for', 'prefit', 'receptive_field_for',
           'schedulers', 'util']
