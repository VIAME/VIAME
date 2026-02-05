# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Netharn Core Training Framework

This is the core training framework originally from the netharn package.
It provides FitHarn, XPU, and other training utilities.
"""

__version__ = '0.6.2'

# Suppress known harmless warnings:
# - RuntimeWarning from runpy when running submodules with python -m
#   (the package __init__ imports submodules before runpy executes them)
# - NCCL not compiled warning (Windows does not support NCCL)
import warnings as _warnings
_warnings.filterwarnings(
    'ignore',
    message=r".*found in sys\.modules after import of package.*",
    category=RuntimeWarning,
)
_warnings.filterwarnings(
    'ignore',
    message=r".*not compiled with NCCL support.*",
    category=UserWarning,
)
del _warnings

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

# Import submodules - core training framework
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

# Import submodules - detection/classification (from bioharn)
from . import bio_util
from . import compat
from . import detection_models
from . import io

# Import detection/classification modules
from . import clf_dataset
from . import clf_eval
from . import clf_fit
from . import clf_predict
from . import detect_dataset
from . import detect_eval
from . import detect_fit
from . import detect_predict

__all__ = ['Criterion', 'Dynamics', 'FitHarn', 'HiddenFields', 'HiddenShapes',
           'HyperParams', 'Initializer', 'Initializer', 'Loaders', 'Monitor',
           'Optimizer', 'OutputShape', 'OutputShapeFor', 'ReceptiveField',
           'ReceptiveFieldFor', 'Scheduler', 'XPU', 'analytic_for', 'api',
           'configure_hacks', 'configure_workdir', 'criterions', 'data',
           'device', 'exceptions', 'fit_harn', 'hyperparams',
           'initializers', 'layers', 'mixins', 'models', 'monitor',
           'optimizers', 'output_shape_for', 'prefit', 'receptive_field_for',
           'schedulers', 'util',
           # Detection/classification modules
           'bio_util', 'compat', 'detection_models', 'io',
           'clf_dataset', 'clf_eval', 'clf_fit', 'clf_predict',
           'detect_dataset', 'detect_eval', 'detect_fit', 'detect_predict']
