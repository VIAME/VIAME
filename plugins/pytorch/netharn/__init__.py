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


import importlib as _importlib

_ATTRS = {
    'Initializer': '.api', 'Optimizer': '.api', 'Criterion': '.api',
    'Loaders': '.api', 'Scheduler': '.api', 'Dynamics': '.api',
    'configure_hacks': '.api', 'configure_workdir': '.api',
    'XPU': '.device',
    'FitHarn': '.fit_harn',
    'HyperParams': '.hyperparams',
    'Monitor': '.monitor',
    'OutputShapeFor': '.analytic.output_shape_for',
    'OutputShape': '.analytic.output_shape_for',
    'HiddenShapes': '.analytic.output_shape_for',
    'ReceptiveFieldFor': '.analytic.receptive_field_for',
    'ReceptiveField': '.analytic.receptive_field_for',
    'HiddenFields': '.analytic.receptive_field_for',
    'analytic_for': '.analytic',
    'output_shape_for': '.analytic',
    'receptive_field_for': '.analytic',
}

_SUBMODULES = [
    'api', 'criterions', 'data', 'device', 'exceptions', 'fit_harn',
    'hyperparams', 'initializers', 'layers', 'mixins', 'models', 'monitor',
    'optimizers', 'prefit', 'schedulers', 'util', 'analytic',
    'bio_util', 'compat', 'detection_models', 'io',
    'clf_dataset', 'clf_eval', 'clf_fit', 'clf_predict',
    'detect_dataset', 'detect_eval', 'detect_fit', 'detect_predict',
]

__all__ = sorted(set(_ATTRS) | set(_SUBMODULES))


# Everything is resolved on first access: the plugin loader imports this
# package on every viame startup and the eager version cost eight seconds
def __getattr__(name):
    if name in _ATTRS:
        value = getattr(_importlib.import_module(_ATTRS[name], __name__), name)
    elif name in _SUBMODULES:
        value = _importlib.import_module('.' + name, __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
