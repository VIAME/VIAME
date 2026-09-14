"""
Vendored copy of torch_liberator 0.2.1 (https://gitlab.kitware.com/computer-vision/torch_liberator).

This used to be a pip dependency patched in place after install by four
ReplaceStringInFile calls in cmake/custom_install_viame.cmake. Those patches
silently became no-ops whenever upstream touched the lines they matched, and
nothing failed until a model would not load, so the source is carried here
instead with the fixes already applied:

    * initializer.py, deployer.py, xpu_device.py -- torch.load() needs an
      explicit weights_only=False under torch >= 2.6, where the default flipped.
    * exporter.py -- ub.ensure_unicode() was removed from ubelt.

Imports are relative, so this is reachable as
``viame.object_detectors.netharn.netharn.torch_liberator`` and not as a top-level module.
Still pip installed: ``liberator`` (static code extraction, used by exporter)
and ``networkx_algo_common_subtree`` (used by initializer).

Original packaging note:
    mkinit torch_liberator -w
"""

__version__ = '0.2.1'

from . import deployer
from . import exporter

from .deployer import (DeployedModel, deploy,)
from .exporter import (export_model_code,)
from .initializer import (load_partial_state, Pretrained)

__all__ = ['DeployedModel', 'deploy', 'deployer', 'export_model_code',
           'exporter', 'load_partial_state', 'Pretrained']
