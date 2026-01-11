# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Netharn - PyTorch Training Harness for VIAME

This package merges the netharn and bioharn frameworks into a single
location for VIAME's PyTorch-based training and detection functionality.

Subpackages:
    core - Netharn core training framework (FitHarn, XPU, etc.)
    bio  - Bioharn detection/classification modules
"""

__version__ = '0.7.0'

from . import core
from . import bio
