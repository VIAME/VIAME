# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Re-export kwiver.vital.types for convenient access via viame.types.
"""

from kwiver.vital.types import *
from kwiver.vital import types

# Make 'from viame.types import X' work for any X in kwiver.vital.types
__all__ = types.__all__ if hasattr(types, '__all__') else dir(types)
