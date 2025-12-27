# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Backwards compatibility module.

All functions have been moved to utilities.py.
This module re-exports them for backwards compatibility.
"""

from .utilities import (
    vital_config_update,
    pad_img_to_fit_bbox,
    safe_crop,
    recurse_copy,
)
