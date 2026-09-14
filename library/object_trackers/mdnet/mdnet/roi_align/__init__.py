# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from .functions.roi_align import roi_align, roi_align_ada
from .modules.roi_align import RoIAlign, RoIAlignAda, RoIAlignAvg, RoIAlignMax, RoIAlignAdaMax

__all__ = ['roi_align', 'RoIAlign', 'roi_align_ada', 'RoIAlignAda', 'RoIAlignAvg', 'RoIAlignMax', 'RoIAlignAdaMax']
