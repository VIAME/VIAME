# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""COLMAP structure from motion and dense reconstruction.

`reconstruction`, which the 3d tool uses for its non-planar SfM and dense
modes, and `prior_coverage_sfm`, which the register tool delegates to. Both
need pycolmap, and open3d for the dense modes, so they are imported by their
callers when wanted. From `viame.colmap` in P2-T08, installed only with
`VIAME_ENABLE_COLMAP`. Nothing here registers.
"""

__vital_algorithm_declarations__ = []
__sprokit_process_declarations__ = []
