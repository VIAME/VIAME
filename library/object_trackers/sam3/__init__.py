# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The SAM3 tracker. Its model handling is `viame.segmentation.sam3`.

From `viame.pytorch` in P2-T06. Installed only with
`VIAME_ENABLE_PYTORCH-SAM3`.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "track_objects", "sam3_tracker",
      "SAM3 (Segment Anything Model 3) based object tracker with text queries",
      "viame.object_trackers.sam3.sam3_tracker:SAM3Tracker" ),
]
