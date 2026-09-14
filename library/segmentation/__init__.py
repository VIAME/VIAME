# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Setting a detection's mask.

`interactive_segmentation` is the tool DIVE drives, `interactive_service`
the host it talks to, and `segmentation_utils` what both build on. Those
three are run directly, as
`python -m viame.segmentation.interactive_segmentation`, and register
nothing; `watershed_segmenter` came from `viame.opencv` in P2-T05 and does.
"""


# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "segment_via_points", "ocv_watershed",
      "OpenCV watershed-based point segmentation algorithm",
      "viame.segmentation.watershed_segmenter:WatershedSegmenter" ),
]
