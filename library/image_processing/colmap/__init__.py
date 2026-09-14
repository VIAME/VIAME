# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""COLMAP survey registration.

The `colmap_registration` process: affine chains, rig cross-camera consensus
and optional GPS metadata, as a drop-in for `many_image_stabilizer`. From
`viame.colmap` in P2-T08, installed only with `VIAME_ENABLE_COLMAP`. It needs
no pycolmap; the structure-from-motion modules that do are
`viame.measurement.colmap`.
"""

__vital_algorithm_declarations__ = []

# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ).
__sprokit_process_declarations__ = [
    ( "colmap_registration",
      "Multi-camera survey registration (affine chains + rig cross-camera consensus + optional GPS metadata); drop-in for many_image_stabilizer",
      "viame.image_processing.colmap.colmap_registration:ColmapRegistration" ),
]
