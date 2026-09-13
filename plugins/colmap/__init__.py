# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""viame.colmap - COLMAP / survey-registration plugin.

Hosts the colmap_registration sprokit process, declared below so that it
costs nothing until a pipeline wants it. The heavier SfM / dense-reconstruction
modules (reconstruction, prior_coverage_sfm) require pycolmap and are imported
lazily by their callers, so process registration never pulls them in.
"""

# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ). A process registers by
# calling `process_factory.add_process( name, description, ctor )`, and the
# ctor is only ever called -- so a function that imports and constructs is as
# good as the class and costs nothing until a pipeline wants one. See P8-T10.
__sprokit_process_declarations__ = [
    ( "colmap_registration",
      "Multi-camera survey registration (affine chains + rig cross-camera consensus + optional GPS metadata); drop-in for many_image_stabilizer",
      "viame.colmap.colmap_registration:ColmapRegistration" ),
]
