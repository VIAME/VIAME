# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The MDNet tracker, and the vendored MDNet tree it wraps as `mdnet/`.

From `viame.pytorch` in P2-T06. Installed only with
`VIAME_ENABLE_PYTORCH-MDNET`. It was never declared before: `viame.pytorch`
declared what the build that generated its list had enabled, and this
option was off there, so turning it on installed the module and registered
nothing.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "track_objects", "mdnet",
      "MDNet visual object tracker",
      "viame.object_trackers.mdnet.mdnet_tracker:MDNetTracker" ),
]
