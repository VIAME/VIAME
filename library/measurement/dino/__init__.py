# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The DINO feature matcher.

Called from C++ -- `measurement_utilities.cxx` imports
`viame.measurement.dino.dino_matcher` through the Python C API -- and from the
interactive stereo service. It registers nothing, so this package declares
nothing and is absent from `BUILTIN_PLUGIN_PACKAGES`. From `viame.pytorch` in
P2-T06; installed only with `VIAME_ENABLE_PYTORCH-DINO3`.
"""
