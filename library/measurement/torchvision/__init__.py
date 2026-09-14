# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Stereo track pairing by deep descriptor cosine distance.

From `viame.pytorch` in P2-T06. Installed only with
`VIAME_ENABLE_PYTORCH-VISION`.
"""


# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ).
__sprokit_process_declarations__ = [
    ( "pair_stereo_tracks_pytorch",
      "Pair stereo detections using deep descriptor cosine distance",
      "viame.measurement.torchvision.pair_stereo_tracks:PairStereoTracks" ),
]
