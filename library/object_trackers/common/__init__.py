# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""What the ByteTrack-family trackers share.

`kalman` is the constant-velocity box filter ByteTrack, OC-SORT, DeepSORT and
BoT-SORT each carried a copy of until P2-T06, and `track_state` the
new/tracked/lost/removed lifecycle three of them used. It registers nothing.
"""
