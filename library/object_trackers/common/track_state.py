# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The lifecycle state of a ByteTrack-family track.

Shared by ByteTrack, OC-SORT and BoT-SORT since P2-T06, which defined it
identically. DeepSORT's tracks have a different lifecycle -- tentative,
confirmed, deleted -- and keep their own.
"""


class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3
