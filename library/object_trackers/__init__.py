# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Object trackers, in python.

ByteTrack and OC-SORT, the simple and multi-camera homography IOU trackers,
and tube-IoU track-set fusion, from `viame.core` in P2-T06. Their trainers --
ByteTrack and OC-SORT parameter estimation -- are training, and follow in
P2-T07. The pytorch trackers are still `viame.pytorch`.
"""


# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "track_objects", "bytetrack",
      "ByteTrack multi-object tracker with two-stage association",
      "viame.object_trackers.bytetrack_tracker:ByteTrackTracker" ),
    ( "track_objects", "ocsort",
      "OC-SORT / Deep OC-SORT tracker with observation-centric momentum, re-update, recovery, and optional appearance fusion",
      "viame.object_trackers.ocsort_tracker:OCSORTTracker" ),
]


# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ).
__sprokit_process_declarations__ = [
    ( "merge_track_sets_tube_iou",
      "Fusion of multiple object track sets via tube-IoU association",
      "viame.object_trackers.merge_tracks_tube_iou:MergeTracksTubeIoU" ),
    ( "multicam_homog_tracker",
      "Multi-camera IOU-based tracker with homography support",
      "viame.object_trackers.multicam_homog_tracker:MulticamHomogTracker" ),
    ( "simple_homog_tracker",
      "Simple IOU-based tracker with homography support",
      "viame.object_trackers.simple_homog_tracker:SimpleHomogTracker" ),
]
