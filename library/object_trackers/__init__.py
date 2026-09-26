# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Object trackers, in python.

ByteTrack and OC-SORT, the simple and multi-camera homography IOU trackers,
and tube-IoU track-set fusion, from `viame.core` in P2-T06, and since P2-T07
the trainers beside them: ByteTrack and OC-SORT parameter estimation, the
tracker parameter search and the training data handling the tracker
trainers share. The pytorch trackers and their trainers are the gated
subpackages beside this: `pytorch`, `siammask`, `mdnet` and `sam3`.
"""


# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "track_objects", "homog_iou",
      "Fixed-target tracker matching boxes after homography registration",
      "viame.object_trackers.homog_iou_tracker:HomogIOUTracker" ),
    ( "train_tracker", "homog_iou",
      "Homography IoU tracker parameter estimation from track groundtruth",
      "viame.object_trackers.homog_iou_trainer:HomogIOUTrainer" ),
    ( "track_objects", "bytetrack",
      "ByteTrack multi-object tracker with two-stage association",
      "viame.object_trackers.bytetrack_tracker:ByteTrackTracker" ),
    ( "track_objects", "ocsort",
      "OC-SORT / Deep OC-SORT tracker with observation-centric momentum, re-update, recovery, and optional appearance fusion",
      "viame.object_trackers.ocsort_tracker:OCSORTTracker" ),
    ( "train_tracker", "bytetrack",
      "ByteTrack parameter estimation from track groundtruth",
      "viame.object_trackers.bytetrack_trainer:ByteTrackTrainer" ),
    ( "train_tracker", "ocsort",
      "OC-SORT parameter estimation and optional Deep OC-SORT Re-ID training",
      "viame.object_trackers.ocsort_trainer:OCSORTTrainer" ),
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
