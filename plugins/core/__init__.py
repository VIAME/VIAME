# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #


# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
# `kwiver.vital.plugins.discovery` turns each into a stand-in that imports
# its module the first time something asks for an instance.
#
# This replaced a `__vital_algorithm_register__` that imported every
# implementation module at startup, which was the only way the classes came
# to exist for the subclass walk that registers them. See P8-T10.
__vital_algorithm_declarations__ = [
    ( "track_objects", "bytetrack",
      "ByteTrack multi-object tracker with two-stage association",
      "viame.core.bytetrack_tracker:ByteTrackTracker" ),
    ( "track_objects", "ocsort",
      "OC-SORT / Deep OC-SORT tracker with observation-centric momentum, re-update, recovery, and optional appearance fusion",
      "viame.core.ocsort_tracker:OCSORTTracker" ),
    ( "train_detector", "frame_diff",
      "Three-frame difference detector settings estimation",
      "viame.core.frame_diff_trainer:FrameDiffTrainer" ),
    ( "train_tracker", "bytetrack",
      "ByteTrack parameter estimation from track groundtruth",
      "viame.core.bytetrack_trainer:ByteTrackTrainer" ),
    ( "train_tracker", "ocsort",
      "OC-SORT parameter estimation and optional Deep OC-SORT Re-ID training",
      "viame.core.ocsort_trainer:OCSORTTrainer" ),
]

# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ). A process registers by
# calling `process_factory.add_process( name, description, ctor )`, and the
# ctor is only ever called -- so a function that imports and constructs is as
# good as the class and costs nothing until a pipeline wants one. See P8-T10.
__sprokit_process_declarations__ = [
    ( "merge_track_sets_tube_iou",
      "Fusion of multiple object track sets via tube-IoU association",
      "viame.core.merge_tracks_tube_iou:MergeTracksTubeIoU" ),
    ( "multicam_homog_tracker",
      "Multi-camera IOU-based tracker with homography support",
      "viame.core.multicam_homog_tracker:MulticamHomogTracker" ),
    ( "simple_homog_tracker",
      "Simple IOU-based tracker with homography support",
      "viame.core.simple_homog_tracker:SimpleHomogTracker" ),
]
