# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Trainers, in python.

ByteTrack and OC-SORT parameter estimation, the three-frame difference
detector settings estimation, tracker parameter search and the training data
handling, from `viame.core` in P2-T07. The pytorch trainers are still
`viame.pytorch`.
"""


# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "train_detector", "frame_diff",
      "Three-frame difference detector settings estimation",
      "viame.training.frame_diff_trainer:FrameDiffTrainer" ),
    ( "train_tracker", "bytetrack",
      "ByteTrack parameter estimation from track groundtruth",
      "viame.training.bytetrack_trainer:ByteTrackTrainer" ),
    ( "train_tracker", "ocsort",
      "OC-SORT parameter estimation and optional Deep OC-SORT Re-ID training",
      "viame.training.ocsort_trainer:OCSORTTrainer" ),
]
