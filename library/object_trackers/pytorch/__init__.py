# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The pytorch trackers that need nothing past `VIAME_ENABLE_PYTORCH`.

BoT-SORT, DeepSORT, MOTR, and SRNN with the `srnn/` tree it runs on. From
`viame.pytorch` in P2-T06. Their trainers follow in P2-T07.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "track_objects", "botsort",
      "BoT-SORT multi-object tracker with CMC and IoU-ReID fusion",
      "viame.object_trackers.pytorch.botsort_tracker:BoTSORTTracker" ),
    ( "track_objects", "deepsort",
      "DeepSORT multi-object tracker with deep appearance features",
      "viame.object_trackers.pytorch.deepsort_tracker:DeepSORTTracker" ),
    ( "track_objects", "motr",
      "MOTR-style track-query transformer tracker with learned association",
      "viame.object_trackers.pytorch.motr_tracker:MOTRTracker" ),
    ( "track_objects", "srnn",
      "Structural RNN multi-object tracker",
      "viame.object_trackers.pytorch.srnn_tracker:SRNNTracker" ),
]
