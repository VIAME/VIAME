# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""SAM3 point segmentation, text query, and detection and track refinement.
`sam3_tracker` and `sam3_trainer` are trackers and trainers and follow in
P2-T06 and P2-T07.

From `viame.pytorch` in P2-T05. Installed only with `VIAME_ENABLE_PYTORCH-SAM3`, so its
declarations exist only in a build that has it.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "perform_text_query", "sam3",
      "SAM3-based text query for object detection and track refinement",
      "viame.segmentation.sam3.sam3_text_query:SAM3TextQuery" ),
    ( "refine_detections", "sam3",
      "SAM3 (SAM 2.1) based detection refiner for adding segmentation masks",
      "viame.segmentation.sam3.sam3_refiner:Sam3DetectionRefiner" ),
    ( "refine_tracks", "sam3",
      "SAM3 (Segment Anything Model 3) based track refiner with text queries",
      "viame.segmentation.sam3.sam3_refiner:SAM3Refiner" ),
    ( "segment_via_points", "sam3",
      "SAM3-based point segmentation algorithm",
      "viame.segmentation.sam3.sam3_segmenter:SAM3Segmenter" ),
    ( "train_detector", "sam3",
      "SAM3 (Segment Anything Model 3) fine-tuning for segmentation",
      "viame.segmentation.sam3.sam3_trainer:SAM3Trainer" ),
    ( "train_tracker", "sam3",
      "SAM3 tracker fine-tuning with temporal mask propagation",
      "viame.segmentation.sam3.sam3_trainer:SAM3TrackerTrainer" ),
]
