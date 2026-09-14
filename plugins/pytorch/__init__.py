# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ). Nothing
# here imports torch; `kwiver.vital.plugins.discovery` turns each line into a
# stand-in that imports its module the first time something asks for an
# instance.
#
# It used to be thirty-seven `from viame.pytorch import ...` lines inside
# `__vital_algorithm_register__`, run at startup so that the classes existed
# for the subclass walk that does the registering. That import cost **1.4
# seconds of torch** on every `viame` command that touched the plugin
# system -- `viame runner --help` among them -- to make a list of names.
#
# Generated from the running registry rather than read off the source, for
# the reason finding 1.30 gives: the build knows what it registers and the
# source only suggests it. The `tests/baseline/declarations.py --record` this
# used to say regenerates the list was never committed; add an entry by hand.
#
# The detectors, classifiers and segmenters are gated subpackages of
# `viame.object_detectors`, `viame.classifiers` and `viame.segmentation` since
# P2-T05, and the trackers of `viame.object_trackers` since P2-T06, each
# declaring what its option installs. What is left here are the trainers,
# stereo and descriptor modules P2-T06 and P2-T07 take.
__vital_algorithm_declarations__ = [
    ( "train_detector", "mit_yolo",
      "PyTorch MIT YOLO detection training routine",
      "viame.pytorch.mit_yolo_trainer:MITYoloTrainer" ),
    ( "train_detector", "mmdet",
      "PyTorch MMDetection training routine",
      "viame.pytorch.mmdet_trainer:MMDetTrainer" ),
    ( "train_detector", "netharn",
      "PyTorch NetHarn detection training routine",
      "viame.pytorch.netharn_trainer:NetHarnTrainer" ),
    ( "train_detector", "rf_detr",
      "PyTorch RF-DETR detection training routine",
      "viame.pytorch.rf_detr_trainer:RFDETRTrainer" ),
    ( "train_detector", "sam3",
      "SAM3 (Segment Anything Model 3) fine-tuning for segmentation",
      "viame.pytorch.sam3_trainer:SAM3Trainer" ),
    ( "train_detector", "ultralytics",
      "PyTorch Ultralytics YOLO training routine",
      "viame.pytorch.ultralytics_trainer:UltralyticsTrainer" ),
    ( "train_tracker", "botsort",
      "PyTorch BoT-SORT Re-ID model training and parameter estimation",
      "viame.pytorch.botsort_trainer:BoTSORTTrainer" ),
    ( "train_tracker", "deepsort",
      "PyTorch DeepSORT Re-ID model training",
      "viame.pytorch.deepsort_trainer:DeepSORTTrainer" ),
    ( "train_tracker", "motr",
      "MOTR-style track-query transformer training for learned association",
      "viame.pytorch.motr_trainer:MOTRTrainer" ),
    ( "train_tracker", "sam3",
      "SAM3 tracker fine-tuning with temporal mask propagation",
      "viame.pytorch.sam3_trainer:SAM3TrackerTrainer" ),
    ( "train_tracker", "siammask",
      "PyTorch SiamMask tracker training routine",
      "viame.pytorch.siammask_trainer:SiamMaskTrainer" ),
    ( "train_tracker", "siamrpn",
      "PyTorch SiamRPN++ tracker training routine",
      "viame.pytorch.siammask_trainer:SiamRPNTrainer" ),
    ( "train_tracker", "srnn",
      "PyTorch SRNN tracker training routine",
      "viame.pytorch.srnn_trainer:SRNNTrainer" ),
]

# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ). A process registers by
# calling `process_factory.add_process( name, description, ctor )`, and the
# ctor is only ever called -- so a function that imports and constructs is as
# good as the class and costs nothing until a pipeline wants one. See P8-T10.
__sprokit_process_declarations__ = [
    ( "convert_to_onnx",
      "Convert a VIAME model to onnx",
      "viame.pytorch.convert_to_onnx_process:OnnxConverter" ),
]
