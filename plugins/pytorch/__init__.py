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
# source only suggests it. Regenerate with
# `tests/baseline/declarations.py --record` if an implementation is added.
__vital_algorithm_declarations__ = [
    ( "compute_stereo_depth_map", "fast_foundation_stereo",
      "Stereo depth/disparity estimation using NVIDIA Fast-Foundation-Stereo (real-time variant)",
      "viame.pytorch.fast_foundation_stereo:FastFoundationStereo" ),
    ( "compute_stereo_depth_map", "foundation_stereo",
      "Stereo depth/disparity estimation using NVIDIA Foundation-Stereo model",
      "viame.pytorch.foundation_stereo:FoundationStereo" ),
    ( "image_object_detector", "huggingface_zeroshot_detector",
      "HuggingFace ZeroShot Object Detection",
      "viame.pytorch.huggingface_zeroshot_detector:HuggingFaceZeroShotDetector" ),
    ( "image_object_detector", "mit_yolo",
      "PyTorch MIT YOLO detection routine",
      "viame.pytorch.mit_yolo_detector:MITYoloDetector" ),
    ( "image_object_detector", "mmdet",
      "PyTorch MMDetection inference routine",
      "viame.pytorch.mmdet_detector:MMDetDetector" ),
    ( "image_object_detector", "netharn",
      "PyTorch Netharn detection routine",
      "viame.pytorch.netharn_detector:NetharnDetector" ),
    ( "image_object_detector", "netharn_classifier",
      "PyTorch Netharn classification routine",
      "viame.pytorch.netharn_classifier:NetharnClassifier" ),
    ( "image_object_detector", "rf_detr",
      "PyTorch RF-DETR detection routine",
      "viame.pytorch.rf_detr_detector:RFDETRDetector" ),
    ( "image_object_detector", "ultralytics",
      "PyTorch Ultralytics detection routine",
      "viame.pytorch.ultralytics_detector:UltralyticsDetector" ),
    ( "perform_text_query", "sam3",
      "SAM3-based text query for object detection and track refinement",
      "viame.pytorch.sam3_text_query:SAM3TextQuery" ),
    ( "refine_detections", "netharn",
      "PyTorch Netharn refiner routine",
      "viame.pytorch.netharn_refiner:NetharnRefiner" ),
    ( "refine_detections", "rf_detr",
      "Run RF-DETR segmentation/keypoint heads on existing boxes",
      "viame.pytorch.rf_detr_refiner:RFDETRRefiner" ),
    ( "refine_detections", "sam2",
      "SAM2-based detection refiner",
      "viame.pytorch.sam2_refiner:Sam2Refiner" ),
    ( "refine_detections", "sam3",
      "SAM3 (SAM 2.1) based detection refiner for adding segmentation masks",
      "viame.pytorch.sam3_refiner:Sam3DetectionRefiner" ),
    ( "refine_tracks", "sam2",
      "SAM2-based track refiner for adding segmentation masks to tracks",
      "viame.pytorch.sam2_refiner:Sam2TrackRefiner" ),
    ( "refine_tracks", "sam3",
      "SAM3 (Segment Anything Model 3) based track refiner with text queries",
      "viame.pytorch.sam3_refiner:SAM3Refiner" ),
    ( "segment_via_points", "sam2",
      "SAM2-based point segmentation algorithm",
      "viame.pytorch.sam2_segmenter:SAM2Segmenter" ),
    ( "segment_via_points", "sam3",
      "SAM3-based point segmentation algorithm",
      "viame.pytorch.sam3_segmenter:SAM3Segmenter" ),
    ( "track_objects", "botsort",
      "BoT-SORT multi-object tracker with CMC and IoU-ReID fusion",
      "viame.pytorch.botsort_tracker:BoTSORTTracker" ),
    ( "track_objects", "deepsort",
      "DeepSORT multi-object tracker with deep appearance features",
      "viame.pytorch.deepsort_tracker:DeepSORTTracker" ),
    ( "track_objects", "motr",
      "MOTR-style track-query transformer tracker with learned association",
      "viame.pytorch.motr_tracker:MOTRTracker" ),
    ( "track_objects", "sam3_tracker",
      "SAM3 (Segment Anything Model 3) based object tracker with text queries",
      "viame.pytorch.sam3_tracker:SAM3Tracker" ),
    ( "track_objects", "siammask",
      "SiamMask visual object tracker",
      "viame.pytorch.siammask_tracker:SiamMaskTracker" ),
    ( "track_objects", "srnn",
      "Structural RNN multi-object tracker",
      "viame.pytorch.srnn_tracker:SRNNTracker" ),
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
    ( "desc_augmentation",
      "Pytorch-Based Augmentation",
      "viame.pytorch.torchvision_augment_process:DataAugmentation" ),
    ( "pair_stereo_tracks_pytorch",
      "Pair stereo detections using deep descriptor cosine distance",
      "viame.pytorch.pair_stereo_tracks:PairStereoTracks" ),
    ( "pytorch_descriptors",
      "pytorch feature extraction",
      "viame.pytorch.torchvision_descriptors:ResNetDescriptors" ),
]
