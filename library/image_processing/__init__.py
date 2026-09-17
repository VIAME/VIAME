# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python implementations that belong to image_processing.

The OpenCV replacements that came off arrows/ocv in phase 7, and the
multi-camera homography, alignment, optical flow and percentile
normalisation modules that came from `viame.core` in P2-T05.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
# `viame.plugins.discovery` turns each into a stand-in that imports
# its module the first time something asks for an instance.
#
# This replaced a `__vital_algorithm_register__` that imported every
# implementation module at startup, which was the only way the classes came
# to exist for the subclass walk that registers them. See P8-T10.
__vital_algorithm_declarations__ = [
    ( "detect_features", "ocv_SIFT",
      "OpenCV feature detection via the SIFT algorithm",
      "viame.image_processing.ocv_sift_surf:DetectFeaturesSIFT" ),
    ( "detect_features", "ocv_SURF",
      "OpenCV feature detection via the SURF algorithm",
      "viame.image_processing.ocv_sift_surf:DetectFeaturesSURF" ),
    ( "estimate_fundamental_matrix", "ocv",
      "Use OpenCV to estimate a fundamental matrix from feature matches.",
      "viame.image_processing.ocv_estimators:EstimateFundamentalMatrixOCV" ),
    ( "estimate_homography", "ocv",
      "Use OpenCV to estimate a homography from feature matches.",
      "viame.image_processing.ocv_estimators:EstimateHomographyOCV" ),
    ( "extract_descriptors", "ocv_SIFT",
      "OpenCV feature detection via the SIFT algorithm",
      "viame.image_processing.ocv_sift_surf:ExtractDescriptorsSIFT" ),
    ( "extract_descriptors", "ocv_SURF",
      "OpenCV feature detection via the SURF algorithm",
      "viame.image_processing.ocv_sift_surf:ExtractDescriptorsSURF" ),
    ( "image_filter", "equalize_via_percentiles_npy",
      "Numpy percentile normalization with configurable output format",
      "viame.image_processing.equalize_via_percentiles:EqualizeViaPercentiles" ),
    ( "image_filter", "ocv_color_correction",
      "Color correction algorithms: gamma, underwater compensation, gray world white balance",
      "viame.image_processing.ocv_color_correction:ApplyColorCorrection" ),
    ( "image_filter", "ocv_enhancer",
      "Simple illumination normalization using Lab space and CLAHE",
      "viame.image_processing.ocv_enhancer:EnhanceImages" ),
    ( "image_filter", "ocv_optical_flow",
      "Dense Farneback optical-flow image filter",
      "viame.image_processing.optical_flow:OpticalFlowFilter" ),
    ( "image_filter", "vxl_enhancer",
      "Simple illumination normalization using Lab space and CLAHE",
      "viame.image_processing.ocv_enhancer:VXLEnhancer" ),
    ( "match_features", "ocv_flann_based",
      "OpenCV feature matcher using FLANN (Approximate Nearest Neighbors)",
      "viame.image_processing.ocv_flann_matcher:MatchFeaturesFlannBased" ),
    ( "refine_detections", "ocv_grabcut",
      "Set detection segmentation masks using cv::grabCut",
      "viame.image_processing.ocv_segmenters:RefineDetectionsGrabCut" ),
    ( "refine_detections", "ocv_watershed",
      "Set detection segmentation masks using cv::watershed",
      "viame.image_processing.ocv_segmenters:RefineDetectionsWatershed" ),
]


# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ). A process registers by
# calling `process_factory.add_process( name, description, ctor )`, and the
# ctor is only ever called -- so a function that imports and constructs is as
# good as the class and costs nothing until a pipeline wants one.
__sprokit_process_declarations__ = [
    ( "ocv_fft_filter_based_on_ref",
      "Filter image in the frequency based on some template",
      "viame.image_processing.fft_filter_based_on_ref:filter_based_on_ref_process" ),
    ( "align_cameras",
      "Multi-image-pair camera-to-camera registration (MINIMA-LoFTR)",
      "viame.image_processing.align_cameras_process:AlignCamerasProcess" ),
    ( "blank_out_frames",
      "Blank out frames with no object detections on them",
      "viame.image_processing.utility_processes:blank_out_frames" ),
    ( "many_image_stabilizer",
      "Simultaneous multi-image stabilization",
      "viame.image_processing.stabilize_many_images:ManyImageStabilizer" ),
    ( "multicam_homog_blackout",
      "Black out previously-observed regions of registered multi-camera imagery",
      "viame.image_processing.multicam_homog_blackout:MulticamHomogBlackout" ),
    ( "multicam_homog_mosaic",
      "Per-timestep tile mosaic of registered multi-camera imagery",
      "viame.image_processing.multicam_homog_mosaic:MulticamHomogMosaic" ),
    ( "percentile_norm_npy_16_to_8bit",
      "A specialized percentile normalization method",
      "viame.image_processing.utility_processes:percentile_norm_npy_16_to_8bit" ),
]
