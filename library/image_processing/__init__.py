# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python implementations that belong to image_processing."""

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
    ( "image_filter", "ocv_color_correction",
      "Color correction algorithms: gamma, underwater compensation, gray world white balance",
      "viame.image_processing.ocv_color_correction:ApplyColorCorrection" ),
    ( "image_filter", "ocv_enhancer",
      "Simple illumination normalization using Lab space and CLAHE",
      "viame.image_processing.ocv_enhancer:EnhanceImages" ),
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
