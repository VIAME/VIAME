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
    ( "segment_via_points", "ocv_watershed",
      "OpenCV watershed-based point segmentation algorithm",
      "viame.opencv.watershed_segmenter:WatershedSegmenter" ),
]

# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ). A process registers by
# calling `process_factory.add_process( name, description, ctor )`, and the
# ctor is only ever called -- so a function that imports and constructs is as
# good as the class and costs nothing until a pipeline wants one. See P8-T10.
# `ocv_multimodal_registration` is deliberately absent. Its module imports
# and its class exists, but constructing it raises `type trait name
# "homography" not registered`, so it has never been in the compatibility
# baseline -- see open question 2.13. Declaring it would put a name in the
# registry that cannot be built, which is the thing that question is about.
__sprokit_process_declarations__ = [
    ( "gmm_motion_detector",
      "preliminatry fish detection",
      "viame.opencv.stereo_processes:GMMDetectFishProcess" ),
    ( "ocv_fft_filter_based_on_ref",
      "Filter image in the frequency based on some template",
      "viame.opencv.fft_filter_based_on_ref:filter_based_on_ref_process" ),
]
