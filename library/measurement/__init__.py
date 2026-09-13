# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python implementations that belong to measurement."""


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
    ( "compute_stereo_depth_map", "ocv_stereo_disparity",
      "OpenCV stereo disparity map computation using BM or SGBM",
      "viame.measurement.ocv_stereo_disparity:ComputeStereoDisparity" ),
    ( "image_object_detector", "ocv_detect_calibration_targets",
      "Detect calibration targets (checkerboard or dots) with OpenCV",
      "viame.measurement.ocv_calibration_targets:DetectCalibrationTargets" ),
    ( "optimize_cameras", "ocv_optimize_stereo_cameras",
      "Camera optimizer for stereo configurations.",
      "viame.measurement.ocv_optimize_stereo_cameras:OptimizeStereoCameras" ),
]

# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ). A process registers by
# calling `process_factory.add_process( name, description, ctor )`, and the
# ctor is only ever called -- so a function that imports and constructs is as
# good as the class and costs nothing until a pipeline wants one. See P8-T10.
__sprokit_process_declarations__ = [
    ( "ocv_calibrate_single_camera",
      "Estimate one camera's intrinsics from a calibration target track set",
      "viame.measurement.ocv_calibrate_single_camera:CalibrateSingleCamera" ),
]
