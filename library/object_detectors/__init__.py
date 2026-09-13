# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python implementations that belong to object_detectors."""

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
    ( "image_object_detector", "hough_circle",
      "Hough circle detector",
      "viame.object_detectors.hough_circle_detector:HoughCircleDetector" ),
]
