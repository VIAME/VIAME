# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Reading and writing detections and tracks, in python.

The COCO readers and writers, which came from `viame.core` in P2-T04, and
`utilities_coco` that they share. The C++ side of this library also exposes
`viame.file_io._opencv_yaml`, the OpenCV FileStorage subset, so that the
golden recording of what `cv::FileStorage` parsed can be replayed against
the reader that replaced it.
"""


# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
# `kwiver.vital.plugins.discovery` turns each into a stand-in that imports
# its module the first time something asks for an instance.
__vital_algorithm_declarations__ = [
    ( "detected_object_set_input", "coco",
      "Read detections from COCO-style JSON format",
      "viame.file_io.read_detected_object_set_coco:ReadDetectedObjectSetCoco" ),
    ( "detected_object_set_output", "coco",
      "Write detections to COCO-style JSON format",
      "viame.file_io.write_detected_object_set_coco:WriteDetectedObjectSetCoco" ),
    ( "read_object_track_set", "coco",
      "Read object tracks from COCO-style JSON format",
      "viame.file_io.read_object_track_set_coco:ReadObjectTrackSetCoco" ),
    ( "write_object_track_set", "coco",
      "Write object tracks to COCO-style JSON format with track_id field",
      "viame.file_io.write_object_track_set_coco:WriteObjectTrackSetCoco" ),
]
