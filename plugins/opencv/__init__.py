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
# Empty since P2-T05: `ocv_watershed` is `viame.segmentation`,
# `gmm_motion_detector` is `viame.object_detectors` and
# `ocv_fft_filter_based_on_ref` is `viame.image_processing`. What is left in
# this package is `stereo_utils`, `stereo_pipeline` and `prior_coverage_opencv`,
# which nothing registers, and the demo scripts.
__vital_algorithm_declarations__ = []

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
__sprokit_process_declarations__ = []
