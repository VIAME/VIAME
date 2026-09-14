# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""@template_dir@: what this package provides, without importing any of it.

`kwiver.vital.plugins.discovery` reads these lists and registers a stand-in
for each entry that imports its module the first time a pipeline asks for an
instance, so a detector that imports torch costs nothing at startup.

Each algorithm entry is ( interface, name, description, "module:Class" ).
"""

__vital_algorithm_declarations__ = [
    ( "image_object_detector", "@template@",
      "@template@ detector",
      "@template_dir@.@template@_detector:@template@Detector" ),
]

# ( name, description, "module:Class" ) for any sprokit processes
__sprokit_process_declarations__ = []
