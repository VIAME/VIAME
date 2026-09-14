# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Python process registration package for the VIAME examples.

This is the documented form for a package that ships python processes: it
names them, and names the module and class each one lives in, without
importing anything. `kwiver.vital.plugins.discovery` registers each with a
constructor that imports the module the first time a pipeline asks for one.

To add your own process:
    1. Write it in its own module in this directory
    2. Add a line here naming it: ( name, description, "module:Class" )

See hello_world_detector.py for the process itself.
"""

__sprokit_process_declarations__ = [
    ( "hello_world_detector",
      "Example detector that logs a message for each image",
      "viame.examples.hello_world_detector:hello_world_detector" ),
    ( "hello_world_filter",
      "Example filter that logs a message and passes images through",
      "viame.examples.hello_world_filter:hello_world_filter" ),
]
