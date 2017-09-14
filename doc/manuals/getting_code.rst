Getting KWIVER
===============

Refer to the `repository <https://github.com/Kitware/kwiver>`_ for getting and building the KWIVER code base

Code Structure and Provided Functionality
-----------------------------------------

Below is a summary of the key directories in KWIVER and a brief summary of the content they contain.


================ ===========================================================
CMake_           CMake helper scripts
arrows_          The algorithm plugin modules
doc_             Documentation, manuals, release notes
examples_        Examples for running KWIVER (currently out of date)
extras_          Extra utilities (e.g. instrumentation)
sprokit_         Stream processing toolkit
tests_           Testing related support code
vital_           Core libraries source and headers
================ ===========================================================

.. _CMake: https://github.com/Kitware/kwiver/tree/master/CMake
.. _arrows: https://github.com/Kitware/kwiver/tree/master/arrows
.. _doc: https://github.com/Kitware/kwiver/tree/master/doc
.. _examples: https://github.com/Kitware/kwiver/tree/master/examples
.. _extras: https://github.com/Kitware/kwiver/tree/master/extras
.. _sprokit: https://github.com/Kitware/kwiver/tree/master/sprokit
.. _tests: https://github.com/Kitware/kwiver/tree/master/tests
.. _vital: https://github.com/Kitware/kwiver/tree/master/vital

Vital Subdirectories
~~~~~~~~~~~~~~~~~~~~

========================= =========================================================
`vital/algo`_             Abstract algorithm definitions
`vital/bindings`_         C and Python bindings
`vital/config`_           Configuration support code
`vital/exceptions`_       Exception class hierarchy
`vital/io`_               Classes that support reading and writing core data types
`vital/kwiversys`_        Code that supports the OS abstraction layer
`vital/logger`_           Classes that provide logging support
`vital/plugin_loader`_    Classes that provide plugin loading services
`vital/tests`_            Unit tests for vital code
`vital/tools`_            Source for command line utilities
`vital/types`_            Classes for the core data types
`vital/util`_             Source for general purpose utility functions
`vital/video_metadata`_   Classes that support video metadata
========================= =========================================================

.. _`vital/algo`: https://github.com/Kitware/kwiver/tree/master/vital/algo
.. _`vital/bindings`: https://github.com/Kitware/kwiver/tree/master/vital/bindings
.. _`vital/config`: https://github.com/Kitware/kwiver/tree/master/vital/config
.. _`vital/exceptions`: https://github.com/Kitware/kwiver/tree/master/vital/exceptions
.. _`vital/io`: https://github.com/Kitware/kwiver/tree/master/vital/io
.. _`vital/kwiversys`: https://github.com/Kitware/kwiver/tree/master/vital/kwiversys
.. _`vital/logger`: https://github.com/Kitware/kwiver/tree/master/vital/logger
.. _`vital/plugin_loader`: https://github.com/Kitware/kwiver/tree/master/vital/plugin_loader
.. _`vital/tests`: https://github.com/Kitware/kwiver/tree/master/vital/tests
.. _`vital/tools`: https://github.com/Kitware/kwiver/tree/master/vital/tools
.. _`vital/types`: https://github.com/Kitware/kwiver/tree/master/vital/types
.. _`vital/util`: https://github.com/Kitware/kwiver/tree/master/vital/util
.. _`vital/video_metadata`: https://github.com/Kitware/kwiver/tree/master/vital/video_metadata

Arrows Subdirectories
~~~~~~~~~~~~~~~~~~~~~

===================== =========================================================
`arrows/burnout`_     [*Experimental*] Pixel classifiers for heads-up display
                      detection an related tasks using Vibrant_.
`arrows/ceres`_       Algorithms for bundle adjustment and optimization using
                      `Ceres Solver`_.
`arrows/core`_        Algorithms implemented with no additional third party
                      dependencies beyond what Vital uses (Eigen).
`arrows/darknet`_     [*Experimental*] Object detection with the Darknet_ YOLO CNN.
`arrows/matlab`_      An interface for running Matlab code KWIVER 
`arrows/ocv`_         Algorithms implemented using OpenCV_.
                      Includes feature detectors and descriptor, homography
                      and fundamental matrix estimation, image IO, and more.
`arrows/proj`_        Geographic conversion functions implemented with PROJ4_.
`arrows/uuid`_        [*Experimental*] Implementation of unique IDs using libuuid
`arrows/viscl`_       [*Experimental*] Algorithms using VisCL to implement
                      algorithms in OpenCL 
`arrows/vxl`_         Algorithms implemnted using the VXL_ libraries.
                      Includes bundle adjustment, homography estimation, video
                      file reading, and more.
===================== =========================================================

.. _`arrows/burnout`: https://github.com/Kitware/kwiver/tree/master/arrows/burnout
.. _`arrows/ceres`: https://github.com/Kitware/kwiver/tree/master/arrows/ceres
.. _`arrows/core`: https://github.com/Kitware/kwiver/tree/master/arrows/core
.. _`arrows/darknet`: https://github.com/Kitware/kwiver/tree/master/arrows/darknet
.. _`arrows/matlab`: https://github.com/Kitware/kwiver/tree/master/arrows/matlab
.. _`arrows/ocv`: https://github.com/Kitware/kwiver/tree/master/arrows/ocv
.. _`arrows/proj`: https://github.com/Kitware/kwiver/tree/master/arrows/proj
.. _`arrows/uuid`: https://github.com/Kitware/kwiver/tree/master/arrows/uuid
.. _`arrows/viscl`: https://github.com/Kitware/kwiver/tree/master/arrows/viscl
.. _`arrows/vxl`: https://github.com/Kitware/kwiver/tree/master/arrows/vxl

Sprokit Subdirectories
~~~~~~~~~~~~~~~~~~~~~~

====================== =========================================================
`sprokit/cmake`_       CMake helper scripts specific to Sprokit
`sprokit/conf`_        Configuration files CMake will tailor to the build system
                       machine and directory structure
`sprokit/doc`_         Further documenation related to sprokit
`sprokit/extra`_       General scripts, hooks, and cofigurations for use with 3rd
                       party tools (e.g. git and vim)
`sprokit/pipelines`_   Example pipeline files demonstrating the execution of
                       various arrows through sprokit
`sprokit/processes`_   General utility processess that encapsulate various arrows
                       for core funcionality  
`sprokit/src`_         Core infrastructure code for defining, chaining, and
                       executing Sprokit processes 
`sprokit/tests`_       Sprokit unit tests
====================== =========================================================


.. _`sprokit/cmake`: https://github.com/Kitware/kwiver/tree/master/sprokit/cmake
.. _`sprokit/conf`: https://github.com/Kitware/kwiver/tree/master/sprokit/conf
.. _`sprokit/doc`: https://github.com/Kitware/kwiver/tree/master/sprokit/doc
.. _`sprokit/extra`: https://github.com/Kitware/kwiver/tree/master/sprokit/extra
.. _`sprokit/pipelines`: https://github.com/Kitware/kwiver/tree/master/sprokit/pipelines
.. _`sprokit/processes`: https://github.com/Kitware/kwiver/tree/master/sprokit/processes
.. _`sprokit/src`: https://github.com/Kitware/kwiver/tree/master/sprokit/src
.. _`sprokit/tests`: https://github.com/Kitware/kwiver/tree/master/sprokit/tests


.. _`Ceres Solver`: http://ceres-solver.org/
.. _Vibrant: https://github.com/Kitware/vibrant
.. _Darknet: https://pjreddie.com/darknet/yolo/
.. _OpenCV: http://opencv.org/
.. _PROJ4: http://proj4.org/
.. _VXL: https://github.com/vxl/vxl/