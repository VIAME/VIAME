Introduction
============

The KWIVER toolkit is a collection of software tools designed to
tackle challenging image and video analysis problems and other related
challenges. Recently started by Kitware’s Computer Vision and
Scientific Visualization teams, KWIVER is an ongoing effort to
transition technology developed over multiple years to the open source
domain to further research, collaboration, and product development.
KWIVER is a collection of C++ libraries with C and Python bindings
and uses an permissive `BSD License <LICENSE>`_.

One of the primary design goals of KWIVER is to make it easier to pull
together algorithms from a wide variety of third-party, open source
image and video processing projects and integrate them into highly
modular, run-time configurable systems. 

This goal is achieved through the three main components of KWIVER: Vital, Arrows, and Sprokit.

Vital
-----
Vital is core of KWIVER and is designed to provide data and algorithm
abstractions with minimal library dependencies.  Vital only depends on
the C++ standard library and the header-only Eigen_ library.  Vital defines
the core data types and abstract interfaces for core vision algorithms
using these types.  Vital also provides various system utility functions
like logging, plugin management, and configuration file handling.  Vital
does **not** provide implementations of the abstract algorithms.
Implementations are found in Arrows and are loaded dynamically at run-time
via plugins.

The design of KWIVER allows end-user applications to link only against
the Vital libraries and have minimal hard dependencies.  One can then
dynamically add algorithmic capabilities, with new dependencies, via
plugins without needing to recompile Vital or the application code.
Only Vital is built by default when building KWIVER without enabling
any options in CMake.

Arrows
------
Arrows is the collection of plugins that provides implementations of the
algorithms declared in Vital.  Each arrow can be enabled or disabled
in build process through CMake options.  Most arrows bring in additional
third-party dependencies and wrap the capabilities of those libraries
to make them accessible through the Vital APIs.  The code in Arrows
also converts or wrap data types from these external libraries into
Vital data types.  This allows interchange of data between algorithms
from different arrows using Vital types as the intermediary.

Capabilities are currently organized into Arrows based on what third
party library they require.  However, this arrangement is not required
and may change as the number of algorithms and arrows grows.  Some
arrows, like `core <arrows/core>`_, require no additional dependencies.
Some examples of the provided Arrows are:

* `ocv <arrows/ocv>`__ - provides algorithms from OpenCV_
* `ceres <arrows/ceres>`__ - provides algorithms from `Ceres Solver`_
* `vxl <arrow/vxl>`__ - provides algorithms from VXL_

Sprokit
-------
Sprokit is a "**S**\ tream **Pro**\ cessing Tool\ **kit**" that provides
infrastructure for chaining together algorithms into pipelines for
processing streaming data sources.  The most common use case of Sprokit
is for video processing, but Sprokit is data type agnostic and could be
used for any type of streaming data.  Sprokit allows the user to dynamically
connect and configure a pipeline by chaining together processing nodes
called "processes" into a directed graph with data sources and sinks.
Sprokit schedules the jobs to run each process and keep data flowing through
pipeline.  Sprokit also allows processes written in Python to be
interconnected with those written in C++.

Code Structure and Provided Functionality
=========================================

Below is a summary of the key directories in KWIVER and a brief summary of
the content they contain.


================ ===========================================================
`<CMake>`_       CMake helper scripts
`<arrows>`_      The algorithm plugin modules
`<doc>`_         Documentation, manuals, release notes
`<examples>`_    Examples for running KWIVER (currently out of date)
`<extras>`_      Extra utilities (e.g. instrumentation)
`<sprokit>`_     Stream processing toolkit
`<tests>`_       Testing related support code
`<vital>`_       Core libraries source and headers
================ ===========================================================

Vital
-----

========================= =========================================================
`<vital/algo>`_           Abstract algorithm definitions
`<vital/bindings>`_       C and Python bindings
`<vital/config>`_         Configuration support code
`<vital/exceptions>`_     Exception class hierarchy
`<vital/io>`_             Classes that support reading and writing core data types
`<vital/kwiversys>`_      Code that supports the OS abstraction layer
`<vital/logger>`_         Classes that provide logging support
`<vital/plugin_loader>`_  Classes that provide plugin loading services
`<vital/tests>`_          Unit tests for vital code
`<vital/tools>`_          Source for command line utilities
`<vital/types>`_          Classes for the core data types
`<vital/util>`_           Source for general purpose utility functions
`<vital/video_metadata>`_ Classes that support video metadata
========================= =========================================================

Arrows
------

===================== =========================================================
`<arrows/burnout>`_   [*Experimental*] Pixel classifiers for heads-up display
                      detection an related tasks using Vibrant_.
`<arrows/ceres>`_     Algorithms for bundle adjustment and optimization using
                      `Ceres Solver`_.
`<arrows/core>`_      Algorithms implemented with no additional third party
                      dependencies beyond what Vital uses (Eigen).
`<arrows/darknet>`_   [*Experimental*] Object detection with the Darknet_ YOLO CNN.
`<arrows/matlab>`_    An interface for running Matlab code KWIVER 
`<arrows/ocv>`_       Algorithms implemented using OpenCV_.
                      Includes feature detectors and descriptor, homography
                      and fundamental matrix estimation, image IO, and more.
`<arrows/proj>`_      Geographic conversion functions implemented with PROJ4_.
`<arrows/uuid>`_      [*Experimental*] Implementation of unique IDs using libuuid
`<arrows/viscl>`_     [*Experimental*] Algorithms using VisCL to implement
                      algorithms in OpenCL 
`<arrows/vxl>`_       Algorithms implemnted using the VXL_ libraries.
                      Includes bundle adjustment, homography estimation, video
                      file reading, and more.
===================== =========================================================

Sprokit
-------

====================== =========================================================
`<sprokit/cmake>`_     CMake helper scripts specific to Sprokit
`<sprokit/conf>`_      Configuration files CMake will tailor to the build system
                       machine and directory structure
`<sprokit/doc>`_       Further documenation related to sprokit
`<sprokit/extra>`_     General scripts, hooks, and cofigurations for use with 3rd
                       party tools (e.g. git and vim)
`<sprokit/pipelines>`_ Example pipeline files demonstrating the execution of
                       various arrows through sprokit
`<sprokit/processes>`_ General utility processess that encapsulate various arrows
                       for core funcionality  
`<sprokit/src>`_       Core infrastructure code for defining, chaining, and
                       executing Sprokit processes 
`<sprokit/tests>`_     Sprokit unit tests
====================== =========================================================

