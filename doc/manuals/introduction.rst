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


.. _Boost: http://www.boost.org/
.. _CDash: http://www.cdash.org/
.. _Eigen: http://eigen.tuxfamily.org/
.. _Fletch: https://github.com/Kitware/fletch
.. _Kitware: http://www.kitware.com/
.. _MAP-Tk: https://github.com/Kitware/maptk
.. _Travis CI: https://travis-ci.org/
.. _VIAME: https://github.com/Kitware/VIAME
.. _`Ceres Solver`: http://ceres-solver.org/
.. _OpenCV: http://opencv.org/
.. _VXL: https://github.com/vxl/vxl/