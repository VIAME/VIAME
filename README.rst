##################################################
                     KWIVER
##################################################
Kitware Image and Video Exploitation and Retrieval
==================================================

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
modular, run-time configurable systems.  This goal is achieved through
the three main components of KWIVER: Vital, Arrows, and Sprokit.

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

.. _`Ceres Solver`: http://ceres-solver.org/
.. _Eigen: http://eigen.tuxfamily.org/
.. _OpenCV: http://opencv.org/
.. _VXL: https://github.com/vxl/vxl/


Building KWIVER
===============

Fletch
------

KWIVER, especially Arrows, has a number of dependencies on 3rd party
open source libraries.  Most of these dependencies are optional
but useful in practice, and the number of dependencies is expected to
grow as we expand Arrows.  To make it easier to build KWIVER, especially
on systems like Microsoft Windows that do not have package manager,
Fletch_ was developed to gather, configure and build those packages
for use with KWIVER.  Fletch is a CMake_ based "super-build" that
takes care of most of the build details for you.

.. _Fletch: https://github.com/Kitware/fletch
.. _CMake: https://www.cmake.org

To build Fletch_, refer to the README file in that repository.

Running CMake
-------------

We recommend building kwiver out of the source directory to prevent
mixing source files with compiled products.  Create a build directory
in parallel with the kwiver source directory.  From the command line,
enter the empty build directory and run::

    $ ccmake /path/to/kwiver/source

where the path above is the location of your kwiver source tree.  The
ccmake tool allows for interactive selection of CMake options.
Alternatively, using the CMake GUI you can set the source and build
directories accordingly and press the "Configure" button.  When
building with Fletch it is preferable to set the ``fletch_DIR`` on the
command line like this::

    $ ccmake /path/to/kwiver/source -Dfletch_DIR=/path/to/fletch/install

Other CMake options can also be passed on the command line in this way
if desired.

CMake Options
-------------

The following are the most important CMake configuration options for KWIVER.

* CMAKE_BUILD_TYPE -- The compiler mode, usually Debug or Release
* CMAKE_INSTALL_PREFIX -- The path to where you want the kwiver build products to install
* KWIVER_ENABLE_ARROWS -- Enable algorithm implementation plugins
* KWIVER_ENABLE_C_BINDINGS -- Whether to build the Vital C bindings
* KWIVER_ENABLE_DOCS -- Turn on building the Doxygen documentation
* KWIVER_ENABLE_LOG4CPLUS -- Enable log4cplus logger back end
* KWIVER_ENABLE_PYTHON -- Enable the Vital Python bindings (requires KWIVER_ENABLE_C_BINDINGS)
* KWIVER_ENABLE_SPROKIT -- Enable the Stream Processing Toolkit
* KWIVER_ENABLE_TESTS -- Build the unit tests
* KWIVER_ENABLE_TOOLS -- Build the command line tools (e.g. plugin_explorer)
* fletch_DIR -- Install directory for the Fletch support packages.

There are many more options.  Specifically, there are numerous options
for third-party projects prefixed with ``KWIVER_ENABLE_`` that enable
building the Arrows plugins that depend on those projects.  When building
with the support of Fletch_ (set ``fletch_DIR``) the enable options for
packages built by Fletch should be turned on by default.


Dependencies
------------

Vital has minimal required dependencies (only Eigen_).
Sprokit additionally relies on Boost_.
Arrows and Sprokit processes are structured so that
the code that depends on an external package is in a directory with
the major dependency name (e.g. vxl, ocv). The dependencies can be
turned ON or OFF through CMake variables.

.. _Boost: http://www.boost.org/

Compiling
---------

Once your CMake command has completed and generated the build files,
compile in the standard way for your build environment.  On Linux
this is typically running ``make``.


Running KWIVER
==============

Once you've built KWIVER, you'll want to test that it's working on
your system.  From a command prompt execute the following command::

	source </path/to/kwiver/build>/install/setup_KWIVER.sh

Where `</path/to/kwiver/build>` is the actual path of your KWIVER
CMake build directory.

This will set up your PATH, PYTHONPATH and other environment variables
to allow KWIVER to work conveniently.


Vital
=====

Vital is an open source C++ collection of libraries and tools that
supply basic types and services to the Kitware KWIVER imagery tool
kit.

Overview of Directories
-----------------------

* CMake -- contains CMake helper scripts
* tests -- contains testing related support code
* vital -- contains the core library source and headers
* vital/algo -- contains abstract algorithm definitions
* vital/bindings -- contains C and Python bindings
* vital/config -- contains configuration support code
* vital/exceptions -- contains the exception class hierarchy
* vital/io -- contains the classes that support reading and writing core data types
* vital/kwiversys -- contains the code that supports the OS abstraction layer
* vital/logger -- contains the classes that provide logging support
* vital/plugin_loader --   contains the classes that provide plugin loading services
* vital/tests -- contains the main testing code
* vital/tools -- contains source for command line utilities
* vital/types -- contains the source for the core data types
* vital/util --   contains the source for general purpose utilities
* vital/video_metadata -- contains the classes that support video metadata


Contributing
============

For details on how to contribute to KWIVER, including code style and branch
naming conventions, please read `<CONTRIBUTING.rst>`_.


Getting Help
============

Please join the
`kwiver-users <http://public.kitware.com/mailman/listinfo/kwiver-users>`_
mailing list to discuss KWIVER or to ask for help with using KWIVER.
For less frequent announcements about KWIVER and projects built on KWIVER,
please join the
`kwiver-announce <http://public.kitware.com/mailman/listinfo/kwiver-announce>`_
mailing list.


Acknowledgements
================

The authors would like to thank AFRL/Sensors Directorate for their support
of this work via SBIR Contract FA8650-14-C-1820. The portions of this work
funded by the above contract are approved for public release via case number
88ABW-2017-2725.

The authors would like to thank IARPA for their support of this work via the
DIVA program.

The authors would like to thank NOAA for their support of this work via the
NOAA Fisheries Strategic Initiative on Automated Image Analysis.
