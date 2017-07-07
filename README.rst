##################################################
                     KWIVER
##################################################
--------------------------------------------------
Kitware Image and Video Exploitation and Retrieval
--------------------------------------------------

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

KWIVER has (and will have more) a number of dependencies on 3rd party
Open Source libraries.  To make it easier to build KWIVER, especially
on systems like Microsoft Windows that don't have package manager,
Fletch_ was developed to gather, configure and build those packages
for use with KWIVER.  Fletch is a CMake_ based "super-build" that
takes care of most of the build details for you.

.. _Fletch: https://github.com/Kitware/fletch
.. _CMake: https://www.cmake.org

To build Fletch, refer to the README file in that repository.


kwiver
------

Once Fletch has been built, it's possible to build the `kwiver`
repository as well.  This repo is also a CMake build and can be
fetched with this command::

	git clone https://github.com/Kitware/kwiver.git

The build can be configured with this command::

	cmake -DKWIVER_ENABLE_PYTHON:BOOL=ON -DKWIVER_ENABLE_ARROWS:BOOL=ON -DKWIVER_ENABLE_PROCESSES:BOOL=ON -DKWIVER_ENABLE_TESTS:BOOL=ON -DKWIVER_ENABLE_TOOLS:BOOL=ON -DKWIVER_USE_BUILD_TREE:BOOL=ON -DKWIVER_ENABLE_VXL:BOOL=ON -DKWIVER_ENABLE_SPROKIT:BOOL=ON -DKWIVER_ENABLE_OPENCV:BOOL=ON -Dfletch_DIR:PATH=/path/to/fletch/build/directory /path/to/kwiver/source/directory

As with Fletch, if you want to specify a particular Python
installation (such as Anaconda) use the the `-DPYTHON...` command
arguments as outlined in the Fletch section.

Once your `cmake` command has completed, use `make` (on Linux) to build it.


Running KWIVER
==============

Once you've built KWIVER, you'll want to test that it's working on
your system.  From a command prompt execute the following command::

	source </path/to/kwiver/build>/install/setup_KWIVER.sh

Where `</path/to/kwiver/build>` is the actual path of your KWIVER CMake build directory.

This will set up your PATH, PYTHONPATH and other environment variables
to allow KWIVER to work conveniently.

The central component of KWIVER is vital which supplies basic data
types and fundimental alrogithms.  In addition, we use sprokit's
pipelining facilities to manage, integrate and run many of KWIVER's
modules and capabilities.  To see what modules (called processes in
sprockit) are available, run the following command::

    $ plugin_explorer --process -b

Here's a typical list of modules (note that as KWIVER expands, this
list is likely to grow):

---- All process Factories

Factories that create type "sprokit::process"
    Process type: frame_list_input          Reads a list of image file names and generates stream of images and
       associated time stamps

    Process type: stabilize_image          Generate current-to-reference image homographies

    Process type: detect_features          Detect features in an image that will be used for stabilization

    Process type: extract_descriptors          Extract descriptors from detected features

    Process type: feature_matcher          Match extracted descriptors and detected features

    Process type: compute_homography          Compute a frame to frame homography based on tracks

    Process type: compute_stereo_depth_map          Compute a stereo depth map given two frames

    Process type: draw_tracks          Draw feature tracks on image

    Process type: read_d_vector          Read vector of doubles

    Process type: refine_detections          Refines detections for a given frame

    Process type: image_object_detector          Apply selected image object detector algorithm to incoming images.

    Process type: image_filter          Apply selected image filter algorithm to incoming images.

    Process type: image_writer          Write image to disk.

    Process type: image_file_reader          Reads an image file given the file name.

    Process type: detected_object_input          Reads detected object sets from an input file. Detections read from the
       input file are grouped into sets for each image and individually
       returned.

    Process type: detected_object_output          Writes detected object sets to an output file. All detections are written
       to the same file.

    Process type: detected_object_filter          Filters sets of detected objects using the detected_object_filter
       algorithm.

    Process type: video_input          Reads video files and produces sequential images with metadata per frame.

    Process type: draw_detected_object_set          Draws border around detected objects in the set using the selected
       algorithm.

    Process type: track_descriptor_input          Reads track descriptor sets from an input file.

    Process type: track_descriptor_output          Writes track descriptor sets to an output file. All descriptors are
       written to the same file.

    Process type: image_viewer          Display input image and delay

    Process type: draw_detected_object_boxes          Draw detected object boxes on images.

    Process type: collate          Collates data from multiple worker processes

    Process type: distribute          Distributes data to multiple worker processes

    Process type: pass          Pass a data stream through

    Process type: sink          Ignores incoming data

    Process type: any_source          A process which creates arbitrary data

    Process type: const          A process wth a const flag

    Process type: const_number          Outputs a constant number

    Process type: data_dependent          A process with a data dependent type

    Process type: duplicate          A process which duplicates input

    Process type: expect          A process which expects some conditions

    Process type: feedback          A process which feeds data into itself

    Process type: flow_dependent          A process with a flow dependent type

    Process type: multiplication          Multiplies numbers

    Process type: multiplier_cluster          A constant factor multiplier cluster

    Process type: mutate          A process with a mutable flag

    Process type: numbers          Outputs numbers within a range

    Process type: orphan_cluster          A dummy cluster

    Process type: orphan          A dummy process

    Process type: print_number          Print numbers to a file

    Process type: shared          A process with the shared flag

    Process type: skip          A process which skips input data

    Process type: tagged_flow_dependent          A process with a tagged flow dependent types

    Process type: take_number          Print numbers to a file

    Process type: take_string          Print strings to a file

    Process type: tunable          A process with a tunable parameter

    Process type: input_adapter          Source process for pipeline. Pushes data items into pipeline ports. Ports
       are dynamically created as needed based on connections specified in the
       pipeline file.

    Process type: output_adapter          Sink process for pipeline. Accepts data items from pipeline ports. Ports
       are dynamically created as needed based on connections specified in the
       pipeline file.

    Process type: template          Description of process. Make as long as necessary to fully explain what
       the process does and how to use it. Explain specific algorithms used,
       etc.

    Process type: kw_archive_writer          Writes kw archives

    Process type: test_python_process          A test Python process

    Process type: pyprint_number          A Python process which prints numbers

This is the list of modules that can be included in a Sprokit
pipeline.  We're going to use the `numbers` module and the the
`print_number` module to create a very simple pipeline.  To learn more
about the `numbers` module we'll again use `plugin_explorer` this time
to get details on a particular module.  For `numbers` we'll use the
following command::

    $ plugin_explorer --process --type numbers -d --config

    Factories that create type "sprokit::process"

      Process type: numbers
      Description:        Outputs numbers within a range

        Properties: _no_reentrant,
        -- Configuration --
        Name       : end
        Default    : 100
        Description: The value to stop counting at.
        Tunable    : no

        Name       : start
        Default    : 0
        Description: The value to start counting at.
        Tunable    : no

      Input ports:
      Output ports:
        Name       : number
        Type       : integer
        Flags      : _required,
        Description: Where the numbers will be available.

And for `print_number`, we'll use::

    $ plugin_explorer --process --type print_number -d --config

    Factories that create type "sprokit::process"

      Process type: print_number
      Description:        Print numbers to a file

        Properties: _no_reentrant,
        -- Configuration --
        Name       : output
        Default    :
        Description: The path of the file to output to.
        Tunable    : no

      Input ports:
        Name       : number
        Type       : integer
        Flags      : _required,
        Description: Where numbers are read from.

      Output ports:


The output of these commands tells us enough about each process to
construct a Sprockit ".pipe" file that defines a processing pipeline.
In particular we'll need to know how to configure each process (the
"Configuration") and how they can be hooked together (the input and
output "Ports").

KWIVER comes with a sample
[sprokit/pipelines/number_flow.pipe](sprokit/pipelines/number_flow.pipe)
file that configures and connects the pipeline so that the `numbers`
process will generate a set of integers from 1 to 99 and the
`print_number` process will write those to a file called
`numbers.txt`.  Of particular interest is the section at the end of
the file that actually "hooks up" the pipeline.

To run the pipeline, we'll use the Sprokit `pipeline_runner` command::

    $ pipeline_runner -p </path/to/kwiver/source>/sprokit/pipelines/number_flow.pipe

After the pipeline completes, you should find a file, `numbers.txt`, in your working directory.


Python Processes
----------------

One of KWIVER's great strengths (as provided by sprokit) is the
ability to create hybrid pipelines which combine C++ and Python
processes in the same pipeline.  This greatly facilitates prototyping
complex processing pipelines.  To test this out we'll still use the
`numbers` process, but we'll use a Python version of the
`print_number` process called `kw_print_number_process` the code for
which can be seen in
[sprokit/processes/python/kw_print_number_process.py](sprokit/processes/python/kw_print_number_process.py).
As usual, we can lean about this process with the following command::

    $ plugin_explorer --process --type kw_print_number_process -d --config

    Process type: kw_print_number_process
      Description: A Simple Kwiver Test Process
      Properties: _no_reentrant, _python
    Configuration:
      Name       : output
      Default    : .
      Description: The path for the output file.
      Tunable    : no

    Input ports:
      Name       : input
      Type       : integer
      Flags      : _required
      Description: Where numbers are read from.

    Output ports:

As you can see, the process is very similar to the C++ `print_number`
process.  As a result, the [".pipe" file is very
similar](sprokit/pipelines/number_flow_python.pipe).

In order to get around limitations imposed by the Python Global
Interpreter Lock, we'll use a different Sprokit scheduler for this
pipeline.  The `pythread_per_process` scheduler which does essentially
what it says: it creates a Python thread for every process in the
pipeline::

	pipeline_runner -S pythread_per_process -p </path/to/kwiver/source>/sprokit/pipelines/number_flow_python.pipe>

As with the previous pipeline, the numbers will be written to an output file, this time `numbers_from_python.txt`



vital
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
directories accordingly and press the "Configure" button.


CMake Options
=============

* CMAKE_BUILD_TYPE -- The compiler mode, usually Debug or Release
* CMAKE_INSTALL_PREFIX -- The path to where you want the kwiver build products to install
* KWIVER_BUILD_SHARED -- Build shared or static libraries
* KWIVER_ENABLE_ARROWS -- Enable algorithm implementations
* KWIVER_ENABLE_DOCS -- Turn on building the Doxygen documentation
* KWIVER_ENABLE_LOG4CLUS -- Enable log4cplus logger back end
* KWIVER_ENABLE_PYTHON -- Enable the python bindings
* KWIVER_ENABLE_TESTS -- Build the unit tests
* KWIVER_USE_BUILD_TREE -- When building the plugin manager, whether to include the build directory in the search path.
* KWIVER_ENABLE_C_BINDINGS -- Whether to build the C bindings
* fletch_DIR -- Build directory for the Fletch support packages.

There are many more options

Dependencies
------------

Vital has minimal required dependencies. Sprokit pipeline framework
relies on boost.  Arrows and sprokit processes are structured so that
the code that depends on an external package is in a directory with
the major dependency name (e.g. vxl, ocv). The dependencies can be
turned ON or OFF through CMake variables.

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
