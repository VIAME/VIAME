# vital #

Vital is an open source C++ collection of libraries and tools that supply basic types and services to the Kitware KWIVER imagery tool kit.

## Overview of Directories


* CMake -- contains CMake helper scripts
* tests -- contains testing related support code
* vital -- contains the core library source and headers
* vital/algo -- contains abstract algorithm definitions
* vital/bindings -- contains 'c' and Python bindings
* vital/config -- contains configuration support code
* vital/exceptions -- contains the exception class hierarchy
* vital/io -- contains the classes that support reading and writing core data types
* vital/kwiversys -- contains the code that supports the OS abstraction layer
* vital/logger -- contains the classes tha tprovide logging support
* vital/tests -- contains the main testing code
* vital/tools -- contains source for command line utilities
* vital/types -- contains the source for the core data types

## Building vital

vital uses CMake (www.cmake.org) for easy cross-platform compilation. The minimum required version of CMake is 3.0, but newer versions are recommended.

# Running CMake

We recommend building vital out of the source directory to prevent mixing source files with compiled products.  Create a build directory in parallel with the vital source directory.  From the command line, enter the empty build directory and run

    $ ccmake /path/to/vital/source

where the path above is the location of your vital source tree.  The ccmake tool allows for interactive selection of CMake options.  Alternatively, using the CMake GUI you can set the source and build directories accordingly and press the "Configure" button.


# CMake Options

* CMAKE_BUILD_TYPE -- The compiler mode, usually Debug or Release
* CMAKE_INSTALL_PREFIX -- The path to where you want the vital build products to install
* KWIVER_BUILD_SHARED -- Build shared or static libraries
* KWIVER_ENABLE_DOCS -- Turn on building the Doxygen documentation
* KWIVER_ENABLE_LOG4CXX -- Enable log4cxx logger back end
* KWIVER_ENABLE_PYTHON -- Enable the python bindings
* KWIVER_ENABLE_TESTS -- Build the unit tests
* KWIVER_LIB_SUFFIX -- String suffix appended to the library directory name we install into.
* KWIVER_USE_BUILD_PLUGIN_DIR -- When building the plugin manager, wether to include the build directory in the sesarch path.
* VITAL_ENABLE_C_LIB -- Whether to build the c bindings
* fletch_DIR -- Build directory for the Fletch support packages.

## Dependencies

Vital has minimal required dependencies at the core level.  Enabling add-on
modules adds additional capabilities as well as additional dependencies.

### Required

These dependencies are supplied (or will be supplied) by the Fletch package of 3rd party dependencies.

[Boost](http://www.boost.org/) (>= v1.55)
[Eigen](http://eigen.tuxfamily.org/) (>= 3.0)
[log4cxx] (https://logging.apache.org/log4cxx/) (>= 0.10.0)
[Apache Runtime] (https://apr.apache.org/)
