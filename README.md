# vital #

Vital is an open source C++ collection of libraries and tools that supply basic types and services to the Kitware KWIVER imagery tool kit.

## Overview of Directories ##


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

## Building vital ##

vital uses CMake (www.cmake.org) for easy cross-platform compilation. The minimum required version of CMake is 3.0, but newer versions are recommended.

# Running CMake #

We recommend building vital out of the source directory to prevent mixing source files with compiled products.  Create a build directory in parallel with the vital source directory.  From the command line, enter the empty build directory and run

    $ ccmake /path/to/vital/source

where the path above is the location of your vital source tree.  The ccmake tool allows for interactive selection of CMake options.  Alternatively, using the CMake GUI you can set the source and build directories accordingly and press the "Configure" button.


# CMake Options #

* CMAKE_BUILD_TYPE -- The compiler mode, usually Debug or Release
* CMAKE_INSTALL_PREFIX -- The path to where you want the vital build products to install
* VITAL_BUILD_SHARED -- Build shared or static libraries
* VITAL_ENABLE_DOCS -- Turn on building the Doxygen documentation
* VITAL_ENABLE_LOG4CXX -- Enable log4cxx logger back end
* VITAL_ENABLE_PYTHON -- Enable the python bindings
* VITAL_ENABLE_TESTS -- Build the unit tests
* KWIVER_USE_BUILD_PLUGIN_DIR -- When building the plugin manager, wether to include the build directory in the sesarch path.
* VITAL_ENABLE_C_LIB -- Whether to build the c bindings
* fletch_DIR -- Build directory for the Fletch support packages.

## Dependencies ##

Vital has minimal required dependencies at the core level.  Enabling add-on
modules adds additional capabilities as well as additional dependencies.

### Required ##

These dependencies are supplied (or will be supplied) by the Fletch package of 3rd party dependencies.

[Boost](http://www.boost.org/) (>= v1.55)
[Eigen](http://eigen.tuxfamily.org/) (>= 3.0)
[log4cxx] (https://logging.apache.org/log4cxx/) (>= 0.10.0)
[Apache Runtime] (https://apr.apache.org/)

Development
===========

When developing on vital, please keep to the prevailing style of the code.
Some guidelines to keep in mind for different languages in the codebase are as
follows:

CMake
-----

  * 2-space indentation
  * Lowercase for private variables
  * Uppercase for user-controlled variables
  * Prefer functions over macros
    - They have variable scoping and debugging them is much easier
  * Prefer ``foreach (IN LISTS)`` and ``list(APPEND)``
  * Prefer ``kwiver_configure_file`` over ``configure_file`` when possible to
    avoid adding dependencies to the configure step
  * Use the ``kwiver_`` wrappers of common commands (e.g., ``add_library``,
    ``add_test``, etc.) as they automatically Do The Right Thing with
    installation, compile flags, build locations, and more)
  * Quote *all* paths and variable expansions unless list expansion is required
    (usually in command arguments or optional arguments)

C++
---

  * 2-space indentation
  * Use lowercase with underscores for symbol names
  * Store intermediate values into local ``const`` variables so that they are
    easily available when debugging
  * There is no fixed line length, but keep it reasonable
  * Default to using ``const`` everywhere
  * All modifiers of a type go *after* the type (e.g., ``char const*``, not
    ``const char*``)
  * Export symbols (or import them if possible)
  * Use braces around all control (even single-line if) blocks
  * Use typedefs
  * Use exceptions and return values, not error codes and output parameters
    - This allows for chaining functions, works with ``<algorithm>`` better,
      and allows more variables to be ``const``

Python
------

  * Follow PEP8
  * When catching exceptions, catch the type then use ``sys.exc_info()`` so
    that it works in Python versions from 2.4 to 3.3
  * No metaclasses; they don't work with the same syntax in Python2 and Python3
  * Avoid 'with' since it doesn't work in Python 2.4

Testing
-------

Generally, all new code should come with tests. The goal is sustained
95% coverage and higher (due to impossible-to-generically-create
corner cases such as files which are readable, but error out in the
middle). Tests should be grouped into a single executable for each
class, group of cooperating classes (e.g., types tests), or
higher-level use case. In C++, use the ``TEST_`` macros which will
hook into the testing infrastructure automatically and in Python, name
functions so that they start with ``test_`` and they will be picked up
automatically.
