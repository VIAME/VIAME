Vital Architecture
==================

Vital is the core of KWIVER and is designed to provide data and algorithm
abstractions with minimal library dependencies. Vital only depends on
the C++ standard library and the header-only Eigen_ library.  Vital defines
the core data types and abstract interfaces for core vision algorithms
using these types.  Vital also provides various system utility functions
like logging, plugin management, and configuration file handling.  Vital
does **not** provide implementations of the abstract algorithms.
Implementations are found in Arrows and are loaded dynamically, by vital,
at run-time via plugins. 

The design of KWIVER allows end-user applications to link only against
the Vital libraries and have minimal hard dependencies.  
One can then dynamically add algorithmic capabilities, with new dependencies, via
plugins without needing to recompile Vital or the application code.
Only Vital is built by default when building KWIVER without enabling
any options in CMake. You will need to enable various Arrows in order 
for vital to instantiate those various implementations.

The Vital API is all that applications need to control the execute any KWIVER algorithm arrow. 
In the following sections we will breakdown the various the algorithms and data types provided in vital based on their functionality.

.. toctree::
   :maxdepth: 2

   images
   detections
   doxygen
   

.. _Eigen: http://eigen.tuxfamily.org/
