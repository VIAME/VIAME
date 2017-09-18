Arrow Architecture
===================

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
arrows, like core , require no additional dependencies.
The provided Arrows are:

.. toctree::
   :maxdepth: 3

   core
   burnout
   ceres
   darknet
   matlab
   opencv
   proj
   uuid
   viscl
   vxl
   