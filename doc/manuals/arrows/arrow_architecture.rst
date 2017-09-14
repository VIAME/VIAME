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
arrows, like `core <arrows/core>`_, require no additional dependencies.
Some examples of the provided Arrows are:

* `ocv <arrows/ocv>`__ - provides algorithms from OpenCV_
* `ceres <arrows/ceres>`__ - provides algorithms from `Ceres Solver`_
* `vxl <arrow/vxl>`__ - provides algorithms from VXL_

.. toctree::
   :maxdepth: 3


.. _`Ceres Solver`: http://ceres-solver.org/
.. _OpenCV: http://opencv.org/
.. _VXL: https://github.com/vxl/vxl/