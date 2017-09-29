Overview of Components
======================

KWIVER contains the following components.

vital
-----
  A core library of abstractions and data types used by various KWIVER components.
  Major elements of VITAL are:

  - Basic data types used throughout Kwiver.
  - Provides abstract algorithm interfaces for implementations in the ARROWS component.
  - Configuration support library providing a common approach to run time configuration of the components.
  - An OS abstraction layer that provides system services in a platform independent manner.
  - flexible logging support that can interface to different logging back ends.
  - General purpose plugin architecture.

Vital Configurartion Support
''''''''''''''''''''''''''''

.. toctree::
   :maxdepth: 3

   vital/config_usage.rst
   vital/config_file_format.rst


Stream Processing Toolkit (sprokit)
-----------------------------------
  Sprokit is the “Stream Processing Toolkit”, a library aiming to
  make processing a stream of data with various algorithms easy.
  It supports divergent and convergent data flows with synchronization
  between them, connection type checking, all with full, first-class
  Python bindings.

  Sprokit also contains a set of processes and example pipelines that
  support basic operations such as image and video input and display,
  wrappers for common algorithms.

  .. toctree::
   :maxdepth: 3

   sprokit/pipeline_design.rst


ARROWS
------
  ARROWS is an open source C++ collection of algorithms
  for making measurements from aerial video. Initial capability
  focuses on estimating the camera flight trajectory and a sparse
  3D point cloud of the scene.
