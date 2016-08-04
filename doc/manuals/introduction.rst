Introduction
============

The Kitware Image and Video Exploitation and Retrieval (KWIVER)
toolkit is a collection of software tools designed to tackle
difficult image and video analysis problems and other related
challenges.  KWIVER is an ongoing effort to
transition technology developed over multiple years by Kitware's
computer vision group to the open
source domain in order to further research, collaboration, and product
development.

KWIVER contains the following components.

`VITAL`_
  A core library of abstractions and data types used by various KWIVER components.
  Major elements of VITAL are:
  - Basic data types used throughout Kwiver.
  - Provides abstract algorithm interfaces for implementations in the ARROWS component.
  - Configuration support library providing a common approach to run time configuration of the components.
  - An OS abstraction layer that provides system services in a platform independent manner.
  - flexible logging support that can interface to different logging back ends.
  - General purpose plugin architecture.

`Stream Processing Toolkit (sprokit)`_
  Sprokit is the “Stream Processing Toolkit”, a library aiming to
  make processing a stream of data with various algorithms easy.
  It supports divergent and convergent data flows with synchronization
  between them, connection type checking, all with full, first-class
  Python bindings.

  Sprokit also contains a set of processes and example pipelines that
  support basic operations such as image and video input and display,
  wrappers for common algorithms.

`ARROWS`_
  ARROWS is an open source C++ collection of algorithms
  for making measurements from aerial video. Initial capability
  focuses on estimating the camera flight trajectory and a sparse
  3D point cloud of the scene.

Additionally, a separate repository, Fletch, is a CMake based project
that assists with acquiring and building common Open Source libraries
useful for developing video exploitation tools.

There is no single "correct"
way to build KWIVER.  Rather, depending on your use case you will configure and build KWIVER
in ways that make the tools and libraries you require avaialable to you.  In this documentation
we'll detail and document some of the more common and useful usecases.

.. toctree::
   :maxdepth: 3

   videoisr
   smqtkbridge
