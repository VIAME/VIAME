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

Unlike other open source projects, KWIVER_ is not organized as a
single repository containing all of the code for the project. Rather,
the KWIVER repository located on GitHub serves as a central location
for KWIVER’s documentation and contains sample code, utilities,
tests and other tools that are useful to the KWIVER community. The
bulk of the repository, however, consists of a CMake superbuild
that builds many of the members of the KWIVER “family” of projects.
Each of these members typically has its own source code repository
and in some cases, as with MAP-Tk, can be built and dealt with as
a single stand alone project.  

`Motion-imagery Aerial Photogrammetry Toolkit (MAP-Tk)`_
  MAP-Tk is an open source C++ collection of libraries and tools
  for making measurements from aerial video. Initial capability
  focuses on estimating the camera flight trajectory and a sparse
  3D point cloud of the scene.

`Stream Processing Toolkit (sprokit)`_
  Sprokit is the “Stream Processing Toolkit”, a library aiming to
  make processing a stream of data with various algorithms easy.
  It supports divergent and convergent data flows with synchronization
  between them, connection type checking, all with full, first-class
  Python bindings.

`Social Multimedia Query Toolkit (SMQTK)`_
  A collection of Python tools, with C++ dependencies, for ingesting
  images and video from social media (e.g. YouTube, Twitter),
  computing content-based features, indexing the media based on the
  content descriptors, querying for similar content, and building
  user-defined searches via an interactive query refinement (IQR)
  process.

`ViViA`_
  A collection of Qt based applications for GUIs, visualization and
  exploration of content extracted from video.

`Video and Image-Based Retrieval and Analysis Toolkit (VIBRANT)`_
  An end-to-end system for surveillance video analytics including
  content-based retrieval and alerting using behaviors, actions and
  appearance.

`KWant`_
  A lightweight toolkit for computing detection and tracking metrics
  on a variety of video data. It computes spatial and temporal
  associations between datasets, even with different frame rates.
  It has a flexible input format and can generate XML based results.

`VITAL`_ 
  A core library of abstractions and data types used by various KWIVER components.

.. _KWIVER: https://github.com/Kitware/kwiver
.. _Motion-imagery Aerial Photogrammetry Toolkit (MAP-Tk): https://github.com/Kitware/maptk
.. _Stream Processing Toolkit (sprokit): https://github.com/Kitware/sprokit
.. _ViViA: https://github.com/Kitware/vivia
.. _Video and Image-Based Retrieval and Analysis Toolkit (VIBRANT): https://github.com/Kitware/vibrant
.. _KWant: https://github.com/Kitware/kwant
.. _Social Multimedia Query Toolkit (SMQTK): https://github.com/Kitware/SMQTK
.. _VITAL: https://github.com/Kitware/vital

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
