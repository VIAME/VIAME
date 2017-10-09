Introduction
============

The KWIVER toolkit is a collection of software tools designed to
tackle challenging image and video analysis problems and other related
challenges. Recently started by Kitware’s Computer Vision and
Scientific Visualization teams, KWIVER is an ongoing effort to
transition technology developed over multiple years to the open source
domain to further research, collaboration, and product development.
KWIVER is a collection of C++ libraries with C and Python bindings
and uses an permissive `BSD License <LICENSE>`_.

Visit the `repository <https://github.com/Kitware/kwiver>`_ on how to get and build the KWIVER code base

One of the primary design goals of KWIVER is to make it easier to pull
together algorithms from a wide variety of third-party, open source
image and video processing projects and integrate them into highly
modular, run-time configurable systems. 

This goal is achieved through the three main components of KWIVER: Vital, Arrows, and Sprokit.


.. toctree::
  :hidden:
  
  vital/architecture
  arrows/architecture
  sprokit/architecture

====================================== ====================================================
:doc:`Vital</vital/architecture>`        A set of data types and algorithm interfaces      
:doc:`Arrows</arrows/architecture>`      Various implementations of vital algorithms       
:doc:`Sprokit</sprokit/architecture>`    An infrastructure for chaining together algorithms
====================================== ====================================================
