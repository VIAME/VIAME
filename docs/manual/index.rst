.. VIAME documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VIAME
=====

VIAME is a computer vision application designed for do-it-yourself artificial intelligence
including object detection, object tracking, image mosaicing, stereo measurement,
image/video search, image/video annotation, rapid model generation, and tools for the
evaluation of different algorithms. Originally targeting marine species analytics, it now
contains many common algorithms and libraries, and is also useful as a generic computer vision
library. The core infrastructure connecting different system components is currently the
KWIVER library, which can connect C/C++, python, and matlab nodes together in a graph-like
pipeline architecture. Alongside the pipelined image processing system are a number of
standalone tools for accomplishing the above. Both a desktop and web version exists for
deployments in different types of environments.

Documentation Overview
======================

This manual is synced to the VIAME 'main' branch and is updated frequently, you may
have to press ctrl-F5 to see the latest updates to avoid using your browser cache of
this webpage.

In addition to this manual, there are 4 other types of documentation in VIAME:

1) A `quick-start guide`_ meant for first time users using the desktop version
2) An `overview presentation`_ covering the basic design of VIAME
3) The `VIAME Web and DIVE Desktop docs`_ and in-GUI help menu
4) Our `YouTube video channel`_ (work in progress)

.. _quick-start guide: https://data.kitware.com/api/v1/item/5fdaf1dd2fa25629b99843f8/download
.. _overview presentation: https://www.viametoolkit.org/wp-content/uploads/2020/09/VIAME-AI-Workshop-Aug2020.pdf
.. _VIAME Web and DIVE Desktop docs: https://kitware.github.io/dive
.. _YouTube video channel: https://www.youtube.com/channel/UCpfxPoR5cNyQFLmqlrxyKJw

Contents
========

.. toctree::
   :maxdepth: 1

   Documentation Overview <https://viame.readthedocs.io/en/latest/index.html>
   sections/installing_from_binaries
   sections/building_from_source
   sections/annotation_and_visualization
   sections/examples_overview
   sections/detection_file_conversions
   sections/object_detection
   sections/object_detector_training
   sections/size_measurement
   sections/object_tracking
   sections/image_enhancement
   sections/search_and_rapid_model_generation
   sections/scoring_and_evaluation
   sections/registration_and_mosaicing
   sections/frame_level_classification
   sections/archive_summarization
   Core C++/Python Object Types <http://kwiver.readthedocs.io/en/latest/vital/architecture.html>
   Core Pipelining Architecture <http://kwiver.readthedocs.io/en/latest/sprokit/architecture.html>
   Basic Pipeline Nodes <http://kwiver.readthedocs.io/en/latest/arrows/architecture.html>
   sections/hello_world_pipeline
   sections/plugin_creation
   sections/using_algorithms_in_code
   KWIVER Full Manual <http://kwiver.readthedocs.io/en/latest/>

Example Capabilities
====================

There's a number of core capapbilities within VIAME, click on each of the below images to learn more.

Object Detection
----------------

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/many_scallop_detections_gui.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/object_detection

Measuring Fish Lengths Using Metadata or Stereo
-----------------------------------------------

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/fish_measurement_example.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/size_measurement

Image and Video Search for Rapid Model Generation
-------------------------------------------------

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/01/search_ex.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/search_and_rapid_model_generation

GUIs for Visualization and MLOps
--------------------------------

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/annotation_example.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/annotation_and_visualization

Illumination Normalization and Color Correction
-----------------------------------------------

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/09/color_correct.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/image_enhancement

Detector and Tracker Evaluation
-------------------------------

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/scoring-2.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/scoring_and_evaluation

.. |br| raw:: html

   <br />
