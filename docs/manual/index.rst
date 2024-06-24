.. VIAME documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Summary
=======

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

This manual is synced to the VIAME 'master' branch and is updated frequently, you may
have to press ctrl-F5 to see the latest updates to avoid using your browser cache of
this webpage.


*********
Contents:
*********

.. toctree::
   :maxdepth: 1

   section_links/documentation_overview
   section_links/installing_from_binaries
   section_links/building_from_source
   section_links/annotation_and_visualization
   section_links/examples_overview
   section_links/detection_file_conversions
   section_links/object_detection
   section_links/object_detector_training
   section_links/size_measurement
   section_links/object_tracking
   section_links/image_enhancement
   section_links/search_and_rapid_model_generation
   section_links/scoring_and_evaluation
   section_links/registration_and_mosaicing
   section_links/frame_level_classification
   section_links/archive_summarization
   Core C++/Python Object Types <http://kwiver.readthedocs.io/en/latest/vital/architecture.html>
   Core Pipelining Architecture <http://kwiver.readthedocs.io/en/latest/sprokit/architecture.html>
   Basic Pipeline Nodes <http://kwiver.readthedocs.io/en/latest/arrows/architecture.html>
   section_links/hello_world_pipeline
   section_links/plugin_creation
   section_links/using_algorithms_in_code
   KWIVER Full Manual <http://kwiver.readthedocs.io/en/latest/>

Key Toolkit Capabilities
========================

Object Detection
****************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/many_scallop_detections_gui.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/object_detection

Measuring Fish Lengths Using Stereo
***********************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/fish_measurement_example.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/size_measurement

Image and Video Search for Rapid Model Generation
*************************************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/01/search_ex.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/search_and_rapid_model_generation

GUIs for Visualization and Annotation
*************************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/annotation_example.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/annotation_and_visualization

Illumination Normalization and Color Correction
***********************************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/09/color_correct.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/image_enhancement

Detector and Tracker Evaluation
*******************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/scoring-2.png
   :scale: 50
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/scoring_and_evaluation

.. |br| raw:: html

   <br />
