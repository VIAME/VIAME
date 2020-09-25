.. VIAME documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Summary
=======

VIAME is a computer vision application designed for do-it-yourself artificial intelligence
including object detection, object tracking, image mosaicing, stereo measurement,
image/video search, image/video annotation, rapid model generation, and tools for the
evaluation of different algorithms. Originally targetting marine species analytics, it now
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
   section_links/building_and_installing_viame
   section_links/example_capabilities
   section_links/image_enhancement
   section_links/object_detection
   section_links/object_tracking
   section_links/detection_file_conversions
   section_links/measurement_using_stereo
   section_links/object_detector_training
   section_links/search_and_rapid_model_generation
   section_links/annotation_and_visualization
   section_links/scoring_and_roc_generation
   section_links/archive_summarization
   section_links/image_registration
   section_links/frame_level_classification
   Core C++/Python Object Types <http://kwiver.readthedocs.io/en/latest/vital/architecture.html>
   Core Pipelining Architecture <http://kwiver.readthedocs.io/en/latest/sprokit/architecture.html>
   Basic Pipeline Nodes <http://kwiver.readthedocs.io/en/latest/arrows/architecture.html>
   section_links/hello_world_pipeline
   section_links/external_plugin_creation
   section_links/using_detectors_in_cxx_code
   KWIVER Full Manual <http://kwiver.readthedocs.io/en/latest/>


.. |br| raw:: html

   <br />
