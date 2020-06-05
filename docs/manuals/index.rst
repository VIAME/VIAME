.. VIAME documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://www.viametoolkit.org/wp-content/uploads/2020/02/viame_summary.png
   :scale: 30 %
   :align: center
   :target: http://viame.readthedocs.io/en/latest/index.html

|
Summary
=======

VIAME is a computer vision library designed to integrate several image and video
processing algorithms together in a common distributed processing framework, majorly
targeting marine species analytics. As it contains many common algorithms and compiles
several other popular repositories together as a part of its build process,
VIAME is also useful as a general computer vision toolkit. The core infrastructure
connecting different system components is currently the KWIVER library, which can
connect C/C++, python, and matlab nodes together in a graph-like pipeline architecture. 
Alongside the pipelined image processing system are a number of standalone utilties
for model training, output detection visualization, groundtruth annotation,
detector/tracker evaluation (a.k.a. scoring), image/video search, and rapid model
generation. 

This manual is synced to the VIAME 'master' branch and is updated frequently, you may
have to press ctrl-F5 to see the latest updates to avoid using your browser cache of
this webpage.


*********
Contents:
*********

.. toctree::
   :maxdepth: 1

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
   Core C++/Python Object Types <http://kwiver.readthedocs.io/en/latest/vital/architecture.html>
   Core Pipelining Architecture <http://kwiver.readthedocs.io/en/latest/sprokit/architecture.html>
   Basic Pipeline Nodes <http://kwiver.readthedocs.io/en/latest/arrows/architecture.html>
   section_links/hello_world_pipeline
   section_links/external_plugin_creation
   section_links/using_detectors_in_cxx_code
   KWIVER Full Manual <http://kwiver.readthedocs.io/en/latest/>


.. |br| raw:: html

   <br />
