.. VIAME documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VIAME
=====

VIAME is a computer vision library designed to integrate several image and video processing algorithms together
in a common distributed processing framework, majorly targeting marine species analytics. 
As it contains many common algorithms and compiles several other popular repositories together as a part of its build process,
VIAME is also useful as a general computer vision toolkit. 
Thecore infrastructure connecting different system components is currently the KWIVER library, 
which can connect C/C++, python, and matlab nodes together in a graph-like pipeline architecture. 
Alongside the pipelined image processing system are a number of standalone utilties for model training, 
output detection visualization, groundtruth annotation, detector/tracker evaluation (a.k.a. scoring), image/video search, and rapid model generation. 

Contents:

.. toctree::
   :maxdepth: 1

   example_links/building_viame
   example_links/common_data
   example_links/detection_file_conversions
   example_links/detector_pipelines
   example_links/detector_training
   example_links/detector_training_old_api
   example_links/external_plugin_creation
   example_links/hello_world_pipeline
   example_links/image_and_video_search
   example_links/scoring_and_roc_generation
   example_links/tracking_pipeline
   example_links/using_detectors_in_gui
   example_links/visualizing_detections_in_gui



.. |br| raw:: html

   <br />
