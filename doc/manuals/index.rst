.. VIAME documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/into_image.png
   :scale: 60 %
   :align: center

|

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

Contents:

.. toctree::
   :maxdepth: 1

   example_links/building_viame
   example_links/example_capabilities
   Core C++/Python Object Types <http://kwiver.readthedocs.io/en/latest/vital/architecture.html>
   Core Pipelining Architecture <http://kwiver.readthedocs.io/en/latest/sprokit/architecture.html>
   Basic Processing Nodes <http://kwiver.readthedocs.io/en/latest/arrows/architecture.html>
   example_links/hello_world_pipeline
   example_links/detector_pipelines
   example_links/tracking_pipeline
   example_links/detection_file_conversions
   example_links/measurement_using_stereo
   example_links/detector_training
   example_links/image_and_video_search
   example_links/using_detectors_in_cxx_code
   example_links/visualizing_detections_in_gui
   example_links/scoring_and_roc_generation
   example_links/external_plugin_creation
   KWIVER Manual <http://kwiver.readthedocs.io/en/latest/>


.. |br| raw:: html

   <br />
