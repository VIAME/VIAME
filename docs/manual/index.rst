.. VIAME documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VIAME
=====

Video and Image Analytics for Multiple Environments (`VIAME`_) is a computer vision application
designed for do-it-yourself artificial intelligence including object detection, object tracking,
image/video annotation, image/video search, image mosaicing, image enhancement, size measurement,
multi-camera data processing, rapid model generation, and tools for the evaluation of different
algorithms. Originally targeting marine species analytics, VIAME now contains many common
algorithms and libraries, and is also useful as a generic computer vision toolkit.
It contains a number of standalone tools for accomplishing the above, a pipeline framework
which can connect C/C++, python, and matlab nodes together in a multi-threaded fashion, and
multiple algorithms resting on top of the pipeline infrastructure. Lastly, a portion of the
algorithms have been integrated into both desktop and web user interfaces for deployments in
different types of environments, with an open annotation archive and example of the web
platform available at `viame.kitware.com`_.

.. _VIAME: https://www.viametoolkit.org
.. _viame.kitware.com: https://viame.kitware.com

Documentation Overview
======================

This manual is synced to the VIAME `"main"`_ branch and is updated frequently, though you
may have to press ctrl-F5 to see the latest updates to avoid using your browser cache of
this webpage if you have used it previously. In addition to this manual, there are 4 useful
types of documentation:

.. _"main": https://github.com/VIAME/VIAME

1) A `quick-start guide`_ meant for first time users using the desktop version
2) An `overview presentation`_ covering the basic design of VIAME
3) The `VIAME Web and DIVE Desktop docs`_ and in-GUI help menu
4) Our `YouTube video channel`_ (work in progress)

.. _quick-start guide: https://viame.readthedocs.io/en/latest/sections/quick_start_guide.html
.. _overview presentation: https://www.viametoolkit.org/wp-content/uploads/2020/09/VIAME-AI-Workshop-Aug2020.pdf
.. _VIAME Web and DIVE Desktop docs: https://kitware.github.io/dive
.. _YouTube video channel: https://www.youtube.com/channel/UCpfxPoR5cNyQFLmqlrxyKJw

Contents
========

.. toctree::
   :maxdepth: 1

   Documentation Overview <https://viame.readthedocs.io/en/latest/index.html>
   sections/quick_start_guide
   sections/model_generation_workflows
   sections/installing_from_binaries
   sections/building_from_source
   sections/annotation_and_visualization
   DIVE Interface <sections/dive/index>
   sections/command_line_interface
   Python Interface <sections/python_interface>
   sections/project_folders
   sections/examples_overview
   sections/detection_file_conversions
   sections/object_detection
   sections/object_detector_training
   sections/stereo_measurement
   sections/monocular_measurement
   sections/object_tracking
   sections/image_enhancement
   sections/search_and_rapid_model_generation
   sections/text_query_and_vlm
   sections/scoring_and_evaluation
   sections/registration_and_mosaicing
   sections/frame_level_classification
   sections/scene_segmentation
   sections/archive_summarization
   Core C++/Python Object Types <https://kwiver.readthedocs.io/en/latest/vital/architecture.html>
   Core Pipelining Architecture <https://kwiver.readthedocs.io/en/latest/sprokit/architecture.html>
   Basic Pipeline Nodes <https://kwiver.readthedocs.io/en/latest/arrows/architecture.html>
   sections/example_pipeline
   sections/plugin_creation
   sections/using_algorithms_in_code
   KWIVER Full Manual <https://kwiver.readthedocs.io/en/latest/>

Example Capabilities
====================

There are a number of core capabilities within the software, click on each of the below images to learn more.

Object Detection and Tracking

.. image:: _static/images/many_scallop_detections_gui.jpg
   :alt: Many scallop detections gui
   :width: 30%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/object_detection

.. image:: _static/images/Capabilities_Object_Detection.jpg
   :alt: Capabilities object detection
   :width: 27.5%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/object_detection

.. image:: _static/images/Text-Query-Result1.jpg
   :alt: Text query result
   :width: 27.5%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/object_tracking

User Interfaces for Annotation, Visualization, and Detector Model Training

.. image:: _static/images/dive_banner.jpg
   :alt: DIVE annotator
   :width: 29%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/annotation_and_visualization

.. image:: _static/images/Point-Segmentation.jpg
   :alt: Point segmentation
   :width: 29%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/annotation_and_visualization

.. image:: _static/images/Train_From_Dive.png
   :alt: Train from dive
   :width: 29%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/annotation_and_visualization

Measuring Animal Lengths Using Metadata or Stereo

.. image:: _static/images/fish_measurement_example.jpg
   :alt: Fish measurement example
   :width: 29%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/stereo_measurement

.. image:: _static/images/Calibration-Show-Features-On-Success1.jpg
   :alt: Calibration show features on success
   :width: 29%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/stereo_measurement

.. image:: _static/images/Stereo-Seamap-Short1.jpg
   :alt: Stereo seamap short
   :width: 29%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/stereo_measurement

Text, Image, Video Search for Rapid Model Generation

.. image:: _static/images/iqr_11_initial_results.jpg
   :alt: Iqr 11 initial results
   :width: 29%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/video_and_image_search

.. image:: _static/images/Perform-Text-Query.jpg
   :alt: Perform text query
   :width: 29%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/video_and_image_search

Illumination Normalization and Color Correction

.. image:: _static/images/color_correct.jpg
   :alt: Color correct
   :width: 29%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/image_enhancement

.. image:: _static/images/Image_Filter_in_DIVE.jpg
   :alt: Image filter in dive
   :width: 29%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/image_enhancement

Detector and Tracker Evaluation

.. image:: _static/images/Score_PRC.png
   :alt: Score prc
   :width: 20%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation

.. image:: _static/images/Score_Confusion_Matrix.jpg
   :alt: Score confusion matrix
   :width: 17.5%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation

.. image:: _static/images/Score_ROC.png
   :alt: Score roc
   :width: 20%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation

.. image:: _static/images/Score_MAP_Table.png
   :alt: Score map table
   :width: 20%
   :target: https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation

.. |br| raw:: html

   <br />
