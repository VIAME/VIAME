# VIAME { .hidden-title }

Video and Image Analytics for Multiple Environments ([VIAME](https://www.viametoolkit.org)) is a computer vision application designed for do-it-yourself artificial intelligence including object detection, object tracking, data annotation, multi-camera processing, size measurement, image enhancement, rapid model generation, query-based search, mosaicing, and tools for the evaluation of algorithms. Originally targeting marine species analytics, VIAME now contains many algorithms and libraries, and is useful as a generic computer vision toolkit. It contains a number of tools for accomplishing the above, a pipeline framework which can connect C++/Python nodes in a multi-threaded fashion, and multiple algorithms resting on top of the pipeline infrastructure. Lastly, a portion of the algorithms have been integrated into both desktop and web user interfaces for deployments in different environments, with an open annotation archive and example of the web platform available at [viame.kitware.com](https://viame.kitware.com).

## Documentation Overview

This manual is synced to the VIAME ["main"](https://github.com/VIAME/VIAME) branch and is updated frequently, though you may have to press ctrl-F5 to see the latest updates to avoid using your browser cache of this webpage if you have used it previously. In addition to this manual, there are 4 useful types of documentation:

1.  A [quick-start guide](https://viame.github.io/VIAME/sections/quick_start_guide.html) meant for first time users using the desktop version
2.  An [overview presentation](https://www.viametoolkit.org/wp-content/uploads/2020/09/VIAME-AI-Workshop-Aug2020.pdf) covering the basic design of VIAME
3.  The [VIAME Web and DIVE Desktop docs](https://kitware.github.io/dive) and in-GUI help menu
4.  Our [YouTube video channel](https://www.youtube.com/channel/UCpfxPoR5cNyQFLmqlrxyKJw) (work in progress)

## Example Capabilities

There are a number of core capabilities within the software, click on each of the below images to learn more.

Object Detection and Tracking

<p align="center">
<a href="https://github.com/VIAME/VIAME/tree/main/examples/object_tracking"><img src="_static/images/Text-Query-Result1.jpg" alt="Text query result" width="27.5%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/object_detection"><img src="_static/images/Capabilities_Object_Detection.jpg" alt="Capabilities object detection" width="27.5%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/object_detection"><img src="_static/images/many_scallop_detections_gui.jpg" alt="Many scallop detections gui" width="30%"></a>
</p>

User Interfaces for Annotation, Visualization, and Detector Model Training

<p align="center">
<a href="https://github.com/VIAME/VIAME/tree/main/examples/user_interfaces"><img src="_static/images/dive_banner.jpg" alt="DIVE annotator" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/user_interfaces"><img src="_static/images/Point-Segmentation.jpg" alt="Point segmentation" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/user_interfaces"><img src="_static/images/Train_From_Dive.png" alt="Train from dive" width="29%"></a>
</p>

Measuring Animal Lengths Using Metadata or Stereo

<p align="center">
<a href="https://github.com/VIAME/VIAME/tree/main/examples/stereo_measurement"><img src="_static/images/Calibration-Show-Features-On-Success1.jpg" alt="Calibration show features on success" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/stereo_measurement"><img src="_static/images/Stereo-Seamap-Short1.jpg" alt="Stereo seamap short" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/stereo_measurement"><img src="_static/images/fish_measurement_example.jpg" alt="Fish measurement example" width="29%"></a>
</p>

Text, Image, Video Search for Rapid Model Generation

<p align="center">
<a href="https://github.com/VIAME/VIAME/tree/main/examples/video_and_image_search"><img src="_static/images/Perform-Text-Query.jpg" alt="Perform text query" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/video_and_image_search"><img src="_static/images/iqr_11_initial_results.jpg" alt="Video search initial results" width="29%"></a>
</p>

Illumination Normalization and Color Correction

<p align="center">
<a href="https://github.com/VIAME/VIAME/tree/main/examples/image_enhancement"><img src="_static/images/color_correct.jpg" alt="Color correct" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/image_enhancement"><img src="_static/images/Image_Filter_in_DIVE.jpg" alt="Image filter in dive" width="29%"></a>
</p>

Detector and Tracker Evaluation

<p align="center">
<a href="https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation"><img src="_static/images/Score_PRC.png" alt="Score prc" width="20%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation"><img src="_static/images/Score_Confusion_Matrix.jpg" alt="Score confusion matrix" width="17.5%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation"><img src="_static/images/Score_ROC.png" alt="Score roc" width="20%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation"><img src="_static/images/Score_MAP_Table.png" alt="Score map table" width="20%"></a>
</p>

## Contents

- [Documentation Overview](index.md)
- [Quick-Start Guide](sections/quick_start_guide.md)
- [Installing VIAME from Binaries](sections/installing_from_binaries.md)
- [Building VIAME From Source](sections/building_from_source.md)
- [Model Generation Workflows](sections/model_generation_workflows.md)
- [Model Zoo and Add-Ons](https://github.com/VIAME/VIAME/wiki/Model-Zoo-and-Add-Ons)
- [User Interfaces](sections/user_interfaces.md)
- [DIVE Interface](sections/dive/index.md)
- [Interactive Annotation](sections/interactive_annotation.md)
- [Python Interface](sections/python_interface.md)
- [Project Folders](sections/project_folders.md)
- [Scripts and Example Folders](sections/examples_overview.md)
- [Command Line Interface](sections/command_line_interface.md)
- [Detection File Formats](sections/detection_file_formats.md)
- [Detection File Conversions](sections/detection_file_conversions.md)
- [Object Detection](sections/object_detection.md)
- [Detector Training](sections/object_detector_training.md)
- [Config File Guide](https://github.com/VIAME/VIAME/wiki/Config-Layout-and-Frameworks)
- [Stereo Measurement](sections/stereo_measurement.md)
- [Monocular Measurement](sections/monocular_measurement.md)
- [Object Tracking](sections/object_tracking.md)
- [Image Enhancement and Filtering](sections/image_enhancement.md)
- [Video and Image Search](sections/search_and_rapid_model_generation.md)
- [Text Query and VLM](sections/text_query_and_vlm.md)
- [Scoring Detectors and Trackers](sections/scoring_and_evaluation.md)
- [Registration and Mosaicing](sections/registration_and_mosaicing.md)
- [Frame Level Classification](sections/frame_level_classification.md)
- [Scene Segmentation](sections/scene_segmentation.md)
- [Archive Summarization](sections/archive_summarization.md)
- [Core C++/Python Object Types](https://kwiver.readthedocs.io/en/latest/vital/architecture.html)
- [Core Pipelining Architecture](https://kwiver.readthedocs.io/en/latest/sprokit/architecture.html)
- [Basic Pipeline Nodes](https://kwiver.readthedocs.io/en/latest/arrows/architecture.html)
- [New Module Creation](sections/example_pipeline.md)
- [Plugin Creation](sections/plugin_creation.md)
- [Using Algorithms in Code](sections/using_algorithms_in_code.md)
- [KWIVER Full Manual](https://kwiver.readthedocs.io/en/latest/)
- [VIEW Interface](sections/view_interface.md)
