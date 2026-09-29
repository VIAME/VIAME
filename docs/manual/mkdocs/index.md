# VIAME

Video and Image Analytics for Multiple Environments ([VIAME](https://www.viametoolkit.org)) is a computer vision application designed for do-it-yourself artificial intelligence including object detection, object tracking, image/video annotation, image/video search, image mosaicing, image enhancement, size measurement, multi-camera data processing, rapid model generation, and tools for the evaluation of different algorithms. Originally targeting marine species analytics, VIAME now contains many common algorithms and libraries, and is also useful as a generic computer vision toolkit. It contains a number of standalone tools for accomplishing the above, a pipeline framework which can connect C/C++, python, and matlab nodes together in a multi-threaded fashion, and multiple algorithms resting on top of the pipeline infrastructure. Lastly, a portion of the algorithms have been integrated into both desktop and web user interfaces for deployments in different types of environments, with an open annotation archive and example of the web platform available at [viame.kitware.com](https://viame.kitware.com).

## Documentation Overview

This manual is synced to the VIAME ["main"](https://github.com/VIAME/VIAME) branch and is updated frequently, though you may have to press ctrl-F5 to see the latest updates to avoid using your browser cache of this webpage if you have used it previously. In addition to this manual, there are 4 useful types of documentation:

1.  A [quick-start guide](https://viame.readthedocs.io/en/latest/sections/quick_start_guide.html) meant for first time users using the desktop version
2.  An [overview presentation](https://www.viametoolkit.org/wp-content/uploads/2020/09/VIAME-AI-Workshop-Aug2020.pdf) covering the basic design of VIAME
3.  The [VIAME Web and DIVE Desktop docs](https://kitware.github.io/dive) and in-GUI help menu
4.  Our [YouTube video channel](https://www.youtube.com/channel/UCpfxPoR5cNyQFLmqlrxyKJw) (work in progress)

## Contents

- [Documentation Overview](index.md)
- [Quick-Start Guide](sections/quick_start_guide.md)
- [Model Generation Workflows](sections/model_generation_workflows.md)
- [Installing VIAME from Binaries](sections/installing_from_binaries.md)
- [Building VIAME From Source](sections/building_from_source.md)
- [User Interfaces](sections/annotation_and_visualization.md)
- [DIVE Interface](sections/dive/index.md)
- [Command Line Interface](sections/command_line_interface.md)
- [Python Interface](sections/python_interface.md)
- [Project Folders](sections/project_folders.md)
- [Scripts and Example Folders](sections/examples_overview.md)
- [Detection File Formats and Conversions](sections/detection_file_conversions.md)
- [Object Detection](sections/object_detection.md)
- [Detector Training](sections/object_detector_training.md)
- [Size Measurement](sections/size_measurement.md)
- [Object Tracking](sections/object_tracking.md)
- [Image Enhancement and Filtering](sections/image_enhancement.md)
- [Video and Image Search](sections/search_and_rapid_model_generation.md)
- [Rapid Model Generation](sections/rapid_model_generation.md)
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

## Example Capabilities

There are a number of core capabilities within the software, click on each of the below images to learn more.

Object Detection and Tracking

<a href="https://github.com/VIAME/VIAME/tree/main/examples/object_detection"><img src="_static/images/many_scallop_detections_gui.jpg" alt="Many scallop detections gui" width="30%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/object_detection"><img src="_static/images/Capabilities_Object_Detection.jpg" alt="Capabilities object detection" width="27.5%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/object_tracking"><img src="_static/images/Text-Query-Result1.jpg" alt="Text query result" width="27.5%"></a>

User Interfaces for Annotation, Visualization, and Detector Model Training

<a href="https://github.com/VIAME/VIAME/tree/main/examples/annotation_and_visualization"><img src="_static/images/dive_banner.jpg" alt="DIVE annotator" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/annotation_and_visualization"><img src="_static/images/Point-Segmentation.jpg" alt="Point segmentation" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/annotation_and_visualization"><img src="_static/images/Train_From_Dive.png" alt="Train from dive" width="29%"></a>

Measuring Animal Lengths Using Metadata or Stereo

<a href="https://github.com/VIAME/VIAME/tree/main/examples/size_measurement"><img src="_static/images/fish_measurement_example.jpg" alt="Fish measurement example" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/size_measurement"><img src="_static/images/Calibration-Show-Features-On-Success1.jpg" alt="Calibration show features on success" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/size_measurement"><img src="_static/images/Stereo-Seamap-Short1.jpg" alt="Stereo seamap short" width="29%"></a>

Text, Image, Video Search for Rapid Model Generation

<a href="https://github.com/VIAME/VIAME/tree/main/examples/video_and_image_search"><img src="_static/images/iqr_11_initial_results.jpg" alt="Iqr 11 initial results" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/video_and_image_search"><img src="_static/images/Perform-Text-Query.jpg" alt="Perform text query" width="29%"></a>

Illumination Normalization and Color Correction

<a href="https://github.com/VIAME/VIAME/tree/main/examples/image_enhancement"><img src="_static/images/color_correct.jpg" alt="Color correct" width="29%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/image_enhancement"><img src="_static/images/Image_Filter_in_DIVE.jpg" alt="Image filter in dive" width="29%"></a>

Detector and Tracker Evaluation

<a href="https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation"><img src="_static/images/Score_PRC.png" alt="Score prc" width="20%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation"><img src="_static/images/Score_Confusion_Matrix.jpg" alt="Score confusion matrix" width="17.5%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation"><img src="_static/images/Score_ROC.png" alt="Score roc" width="20%"></a>
<a href="https://github.com/VIAME/VIAME/tree/main/examples/scoring_and_evaluation"><img src="_static/images/Score_MAP_Table.png" alt="Score map table" width="20%"></a>
