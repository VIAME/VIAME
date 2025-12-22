
=====================
Archive Summarization
=====================

This document corresponds to the `Archive Summarization`_ example folder within a VIAME
desktop installation.

.. _Archive Summarization: https://github.com/VIAME/VIAME/tree/master/examples/archive_summarization

Overview
--------

This example demonstrates how to process video archives to create:

1. **Searchable Index** - An indexed database of video content for visual queries
2. **Detection Plots** - Timeline visualizations showing organism counts over time
3. **Interactive Interfaces** - Web-based tools for searching and browsing results

This is useful for processing large video archives from underwater cameras (MOUSS,
drop cameras, ROVs, etc.) where you want to quickly find specific content or
analyze species distributions over time.


Available Scripts
-----------------

+-------------------------------+-----------------------------------------------+
| Script                        | Description                                   |
+===============================+===============================================+
| summarize_and_index_videos    | Full processing: detection, plots, and index  |
| summarize_videos              | Detection and plots only (no search index)    |
| launch_timeline_interface     | View detection timeline plots                 |
| launch_search_interface       | Query the indexed database visually           |
+-------------------------------+-----------------------------------------------+


Quick Start
-----------

**Step 1: Process your videos**

Edit the script to point to your video directory, then run:

Linux:

.. code-block:: bash

   ./summarize_and_index_videos.sh

Windows:

.. code-block:: batch

   summarize_and_index_videos.bat


**Step 2: View results**

For timeline visualization:

.. code-block:: bash

   ./launch_timeline_interface.sh

For search interface:

.. code-block:: bash

   ./launch_search_interface.sh


Script Configuration
--------------------

The ``summarize_and_index_videos`` script accepts several key parameters:

- ``-d INPUT_DIRECTORY`` - Path to your video folder
- ``-p pipelines/index_mouss.pipe`` - Detection/indexing pipeline to use
- ``-plot-objects <species_list>`` - Comma-separated species names to plot
- ``-plot-threshold 0.25`` - Minimum detection confidence for plotting
- ``-frate 2`` - Frame rate for processing (frames per second)
- ``-plot-smooth 2`` - Smoothing factor for timeline plots
- ``--build-index`` - Enable building the search index
- ``--detection-plots`` - Enable generation of detection plots


Output Structure
----------------

After processing, results are organized as:

::

   database/
   ├── video1_detections.csv    # Detection results
   ├── video1_timeline.png      # Species timeline plot
   ├── video2_detections.csv
   ├── video2_timeline.png
   └── ...

   index/                       # Searchable index (if --build-index used)
   └── ...


Pipeline Selection
------------------

The default uses ``pipelines/index_mouss.pipe`` designed for MOUSS underwater
camera systems. For other camera types, you may need to modify the pipeline
or use a different pre-trained model.

Available pipelines vary by installation but may include:

- ``index_mouss.pipe`` - MOUSS bottom fish
- ``index_habcam.pipe`` - HabCam scallop surveys
- Custom pipelines for your specific use case


Related Examples
----------------

- ``search_and_rapid_model_generation/`` - Similar search interface with model training
- ``object_detection/`` - Running detectors on images and videos
- ``scoring_and_evaluation/`` - Evaluating detection results



