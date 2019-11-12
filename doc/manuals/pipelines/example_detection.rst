Example Detection
=================

These pipelines features the example_detection algorithm in kwiver_algo_core.
This algorithm simply takes in a set of images or a video and generates dummy detections for each image/frame.
Then the detections boxes are drawn on the frame and displayed in a window.
It is a good example for how to use detection data types in kwiver.

Setup
-----

The pipefiles associated with this tutorial are <kwiver build directory>examples/pipelines/example_detector_on_image.pipe
and <kwiver build directory>examples/pipelines/example_detector_on_video.pipe
You will need to have KWIVER_ENABLE_EXAMPLES turned on during CMake configuration of kwiver to get this file.
There is nothing more that will need to be done to execute this pipe file.
You can edit the edit the example_detector_on_video pipe file if you want to change the video file to be viewed,
or the <kwiver build directory>examples/pipelines/image_list.txt if you want to change images to be viewed.

Execution
---------

Run the following command from the kwiver build\bin directory (bin/release on windows)
Relativly point to the darknet_image.pipe or darknet_video.pipe file like this::

  # Windows Example :
  kwiver runner ..\..\examples\pipelines\example_detector_on_image.pipe
  # Linux Example :
  ./kwiver runner ../examples/pipelines/example_detector_on_image.pipe

  # Windows Example :
  kwiver runner ..\..\examples\pipelines\example_detector_on_video.pipe
  # Linux Example :
  ./kwiver runner ../examples/pipelines/example_detector_on_video.pipe

Process Graph
-------------

The following image displays the pipeline graph.
Each process is linked to its associated definition page to learn more about it and the algorithms it uses.

example_detector_on_image
~~~~~~~~~~~~~~~~~~~~~~~~~

.. graphviz:: ../_generated/graphviz/example_detector_on_image.gv

example_detector_on_video
~~~~~~~~~~~~~~~~~~~~~~~~~

.. graphviz:: ../_generated/graphviz/example_detector_on_video.gv
