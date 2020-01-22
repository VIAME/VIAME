Hough Detection
===============

This pipelines features the hough_circle_detection algorithm in kwiver_algo_ocv.
This algorithm simply takes in a set of images and detectes any circles.
Then the detections boxes are drawn on the frame and displayed in a window.

Setup
-----

The pipefile associated with this tutorial are <kwiver build directory>examples/pipelines/hough_detector.pipe
You will need to have KWIVER_ENABLE_EXAMPLES turned on during CMake configuration of kwiver to get this file.
There is nothing more that will need to be done to execute this pipe file.
You can edit the edit the example_detector_on_video pipe file if you want to change the video file to be viewed,
You can edit the <kwiver build directory>examples/pipelines/hough_detector_images.txt if you want to add new images to be used.

Execution
---------

Run the following command from the kwiver build\bin directory (bin/release on windows)
Relativly point to the darknet_image.pipe or darknet_video.pipe file like this::

  # Windows Example :
  kwiver runner ..\..\examples\pipelines\hough_detector.pipe
  # Linux Example :
  ./kwiver runner ../examples/pipelines/hough_detector.pipe


Process Graph
-------------

The following image displays the pipeline graph.
Each process is linked to its associated definition page to learn more about it and the algorithms it uses.


.. graphviz:: ../_generated/graphviz/hough_detector.gv
