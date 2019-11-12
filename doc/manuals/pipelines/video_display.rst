Simple Video
=============

The following pipeline will take in a video file and play it in a window.

Setup
-----

The pipefile associated with this tutorial is located in the <kwiver build directory>examples/pipelines/video_display.pipe
You will need to have KWIVER_ENABLE_EXAMPLES turned on during CMake configuration of kwiver to get this file.
There is nothing more that will need to be done to execute this pipe file.
You can edit the edit the pipe file if you want to change the video file to be viewed.

Execution
---------

Run the following command from the kwiver build\bin directory (bin/release on windows)
Relativly point to the darknet_image.pipe or darknet_video.pipe file like this::

  # Windows Example :
  kwiver runner ..\..\examples\pipelines\video_display.pipe
  # Linux Example :
  ./kwiver runner ../examples/pipelines/video_display.pipe

Process Graph
-------------

The following image displays the pipeline graph.
Each process is linked to its associated definition page to learn more about it and the algorithms it uses.

.. graphviz:: ../_generated/graphviz/video_display.gv
