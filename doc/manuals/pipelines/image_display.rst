Simple Image
============

The following pipeline will take in a set of images and display them in a window.

Setup
-----

The pipefile associated with this tutorial is located in the <kwiver build directory>examples/pipelines/image_display.pipe
You will need to have KWIVER_ENABLE_EXAMPLES turned on during CMake configuration of kwiver to get this file.
There is nothing more that will need to be done to execute this pipe file.
You can edit the <kwiver build directory>examples/pipelines/image_list.txt if you want to add new images to be viewed.

Execution
---------

Run the following command from the kwiver build\bin directory (bin/release on windows)
Relativly point to the darknet_image.pipe or darknet_video.pipe file like this::

  # Windows Example :
  kwiver runner ..\..\examples\pipelines\image_display.pipe
  # Linux Example :
  ./kwiver runner ../examples/pipelines/image_display.pipe

Process Graph
-------------

The following image displays the pipeline graph.
Each process is linked to its associated definition page to learn more about it and the algorithms it uses.

.. graphviz:: ../_generated/graphviz/image_display.gv
