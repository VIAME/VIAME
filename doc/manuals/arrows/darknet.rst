Darknet
=======

Darknet is an open source neural network framework written in C and CUDA.

The following algorithm implementations use Darknet

========================================== ============================================================
:ref:`darknet_detector<darknet_detector>`  .. doxygenclass:: kwiver::arrows::darknet::darknet_detector 
:ref:`darknet_trainer<darknet_trainer>`    .. doxygenclass:: kwiver::arrows::darknet::darknet_trainer  
========================================== ============================================================

In the pipe files, you can tune the algorithm with these variables :
  - darknet:thresh
  - darknet:hier_thresh
  - darknet:gpu_index
  
FAQ
---

I am running out of memory in CUDA...
  Try one or both of these suggestions:
  - Change the darknet/models/virat.cfg variables height,weight to smaller powers of 32
  - Change the darknet/models/virat.cfg variables batch and subdivisions (make sure they are still the same)