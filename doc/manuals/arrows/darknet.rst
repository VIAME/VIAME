Darknet
=======


..  _darknet_detector:

Darknet Detector Algorithm
--------------------------

..  doxygenclass:: kwiver::arrows::darknet::darknet_detector
    :project: kwiver
    :members:

..  _darknet_trainer:

Darknet Trainer Algorithm
-------------------------

..  doxygenclass:: kwiver::arrows::darknet::darknet_trainer
    :project: kwiver
    :members:
    

FAQ
---

I am running out of memory in CUDA...
  Try one or both of these suggestions:
  - Change the darknet/models/virat.cfg variables height,weight to smaller powers of 32
  - Change the darknet/models/virat.cfg variables batch and subdivisions (make sure they are still the same)