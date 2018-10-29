.. image:: /_images/KWIVER_logo.png
   :align: center
   :height: 150px

Introduction
============

KWIVER is a fully featured toolkit for developing Computer Vision *Systems*,
a capability that goes beyond the support of simply developing Computer Vision *Software*.

This distinction is an important one.  There are myriad of software frameworks
that facilitate the development of computer vision software, most notably the
venerable `OpenCV <https://opencv.org>`_, but also including `VXL <http://vxl.sourceforge.net>`_,
`scikit-image <https://scikit-image.org>`_ and a wide range of
others.  The current Deep Learning revolution has additionally spawned  a number
of software frameworks for doing deep learning based computer vision including
`Caffe <http://caffe.berkeleyvision.org>`_, `PyTorch <https://pytorch.org>`_,
`Tensorflow <https://www.tensorflow.org>`_ and others.

Each of these frameworks has their own unique set of capabilities, target user
community, dependencies and levels of difficulty and complexity.  When
developing computer vision *software*, the task frequently boils down to selecting
the most appropriate framework to work with and proceeding from there.

As the task at hand becomes more complicated, however, the burden on the
supporting frameworks, and the task-specific software developed using those
frameworks, becomes heavier.  Real world problems might be better solved by, for
example, fusing OpenCV based motion detections with Faster-RCNN (Caffe) based
appearance detections and then filtering the result against a new
state-of-the-art image segmentation neural network that runs in yet another deep
learning framework. Couple this with the understanding that computer vision
algorithms traditionally are extremely compute intensive, doubly so when one
considers the GPU requirements of modern deep learning frameworks and it is
clear that building computer vision *systems* is a daunting task.

KWIVER is designed and engineered from the ground up to support the development
of systems of this nature.  It has first class features that are designed to
allow the development of fully elaborated systems using a wide variety of
computer vision frameworks -- both traditional and deep learning based -- and a
wide variety of stream processing and multi-processing topologies.  KWIVER
based systems have scaled from small embedded computing platforms such as the
NVIDIA TX2 to large cloud based infrastructure and a wide variety of platforms
in between.

KWIVER is a collection of C++ libraries with C and Python bindings
and uses an permissive `BSD License <LICENSE>`_.

Visit the `repository <https://github.com/Kitware/kwiver>`_ on how to get and build the KWIVER code base
