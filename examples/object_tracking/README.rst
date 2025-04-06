
========================
Object Tracking Examples
========================

********
Overview
********

This document corresponds to the `object tracking` example folder within a VIAME desktop
installation. Object tracking attempts to identify the same object across sequential frames
in either video or image sequences. VIAME currently contains 3 core types of trackers:

.. _object tracking: https://github.com/VIAME/VIAME/blob/master/examples/object_tracking

#. Automatic multi-target trackers, which link object detections on individual frames
#. User-initialized tracking, requiring the user to mark the object on a first start frame
#. Registration-based trackers, to track objects in the same location with a moving camera

Tracking can either be run from scripts, such as those contained within this example, or
from one of the user interfaces within VIAME (e.g. DIVE, VIEW, SEAL).

*******************************
Automatic Multi-Target Trackers
*******************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/computed_track_example.png
   :scale: 60
   :align: center
|
Automatic multi-target trackers (MTT) within VIAME link detections (produced by another 
detection algorithm, see `detection examples`) into tracks. This is accomplished by combining
appearance features for the object being tracked, with kinematic (motion) information as to
the last known location(s) of the object. The current default implementation of this is designed
for either roughly stationary cameras, or cameras with a little bit of camera motion. Excessive
camera motion (say >10% movement per frame) can contribute to worse results without re-training
or updating the default models. The default tracker in the system can be added to any pipeline
file with a detector in it. Sometimes this may yield sufficient results, though other times the
tracker might need to be fine-tuned for a new problem. While object detectors can currently be
re-trained from user interfaces, the tracker cannot, requiring a seperate python script call.
For more information about this process, contact the viame-web@kitware.com mailing list.

.. _object tracking: https://github.com/VIAME/VIAME/blob/master/examples/object_detection

The current default model for performing MTT in VIAME is a variant of the RNN network and 
features described in the "Tracking the Untrackable [TUT2021]" paper. There are a number
of pieces of code used in the approach, including:

[]
[]
[]
[]
[]

*************************
User-Initiliazed Trackers
*************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/computed_track_example.png
   :scale: 60
   :align: center
|
[]

With additional settings modifications, it supports long term linking including

The current default model for performing MTT in VIAME is a variant of the RNN network and 
features described in the "Tracking the Untrackable [TUT2021]" paper. There are a number
of pieces of code used in the approach, including:



***************************
Registration-Based Trackers
***************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/computed_track_example.png
   :scale: 60
   :align: center
|
The current default model for performing MTT in VIAME is a variant of the RNN network and 
features described in the "Tracking the Untrackable [TUT2021]" paper. There are a number
of pieces of code used in the approach, including:

