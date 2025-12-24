
========================
Object Tracking Examples
========================

********
Overview
********

This document corresponds to the `object tracking`_ example folder within a VIAME desktop
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

.. image:: https://github.com/Kitware/dive/blob/main/docs/images/Banner.png
   :scale: 50
   :align: center
|
Automatic multi-target trackers (MTT) link detections (produced by a seperate detection
algorithm, see `detection examples`) into tracks. This is accomplished by combining
appearance features for the object being tracked, with kinematic (motion) information as to
the last known location(s) of the object. The current default implementation of this is designed
for either roughly stationary cameras, or cameras with a little bit of camera motion. Excessive
camera motion (say >10% movement per frame) can contribute to worse results without re-training
or updating the default models. The default tracker in the system can be added to any pipeline
file with a detector in it. Sometimes this may yield sufficient results, though other times the
tracker might need to be fine-tuned for a new problem. While object detectors can currently be
re-trained from user interfaces, the tracker cannot, requiring a seperate python script call.
For more information about this process, contact the viame-web@kitware.com mailing list.
Example trackers can be found in the 'Trackers' dropdown in the DIVE interface.
Example CLI scripts in this folder for MTT trackers include:

* run_generic_tracker (processes a single sequence)
* run_fish_tracker (processes a single sequence)
* bulk_run_fish_tracker (processes multiple sequences)

.. _detection examples: https://github.com/VIAME/VIAME/blob/master/examples/object_detection

The current default model for performing MTT in VIAME is a variant of the RNN network and 
features described in the "Tracking the Untrackable" paper [TUT17]_, where new detections
are tested to see if they belong to an existing track using a variant of the classifier
described in the paper. A hungarian matrix is then used on all track/detection combinations
to make final linking decisions. These trackers are automatically enabled in binary
installations, but when building from source the VIAME_ENABLE_PYTORCH and
VIAME_ENABLE_PYTORCH-VISION enable flags are required. There are a number of pieces of code
used in the approach, including:

* packages/kwiver/python/kwiver/sprokit/processes/pytorch/srnn_tracker.py
* configs/pipelines/tracker_fish.sfd.pipe
* packages/kwiver/vital/types/object_track_set.h
* packages/kwiver/python/kwiver/vital/types/object_track_set.h
* packages/kwiver/python/kwiver/vital/types/object_track_set.cxx
* packages/pytorch-libs/torchvision
* packages/pytorch

.. [TUT17] Sadeghian et al. "Tracking the untrackable: Learning to track multiple cues with long-term dependencies." IEEE ICCV 2017.

*************************
User-Initialized Trackers
*************************

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/computed_track_example.png
   :scale: 50
   :align: center

User-initialized tracking allows users to draw a box on the first frame of an object
(or objects) that the user wants to start tracking from, and then track the object(s)
on subsequent frames. This is useful for rapidly generating track-level annotations
without having to annotate the object on every frame. These pipelines can be run in
the utility dropdown in the DIVE interface, in the VIEW interface pipelines dropdown,
or from the command line in the following scripts:

* run_user_init_tracking
* bulk_run_user_init_tracking

When running on a sequence, detections or tracks of size 1 will trigger user-initialized
tracking. Any tracks of length greater than 1 will not trigger user-initialized tracking
in order to not change them when aiding with annotation generation.
The current default model for performing user-initialized tracking in VIAME is a variant
of the [SiamMask]_ and [SiamRPN]_ algorithms. Re-training of these classifiers is 
currently available in the `object tracker training`_ example folder. These trackers are
automatically enabled in binary installations, but when building from source
the VIAME_ENABLE_PYTORCH and VIAME_ENABLE_PYTORCH-PYSOT enable flags are required.
There are a number of pieces of code used in the system, including:

.. _object tracker training: https://github.com/VIAME/VIAME/blob/master/examples/object_tracker_training

* packages/kwiver/python/kwiver/sprokit/processes/pytorch/pysot_tracker.py
* configs/pipelines/utility_track_selections_default_mask.pipe
* configs/pipelines/utility_track_selections_fish_box_only.pipe
* packages/kwiver/vital/types/object_track_set.h
* packages/kwiver/python/kwiver/vital/types/object_track_set.h
* packages/kwiver/python/kwiver/vital/types/object_track_set.cxx
* packages/pytorch-libs/pysot
* packages/pytorch

.. [SiamMask] Hu et al. "SiamMask: A framework for fast online object tracking and segmentation." IEEE PAMI 2023.
.. [SiamRPN] Li et al. "SiamRPN++: Evolution of siamese visual tracking with very deep networks." IEEE CVPR 2019.

Extensions
----------

A multi-target version of the SiamMask tracker is also available for use in pipelines
as an alternative to the MTT described in the prior section. When combined with a
detection node, the tracker will automatically initialize new tracks when detections
are above some specified threshold. A basic IOU algorithm prevents multiple tracks
from being spawned on detections on subsequent frames.

With additional settings modifications, these trackers also allow for longer term
re-initialization when the target is lost via the Siam methods, but this feature is not
available on the public version of VIAME.


***************************
Registration-Based Trackers
***************************

Registration-based trackers use frame-to-frame image registrations to identify the same
locations in each frame in corresponding frames. These mapped locations are then used
to link the same objects in some world (aka ground) plane. In the context of VIAME,
these trackers are currently used for two purposes: tracking objects on the ground
in aerial imagery, or tracking objects on the ground in fast moving benthic camera
systems pointed at the sea floor.

There are a number of pieces of code used in the approach, including:

* packages/kwiver/python/kwiver/sprokit/processes/multicam_homog_tracker.py
* configs/add-ons/sea-lion/tracker_(multiple).pipe
* packages/kwiver/vital/types/object_track_set.h
* packages/kwiver/algos
