
=======================
Object Tracker Training
=======================

********
Overview
********

This document corresponds to the `object tracker training`_ example folder within a
VIAME desktop installation. VIAME provides a unified training interface for multiple
tracking algorithms. All tracker trainers accept the same input format and are invoked
through the ``viame train`` command with either a training configuration file or the
``-tt`` (tracker type) shorthand.

.. _object tracker training: https://github.com/VIAME/VIAME/blob/master/examples/object_tracker_training

For details on the available tracking algorithms themselves, see the
`object tracking examples`_. For training the upstream detection models used by
multi-target trackers, see the `object detector training examples`_.

.. _object tracking examples: https://github.com/VIAME/VIAME/blob/master/examples/object_tracking
.. _object detector training examples: https://github.com/VIAME/VIAME/blob/master/examples/object_detector_training

.. note::
   The ``viame train`` tracker training option (``-tt``) is a new addition and is
   currently in **beta**. It may change in future releases.

On the command line, tracker training is launched via::

    viame train -i /path/to/training/data -tt tracker_type --threshold 0.0

Or using a configuration file directly::

    viame train -i /path/to/training/data -c train_tracker_config.conf --threshold 0.0

After training completes, the resulting model or parameters are saved to the
``category_models`` directory.


***************************
Training Data Format
***************************

Training data for tracker training uses annotated tracks (not just per-frame
detections). The data should be organized in the following directory structure::

    [root_training_dir]/
        folder1/
            image001.png
            image002.png
            image003.png
            groundtruth.csv
        folder2/
            image001.png
            image002.png
            groundtruth.csv

Each ``groundtruth.csv`` file should contain track annotations in VIAME CSV format,
where each detection includes a track ID linking it to other detections of the same
object across frames. Unlike detector training, no ``labels.txt`` file is necessary --
all annotated tracks are used for training regardless of class.

Supported image formats: .jpg, .jpeg, .tif, .tiff, .png, .bmp.
Supported video formats: .mp4, .mpg, .mpeg, .avi, .wmv, .mov, .webm.


***********************
Available Trainers
***********************

VIAME includes trainers for three categories of tracking algorithms:

#. **Parameter-estimation trainers** -- estimate optimal Kalman filter and matching
   parameters from groundtruth tracks (no neural network training)
#. **Re-ID model trainers** -- train appearance-matching neural networks for
   identifying objects across frames
#. **Deep learning tracker trainers** -- train full tracking networks (SRNN, SiamMask)

+---------------------+----------+-----------+--------------+--------------------------------------------+
| Tracker             | Min Data | GPU Req.  | Training     | Best For                                   |
+=====================+==========+===========+==============+============================================+
| ByteTrack           | 10+      | No        | Param. est.  | Simple motion, sparse scenes               |
+---------------------+----------+-----------+--------------+--------------------------------------------+
| OC-SORT             | 10+      | No        | Param. est.  | Direction changes, moderate motion          |
+---------------------+----------+-----------+--------------+--------------------------------------------+
| DeepSORT            | 50+      | Yes       | Re-ID model  | Dense scenes, occlusion recovery           |
+---------------------+----------+-----------+--------------+--------------------------------------------+
| BoT-SORT            | 50+      | Yes       | Re-ID model  | Moving cameras, fast motion                |
+---------------------+----------+-----------+--------------+--------------------------------------------+
| SRNN                | 100+     | Yes       | Full network | Complex scenarios, many interacting objects|
+---------------------+----------+-----------+--------------+--------------------------------------------+
| SiamMask            | 50+      | Yes       | Full network | User-initialized visual tracking           |
+---------------------+----------+-----------+--------------+--------------------------------------------+
| Adaptive            | 10+      | Varies    | Auto-selects | Unsure which tracker to use                |
+---------------------+----------+-----------+--------------+--------------------------------------------+

The "Min Data" column indicates the recommended minimum number of annotated tracks.


ByteTrack
----------

ByteTrack training estimates optimal Kalman filter parameters from groundtruth tracks.
No neural network is trained -- instead, the trainer analyzes motion statistics to
estimate position uncertainty weights, velocity uncertainty weights, detection
thresholds, and the track buffer size.

- **No GPU required** -- pure statistical parameter estimation
- Estimates: position/velocity uncertainty, detection thresholds, track buffer
- Outputs ``bytetrack_params.json`` with tuned parameters
- Best for simple motion scenarios with sparse scenes and continuous tracks

Training::

    viame train -i training_data -tt bytetrack --threshold 0.0

OC-SORT
--------

OC-SORT extends ByteTrack with velocity direction consistency (VDC). The trainer
estimates all ByteTrack parameters plus an additional VDC weight based on direction
change statistics in the groundtruth tracks.

- **No GPU required** -- parameter estimation with direction analysis
- Better than ByteTrack when objects frequently change direction
- Outputs ``ocsort_params.json`` with tuned parameters

Training::

    viame train -i training_data -tt ocsort --threshold 0.0

DeepSORT
---------

DeepSORT training involves training a Re-ID (re-identification) neural network that
learns to extract appearance features for matching detections across frames. The
network learns to distinguish individual objects by their visual appearance, enabling
recovery after occlusion.

- **GPU required** -- trains a ResNet-18 or ResNet-50 Re-ID backbone
- Requires larger objects for good crop features (min ~2048 pixel area)
- Crop size: 128x64 (HxW), embedding dimension: 512
- Default: 50 epochs, batch size 32, learning rate 0.0003
- Best for dense scenes where appearance distinguishes objects

Training::

    viame train -i training_data -tt deepsort --threshold 0.0

BoT-SORT
---------

BoT-SORT combines Re-ID model training (like DeepSORT) with camera motion
compensation (CMC). The trainer learns appearance features and estimates tracking
parameters tuned for scenarios with camera motion.

- **GPU required** -- trains a Re-ID backbone
- Camera motion compensation for moving cameras (ROVs, drones, handheld)
- EMA-based feature aggregation for temporal smoothness
- Best for fast motion scenarios with camera shake

Training::

    viame train -i training_data -tt botsort --threshold 0.0

SRNN (Structured RNN)
----------------------

The SRNN trainer runs a complex multi-stage training pipeline involving:

1. Data preparation from groundtruth tracks
2. Siamese network training for visual similarity features
3. Feature extraction using the trained Siamese model
4. Individual LSTM training for Appearance (A), Interaction (I), and Motion (M) models
5. Combined SRNN model training

This is the most data-intensive tracker trainer, requiring 100+ annotated tracks
with long track lengths and many concurrent objects. It is the best choice for
complex tracking scenarios with frequent occlusions, interacting targets, and
dense scenes.

- **GPU required** -- multi-stage deep learning pipeline
- Needs substantial data (100+ annotated tracks recommended)
- Best for complex scenarios with many concurrent, interacting objects
- Training timeout: 1 week default

Training::

    viame train -i training_data -tt srnn --threshold 0.0

SiamMask
---------

SiamMask training trains a Siamese network for visual object tracking. The trainer
learns to match object templates across frames, producing both bounding boxes and
segmentation masks. This trains the user-initialized tracker (not the multi-target
tracker).

- **GPU required** -- trains a Siamese tracking network
- Default: 20 epochs, automatic batch size based on GPU memory
- Crop size: 511px (standard for SiamMask architecture)
- Can fine-tune from a pre-trained seed model
- Training timeout: 2 weeks default

Training::

    viame train -i training_data -tt siammask --threshold 0.0


****************************
Adaptive Tracker Training
****************************

VIAME provides an adaptive training mode that automatically analyzes tracking data
statistics and selects the best trainer(s) for the given dataset. The adaptive trainer
computes:

- **Track statistics**: counts, lengths (short/medium/long), fragmentation rates
- **Motion patterns**: velocity (mean, max, std), direction changes
- **Scene density**: concurrent tracks per frame (sparse/medium/dense)
- **Object sizes**: for Re-ID crop sizing decisions
- **Appearance consistency**: within-track size variance
- **Occlusion/proximity**: close track pairs, potential ID switches

Based on these statistics, the adaptive trainer selects up to 3 algorithms from:
ByteTrack (10+ tracks), OC-SORT (10+ tracks), DeepSORT (50+ tracks), BoT-SORT
(50+ tracks), and SRNN (100+ tracks). Each trainer has hard requirements (minimum
track count, minimum track length, minimum object area) and soft preferences that
are scored against the data profile.

Training::

    viame train -i training_data -c train_tracker_adaptive.conf --threshold 0.0

The adaptive trainer outputs a ``tracking_data_statistics.json`` file with the
computed dataset statistics for diagnostics.


**********************
Choosing a Tracker
**********************

When deciding which tracker to train, consider your data characteristics:

**Few tracks, simple motion (10--50 tracks):**
Use **ByteTrack** or **OC-SORT**. Both are parameter-estimation only (no GPU needed)
and work well with limited data. Choose OC-SORT if objects frequently change direction.

**Dense scenes with occlusion (50+ tracks):**
Use **DeepSORT** -- its learned Re-ID features allow recovery after occlusion by
matching object appearance. Requires objects large enough for meaningful appearance
crops (~2048+ pixel area).

**Moving camera (drones, ROVs, handheld):**
Use **BoT-SORT** -- its camera motion compensation handles the global image motion
that confuses purely motion-based trackers.

**Complex scenarios with many interacting objects (100+ tracks):**
Use **SRNN** -- its combined appearance, interaction, and motion LSTMs model complex
multi-object dynamics. Requires the most training data but handles the hardest
scenarios.

**User-initialized tracking (annotation assist):**
Use **SiamMask** training to fine-tune the visual tracker for your specific domain.

**Unsure what to use:**
Run the **adaptive** trainer -- it will analyze your tracking data and select the best
option(s) automatically.


***************
Example Scripts
***************

Training Scripts
-----------------

``train_bytetrack.sh`` / ``.bat``
    Train ByteTrack Kalman filter parameters from groundtruth tracks.

``train_ocsort.sh`` / ``.bat``
    Train OC-SORT parameters (ByteTrack + velocity direction consistency).

``train_deepsort.sh`` / ``.bat``
    Train a DeepSORT Re-ID appearance model.

``train_botsort.sh`` / ``.bat``
    Train a BoT-SORT Re-ID model with camera motion compensation.

``train_srnn.sh`` / ``.bat``
    Train the SRNN multi-stage tracking model.

``train_siammask.sh`` / ``.bat``
    Train a SiamMask visual tracking network.

``train_adaptive.sh`` / ``.bat``
    Run adaptive tracker training (auto-selects best trainer(s)).

``train_st_tracker_viame_csv.sh`` / ``.bat``
    Legacy SiamMask training script (calls trainer directly).
