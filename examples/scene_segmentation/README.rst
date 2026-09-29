
==================
Scene Segmentation
==================

********
Overview
********

This document corresponds to the `scene
segmentation <https://github.com/VIAME/VIAME/tree/main/examples/scene_segmentation>`__
example folder within a VIAME desktop installation.

Segmentation assigns a label to regions of an image rather than to a box around an
object. VIAME covers it at three levels:

- **Object masks**, a polygon outlining each detected object
- **Fine-tuned segmentation models**, for imagery or targets where the stock SAM models
  outline objects poorly
- **Frame-level background labels**, where the whole image is classified by what the
  scene contains, such as the type of seafloor

***********************************
Adding Masks to Existing Detections
***********************************

The ``utility_add_segmentations`` pipelines take a file of box detections and add a
polygon to each one. They read ``detections.csv`` and write ``computed_detections.csv``.

.. list-table::
   :header-rows: 1

   * - Pipeline
     - Method
   * - ``utility_add_segmentations_watershed.pipe``
     - OpenCV watershed, no model or GPU required
   * - ``utility_add_segmentations_sam2.pipe``
     - SAM2, from the SAM2 add-on
   * - ``utility_add_segmentations_sam3.pipe``
     - SAM3, from the SAM3 add-on
   * - ``utility_add_segmentations_default.pipe``
     - Installed by the SAM2 add-on, runs the SAM2 pipeline

Masks can also be drawn interactively in DIVE by clicking on an object, or produced from
a text prompt. See `interactive segmentation in
DIVE <https://viame.github.io/VIAME/sections/annotation_and_visualization.html>`__
and `text-prompted detection and
tracking <https://viame.github.io/VIAME/sections/search_and_rapid_model_generation.html>`__.

Detectors that output masks directly are trained like any other detector, from
annotations that include polygons. See the RF-DETR segmentation configurations in
`detector
training <https://viame.github.io/VIAME/sections/object_detector_training.html>`__.

**********************
Fine-Tuning SAM Models
**********************

The stock SAM weights are trained on general imagery. Fine-tuning them on a small set of
outlined examples helps when the masks they produce are consistently wrong for a
particular problem, for example in turbid water, on low-contrast or camouflaged animals,
or on targets with thin parts such as fins and antennae.

Annotations
===========

Training uses the same folder layout as detector training: one folder per sequence, each
with its imagery and a groundtruth file, plus a ``labels.txt``. Each annotated object
should have a polygon as well as a box. An object with only a box is trained as if the
whole box were the mask, which teaches the model to produce rectangles, so outline every
object that is used.

Training
========

With the SAM2 and SAM3 add-ons installed:

.. code-block:: bash

   viame train -i training_data -c train_detector_sam3.conf --threshold 0.0

This fine-tunes the SAM2.1 base-plus weights that ship with the SAM2 add-on. During
training each annotated box is given to the model as a prompt, and the mask it predicts
is compared with the annotated polygon. The fine-tuned weights are written to
``trained_model/sam3_finetuned.pth``.

The main settings in ``train_detector_sam3.conf`` are:

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Purpose
   * - ``sam2_model_path``
     - ``models/sam2_hbp.pt``
     - Weights to start from.
   * - ``sam2_config_file``
     - ``configs/sam2.1/sam2.1_hiera_b+.yaml``
     - Model definition matching those weights. Change both together.
   * - ``freeze_image_encoder``
     - true
     - Leaves the image encoder unchanged. Recommended, as it holds most of the model
       and is the easiest part to overfit.
   * - ``freeze_prompt_encoder``
     - true
     - Leaves the prompt encoder unchanged.
   * - ``learning_rate``
     - 1e-5
     - Kept low so the model adapts without losing what it already knows.
   * - ``max_epochs``
     - 50
     - Upper limit on training length.
   * - ``batch_size``
     - 4
     - Images per step. Reduce if the GPU runs out of memory.
   * - ``chip_width``, ``chip_height``
     - 1024
     - Network input size.
   * - ``augmentation``
     - ``standard``
     - ``standard``, ``complex`` or ``none``.

With both encoders frozen only the mask decoder is trained, which is the safest starting
point for a small dataset. Unfreeze the image encoder only when there is a large amount
of outlined data and the frozen result is not good enough.

Fine-tuning for tracking
========================

SAM3 can also be fine-tuned to follow an object through a video, using track
annotations:

.. code-block:: bash

   viame train -i training_data -c train_tracker_sam3.conf

Tracks are cut into short clips. The first frame of a clip is prompted with the
annotated box, and each later frame is prompted only with the mask predicted for the
frame before it, which is how the model runs when tracking. This configuration starts
from the SAM3 weights shipped with the add-on, and writes its result to
``trained_model/sam3_tracker_finetuned.pth``.

Using the fine-tuned model
==========================

The file written by ``train_detector_sam3.conf`` holds SAM2 weights, so it is used by
the SAM2 pipelines. It is saved as a bare set of weights, which has to be wrapped once
before those pipelines can load it:

.. code-block:: python

   import torch

   weights = torch.load("trained_model/sam3_finetuned.pth", map_location="cpu")
   torch.save({"model": weights}, "sam2_finetuned.pt")

Then point the pipeline at the new file, leaving its ``cfg`` setting on the matching
model definition:

.. list-table::
   :header-rows: 1

   * - Used by
     - File
     - Setting
   * - Adding masks to detections
     - ``utility_add_segmentations_sam2.pipe``
     - ``refiner:sam2:checkpoint``
   * - Interactive segmentation in DIVE
     - ``common_sam2_segmenter.conf``
     - ``sam2:checkpoint``

*************************************
Frame-Level Background Classification
*************************************

When the question is what the scene contains rather than where each object is, outlining
regions is often unnecessary. A full-frame classifier labels the whole image, which
suits background properties such as substrate type, habitat, or water clarity. It needs
only one label per image to train, and it runs quickly.

Full-frame classifiers are trained as described in `frame level
classification <https://viame.github.io/VIAME/sections/frame_level_classification.html>`__.
In DIVE, the ``empty frame lbls`` utility pipelines add a whole-frame box to each image
so that frame-level labels can be applied to it.

Several properties can be reported for the same image by running one classifier per
property and merging the results. Each classifier reports a single detection covering
the whole frame, carrying the score of every class it knows.

Example: HabCam substrate classification
========================================

The HabCam add-on includes a working example, ``detector_habcam_substrate.pipe``. It
runs 22 full-frame classifiers on each seafloor image and merges their output into one
``computed_detections.csv``. Each classifier answers one question about the background,
including:

- Sediment: gravel levels, shell levels, whole shells, boulders, sand waves
- Beds: scallop, mussel, clam, sand dollar and sea star beds
- Attached and encrusting life: sponges, tunicates, bryozoans, pennatulids, burrowing
  anemones

Every classifier has a ``background`` class for images that do not show the property. It
is reported under the name of the model, as ``no_boulder`` or ``no_scallop_bed`` for
example, so the merged output states for each property whether it is present. The
pipeline is a useful starting point for other surveys: replace the models with
classifiers trained on local frame labels and keep the structure.


.. dive-crosslink

******************
DIVE Documentation
******************

* `DIVE interactive annotation <https://viame.github.io/VIAME/sections/dive/Interactive-Annotation.html>`__ covers point-click segmentation in the interface
* `DIVE pipelines and training <https://viame.github.io/VIAME/sections/dive/Pipeline-Documentation.html>`__ lists the utility pipelines, including those that add segmentations


********************
Code and Build Flags
********************

Flags to enable when building VIAME from source for this example:

* ``VIAME_ENABLE_ONNX``
* ``VIAME_ENABLE_OPENCV``
* ``VIAME_ENABLE_PYTHON``
* ``VIAME_ENABLE_PYTORCH``
* ``VIAME_ENABLE_PYTORCH-SAM2``
* ``VIAME_ENABLE_PYTORCH-SAM3``

Add-ons providing the pipelines or models used: ``habcam``, ``sam2``, ``sam3``.

Command line tools:

* tools/segment.py -- ``viame segment``
* tools/train.cxx -- ``viame train``

Pipeline and configuration files:

* configs/pipelines/utility_add_segmentations_watershed.pipe
* configs/add-ons/sam2/utility_add_segmentations_sam2.pipe
* configs/add-ons/sam3/utility_add_segmentations_sam3.pipe
* configs/add-ons/sam2/utility_add_segmentations_default.pipe
* configs/add-ons/sam3/train_detector_sam3.conf
* configs/add-ons/sam3/train_tracker_sam3.conf
* configs/add-ons/sam2/common_sam2_segmenter.conf
* configs/add-ons/habcam/detector_habcam_substrate.pipe

Source code:

* plugins/pytorch/sam2_refiner.py
* plugins/pytorch/sam3_refiner.py
* plugins/pytorch/sam3_trainer.py
* plugins/core/windowed_refiner.cxx
* plugins/opencv/windowed_refiner.cxx
* plugins/onnx/onnx_classifier.py
