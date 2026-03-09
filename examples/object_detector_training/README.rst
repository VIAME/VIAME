
==========================
Detector Training Examples
==========================

********
Overview
********

This document corresponds to the `object detector training`_ example folder within a
VIAME desktop installation. VIAME provides a unified training interface for multiple
object detection frameworks. All trainers accept the same input format and are invoked
through the ``viame train`` command with a training configuration file.

.. _object detector training: https://github.com/VIAME/VIAME/blob/master/examples/object_detector_training

For details on the available detection algorithms themselves, see the
`object detection examples`_.

.. _object detection examples: https://github.com/VIAME/VIAME/blob/master/examples/object_detection

In the **DIVE** interface, training can be launched from the training menu by selecting
a configuration. On the command line, training is launched via::

    viame train -i /path/to/training/data -c train_config.conf --threshold 0.0

After training completes, the resulting model is saved to a ``category_models``
directory. The trained model can then be run using ``run_trained_model`` scripts or
loaded in DIVE.


***************************
Training Data Format
***************************

Training data should be organized in the following directory structure::

    [root_training_dir]/
        labels.txt
        folder1/
            image001.png
            image002.png
            image003.png
            groundtruth.csv
        folder2/
            image001.png
            image002.png
            groundtruth.csv

Groundtruth annotations can be in any supported format (e.g. viame_csv, kw18, habcam).
The ``labels.txt`` file contains a list of output categories (one per line) for the
trained model.

Alternatively, training data can be specified explicitly using the ``--input-list``,
``--input-truth``, and ``--labels`` flags::

    viame train --input-list images.txt --input-truth annotations.csv \
                --labels labels.txt -c train_config.conf --threshold 0.0

Supported image formats include: .jpg, .jpeg, .tif, .tiff, .png, .sgi, .bmp, .pgm.
Supported video formats include: .mp4, .mpg, .mpeg, .avi, .wmv, .mov, .webm, .ogg.


************
Labels Files
************

The ``labels.txt`` file controls which categories are trained, allows synonyms for the
same category, and supports class hierarchies.

**Synonyms:** Multiple names on the same line are treated as the same output class. The
first name becomes the output label::

    speciesA speciesB speciesC
    speciesD

This trains a model with two output classes: ``speciesA`` (which also matches
``speciesB`` and ``speciesC`` annotations) and ``speciesD``.

**Filtering:** Categories omitted from ``labels.txt`` are excluded from training::

    speciesA
    speciesC
    speciesD

This produces three output classes; any ``speciesB`` annotations are ignored.

**Hierarchies:** Parent-child relationships can be specified for frameworks that
support hierarchical classification::

    genusA
    speciesC :parent=genusA
    speciesD

If no ``labels.txt`` is provided, all unique labels in the groundtruth are used.


***********************
Available Trainers
***********************

The table below summarizes the available detector training frameworks. Choose based
on your data size, object characteristics, and compute resources.

+---------------------+----------+-----------+----------+--------------------------------------------+
| Framework           | Min Data | GPU Req.  | Speed    | Best For                                   |
+=====================+==========+===========+==========+============================================+
| Netharn CFRNN       | 500+     | Yes       | Moderate | General purpose (default), mixed objects    |
+---------------------+----------+-----------+----------+--------------------------------------------+
| Netharn CFRNN Grid  | 300+     | Yes       | Moderate | Small objects, dense scenes (20+ obj/frame) |
+---------------------+----------+-----------+----------+--------------------------------------------+
| RF-DETR             | 400+     | Yes       | Moderate | Dense scenes with occlusion, multi-scale   |
+---------------------+----------+-----------+----------+--------------------------------------------+
| MIT-YOLO v9         | 200+     | Yes       | Fast     | Multi-scale objects, real-time inference    |
+---------------------+----------+-----------+----------+--------------------------------------------+
| Darknet YOLO        | 200+     | Yes       | Fast     | Production, well-tested                    |
+---------------------+----------+-----------+----------+--------------------------------------------+
| Detectron2 FRCNN    | 300+     | Yes       | Moderate | General purpose                            |
+---------------------+----------+-----------+----------+--------------------------------------------+
| LitDet FRCNN        | 300+     | Yes       | Moderate | Large/tall objects, sparse scenes           |
+---------------------+----------+-----------+----------+--------------------------------------------+
| LitDet SSD          | 300+     | Yes       | Fast     | Lightweight / fast inference                |
+---------------------+----------+-----------+----------+--------------------------------------------+
| Netharn Mask R-CNN   | 500+     | Yes       | Slow     | Instance segmentation with masks           |
+---------------------+----------+-----------+----------+--------------------------------------------+
| MMDetection         | 500+     | Yes       | Moderate | Advanced configs, distributed training     |
+---------------------+----------+-----------+----------+--------------------------------------------+
| SVM                 | 50+      | No        | Fast     | Small datasets, CPU-only, quick baseline   |
+---------------------+----------+-----------+----------+--------------------------------------------+
| Adaptive            | 50+      | Varies    | Varies   | Auto-selects best trainer(s)               |
+---------------------+----------+-----------+----------+--------------------------------------------+

The "Min Data" column indicates the recommended minimum number of annotations per
class for reasonable results. More data generally improves performance.


Netharn Cascade Faster R-CNN (Default)
---------------------------------------

The default detector trainer in VIAME. Uses a Cascade Faster R-CNN with ResNeXt-101
backbone at 640x640 resolution. Provides strong accuracy across diverse object types.

- Automatic batch size and learning rate selection
- Windowed/tiled processing for high-resolution imagery
- Complex data augmentation (flips, color jitter, scale variation)
- Supports continuing training from a previously trained model

Training::

    viame train -i training_data -c train_detector_netharn_cfrnn.conf --threshold 0.0

Continue training from a checkpoint::

    viame train -i training_data -c train_detector_netharn_cfrnn.conf \
        -s detector_trainer:ocv_windowed:trainer:netharn:seed_model=category_models/trained_detector.zip \
        --threshold 0.0

Netharn CFRNN Grid (Tiling Mode)
---------------------------------

A variant of the default CFRNN trainer that processes overlapping image tiles instead
of resizing the full image. This is critical for detecting small objects in
high-resolution imagery where targets would be lost during downscaling. The grid
trainer extracts overlapping chips at the native resolution and merges detections.

- Optimized for small objects (under ~30px) in dense scenes
- Configurable chip size and step (overlap between tiles)
- 20 or 40 epoch variants available for faster training
- Best when objects per frame exceed 20

Training::

    viame train -i training_data -c train_detector_netharn_cfrnn.grid_only.conf --threshold 0.0

RF-DETR
--------

Transformer-based DETR detector available in three sizes:

- **Nano** (384px) -- smallest model, fastest training, lower accuracy
- **Base** (560px) -- good balance of speed and accuracy
- **Large** (728px) -- highest accuracy, most compute intensive

Key training parameters: batch size 4, gradient accumulation 4 steps, EMA enabled.
RF-DETR's transformer attention mechanism makes it particularly effective in dense
scenes with overlapping objects and multi-scale variation. It handles occlusion well
compared to anchor-based detectors.

Training (base model)::

    viame train -i training_data -c train_detector_rf_detr_b_560.conf --threshold 0.0

For 16-bit imagery (e.g. thermal/IR), use the ``--normalize-16bit`` flag::

    viame train --input-list images.txt --input-truth annotations.csv \
        --labels labels.txt -c train_detector_rf_detr_b_560.conf \
        --normalize-16bit --threshold 0.0

MIT-YOLO v9
-------------

Modern YOLO variant using the YOLOv9-c architecture at 640x640 resolution. Offers
fast training and inference with competitive accuracy. A good choice when training
data is moderate (200+ annotations) and fast inference is important.

Training::

    viame train -i training_data -c train_detector_mit_yolo_v9_c_640.conf --threshold 0.0

Darknet YOLO
--------------

Mature YOLO implementation supporting YOLOv4 and YOLOv7 at various resolutions
(512--832px). Well-tested in production. Requires the Darknet YOLO add-on.

Training::

    viame train -i training_data -c train_detector_darknet_yolo_640.conf --threshold 0.0

Detectron2 Faster R-CNN
------------------------

Facebook's Detectron2 framework with ResNet-50 + FPN backbone at 800px. Uses
COCO-pretrained weights as initialization.

Training::

    viame train -i training_data -c train_detector_detectron2_frcnn.conf --threshold 0.0

LitDet Faster R-CNN / SSD
---------------------------

PyTorch Lightning-based implementations with built-in TensorBoard logging.

- **Faster R-CNN**: ResNet-50 + FPN at 640px, fine-tuned from COCO weights
- **SSD**: VGG-16 at 300px, lightweight and fast

Training::

    viame train -i training_data -c train_detector_litdet_frcnn.conf --threshold 0.0
    viame train -i training_data -c train_detector_litdet_ssd.conf --threshold 0.0

Netharn Mask R-CNN
-------------------

Instance segmentation variant that produces both bounding boxes and pixel-level masks.
Uses ResNet-50 backbone at 720px. Best for datasets with polygon annotations where
segmentation output is needed.

Training::

    viame train -i training_data -c train_detector_netharn_mask_rcnn_720.conf --threshold 0.0

MMDetection Cascade R-CNN
--------------------------

The MMDetection trainer provides access to the OpenMMLab detection framework,
primarily used for Cascade R-CNN with various backbones including ConvNeXt. Supports
distributed training across multiple GPUs via PyTorch DDP, Slurm, or MPI.

- Cascade R-CNN architecture with configurable backbones
- EQLv2 loss support for imbalanced class distributions
- Input resolution: up to 1333px
- Distributed multi-GPU training support

Training::

    viame train -i training_data -c train_detector_mmdet_cfrnn.conf --threshold 0.0

SVM Classifier
---------------

Classical SVM classifier that operates on top of proposal detections. Very fast to
train and runs on CPU. Best for small datasets or rapid prototyping. Two variants
are available:

- **Over fish detections**: classifies proposals from a fish-specific detector
- **Over generic detections**: classifies proposals from a generic proposal detector

Training::

    viame train -i training_data -c train_detector_svm_over_generic_detections.conf --threshold 0.0


****************************
Adaptive Detector Training
****************************

VIAME provides an adaptive training mode that automatically analyzes training data
statistics and selects the best trainer(s) for the given dataset. The adaptive trainer
considers:

- Annotation counts (total and per-class)
- Object sizes (mean, percentiles, distribution)
- Aspect ratios and scale variance
- Object density per frame
- Overlap and occlusion metrics
- Presence of mask/polygon annotations

Based on these statistics, the adaptive trainer selects up to 3 frameworks from:
SVM (50+ annotations), MIT-YOLO (200+), Netharn Grid (300+), LitDet FRCNN (300+),
RF-DETR (400+), and Netharn CFRNN (500+). Each trainer has hard requirements
(minimum count, minimum area, etc.) and soft preferences that are scored against
the data profile.

Training::

    viame train -i training_data -c train_detector_adaptive.conf --threshold 0.0

The adaptive trainer outputs a ``training_data_statistics.json`` file with the
computed dataset statistics.


**********************
Choosing a Detector
**********************

When deciding which detector to train, consider your dataset size, the characteristics
of the objects you want to detect, and your compute constraints. The adaptive trainer
uses the criteria below internally to auto-select trainers, but understanding them
can help you make a manual choice.

**By dataset size:**

- **Small (50--200 annotations per class):** Use the **SVM** classifier or the
  **adaptive** trainer. SVM requires very little data and trains on CPU.
- **Medium (200--500 per class):** **MIT-YOLO v9** or **Darknet YOLO** train
  efficiently with moderate data. **LitDet FRCNN** is also viable at 300+.
- **Large (500+ per class):** **Netharn CFRNN** (default) or **RF-DETR** are
  recommended for highest accuracy.

**By object characteristics:**

- **Small objects in large images:** Use **Netharn CFRNN Grid** (tiling mode), which
  processes overlapping image chips to better detect small targets. Best when objects
  are under ~30px and there are many per frame.
- **Dense scenes with occlusion (20+ objects per frame):** **RF-DETR** excels here --
  its transformer attention handles overlapping objects well. **Netharn CFRNN Grid**
  is also effective for dense small-object scenes.
- **Large or tall objects in sparse scenes:** **LitDet Faster R-CNN** is optimized for
  large objects (900+ pixel area) with tall aspect ratios and low object density.
- **Multi-scale objects (large size variation):** **MIT-YOLO v9** and **RF-DETR** both
  handle significant scale variation well.
- **Mixed aspect ratios:** **Netharn CFRNN** and **RF-DETR** are robust to varying
  aspect ratios.

**By compute constraints:**

- **Real-time inference needed:** **MIT-YOLO v9**, **LitDet SSD**, or **Darknet YOLO**
  offer the fastest inference.
- **No GPU available:** **SVM** is the only trainer that runs entirely on CPU.
- **Distributed multi-GPU training:** **MMDetection** supports Slurm and MPI-based
  distributed training.

**By output type:**

- **Instance segmentation (masks) needed:** **Netharn Mask R-CNN** produces pixel-level
  masks alongside bounding boxes.
- **Novelty detection (unknown classes):** Use **ReMax DINO** or **ReMax ConvNeXt**
  from the learn add-on to flag out-of-distribution objects.

**Unsure what to use:**
Run the **adaptive** trainer -- it will analyze your data statistics (object sizes,
density, aspect ratios, scale variance, overlap) and select the best option(s)
automatically, training up to 3 models sequentially.


***************
Example Scripts
***************

Training Scripts
-----------------

``train_default.sh`` / ``.bat``
    Train a detector using the default configuration (currently Netharn CFRNN).

``train_netharn_cfrnn.sh`` / ``.bat``
    Train a Netharn Cascade Faster R-CNN detector.

``train_rf_detr_b_560.sh`` / ``.bat``
    Train an RF-DETR base model at 560px resolution.

``train_rf_detr_n_384.sh`` / ``.bat``
    Train an RF-DETR nano model at 384px resolution.

``train_rf_detr_l_728.sh`` / ``.bat``
    Train an RF-DETR large model at 728px resolution.

``train_rf_detr_b_560_16bit.sh`` / ``.bat``
    Train an RF-DETR base model on 16-bit (thermal/IR) imagery.

``train_mit_yolo_v9_c_640.sh`` / ``.bat``
    Train a MIT-YOLO v9-c detector at 640px resolution.

``train_darknet_yolo.sh`` / ``.bat``
    Train a Darknet YOLO detector (requires Darknet add-on).

``train_detectron2_frcnn.sh`` / ``.bat``
    Train a Detectron2 Faster R-CNN detector.

``train_litdet_frcnn.sh`` / ``.bat``
    Train a LitDet Faster R-CNN detector.

``train_litdet_ssd.sh`` / ``.bat``
    Train a LitDet SSD detector.

``train_netharn_mask_rcnn.sh`` / ``.bat``
    Train a Netharn Mask R-CNN instance segmentation detector.

``train_svm_over_fish_dets.sh`` / ``.bat``
    Train an SVM classifier over fish detection proposals.

``train_svm_over_generic_dets.sh`` / ``.bat``
    Train an SVM classifier over generic detection proposals.

``continue_training_cfrnn.sh`` / ``.bat``
    Continue training a Netharn CFRNN model from an existing checkpoint.

Inference Scripts
------------------

``run_trained_model.sh`` / ``.bat``
    Run a trained detector model on new imagery. Uses ``detector_project_folder.pipe``
    which loads the model from the ``category_models`` directory produced by training.
    Supports multi-GPU processing via the ``TOTAL_GPU_COUNT`` and ``PIPES_PER_GPU``
    options.
