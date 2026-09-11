
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

The label file controls which categories are trained, allows synonyms for the
same category, and supports class hierarchies. ``viame train --labels FILE``
accepts ``.txt``, ``.csv``, and ``.json`` files. Without ``--labels``, training
looks for ``labels.txt``, ``labels.csv``, then ``labels.json`` in the input
directory, using the first file found.

**TXT synonyms and spaces:** Multiple names on the same line are treated as the
same output class. The first name becomes the output label. Put names containing
spaces in double or single quotes::

    "sport glove" "athletic glove"
    glove

This trains two output classes: ``sport glove`` (also matching ``athletic glove``
annotations) and ``glove``. Unquoted spaces still separate synonyms, so
``sport glove`` without quotes means the category ``sport`` with synonym ``glove``.
Annotation names themselves need no changes. ``#`` starts a comment outside quotes.
Inside quotes, escape a quote with a backslash or double it.

**Filtering:** Categories and synonyms omitted from the label file are excluded
from training. If no label file is supplied or discovered, training can use all
unique labels from the groundtruth.

**Hierarchies:** Parent-child relationships are separate from synonyms. A synonym
maps annotations onto the canonical output class; a parent remains a distinct
class, with its relationship available to trainers supporting hierarchical
classification. Parents may be declared after their children::

    "sport glove" "athletic glove" :parent="sport equipment"
    glove
    "sport equipment" gear

More than one ``:parent=`` field may be specified for a category.

**CSV:** Use one row per category, with the canonical name in the first field,
followed by synonyms and optional ``:parent=`` fields. There is no header row.
Spaces inside fields do not split names. The equivalent CSV file is::

    sport glove,athletic glove,:parent=sport equipment
    glove
    sport equipment,gear

Use CSV double quotes around fields containing commas or quotes; double an
embedded quote. Leading and trailing whitespace outside quoted fields is ignored.

**JSON:** Use a ``categories`` array with ``name`` and optional ``synonyms``,
``id``, and hierarchy fields. This follows DIVE's COCO hierarchy convention:
``supercategory`` names a parent, while ``parents`` supports multiple parents
when no nonempty ``supercategory`` is supplied::

    {
      "categories": [
        {
          "name": "sport glove",
          "synonyms": ["athletic glove"],
          "supercategory": "sport equipment"
        },
        {"name": "glove"},
        {"name": "sport equipment", "synonyms": ["gear"]}
      ]
    }

DIVE's ``typeHierarchy`` child-to-parent mapping can also supply the hierarchy
(with or without a ``categories`` array)::

    {
      "categories": [
        {"name": "sport glove", "synonyms": ["athletic glove"]},
        "glove",
        {"name": "sport equipment", "synonyms": ["gear"]}
      ],
      "typeHierarchy": {"sport glove": "sport equipment"}
    }

The ``synonyms`` array adds training aliases to the DIVE-compatible category
records. JSON also accepts a bare category array, including a simple list such as
``["sport glove", "glove"]``. Explicit integer IDs determine category ordering;
otherwise categories receive IDs in file order. Hierarchy-only nodes referenced
by JSON are added after the listed categories, as DIVE permits parents without
category records. Duplicate category/synonym names and cyclic hierarchies are
rejected.


***********************
Monitoring Training Runs
***********************

Long training runs can report their progress by email. Adding ``--monitor-email``
to ``viame train`` starts a background monitor next to the run::

    viame train -i training_data -c train_detector_default.conf --threshold 0.0 \
        --monitor-email you@example.com

The monitor sends a test message when it starts, then reports on detected errors
or deadlocks, training stage changes, validation statistics every few epochs, a
periodic heartbeat during long stages, and finally whether the run finished
normally, ended with an error, or stopped unexpectedly. A copy of the training
output is written to ``train.log`` in the output directory (``category_models``
by default), alongside a ``monitor_status.log`` trail of every check.

Mail is sent through an SMTP server given by ``--monitor-smtp host[:port]`` (with
``--monitor-smtp-user`` and the ``VIAME_SMTP_PASSWORD`` environment variable when
the server needs a login), or through a local ``sendmail`` on Linux when no server
is configured. The server, user and sender address can also be set once through
the ``VIAME_SMTP_SERVER``, ``VIAME_SMTP_USER`` and ``VIAME_SMTP_FROM`` environment
variables, which is how the DIVE desktop and web training dialogs deliver mail
when an address is entered there. With neither an SMTP server nor ``sendmail``
available, the reports are still written to the status trail. ``--monitor-poll``
changes how often the run is checked (every 20 minutes by default).

The same monitor can follow a run that was started some other way, for example
a slurm job or a run under ``nohup``, using the ``viame monitor`` tool directly::

    viame monitor start -o category_models -l train.log --job-id 12345 \
        --email you@example.com
    viame monitor start -o category_models -l train.log --pid 4242 \
        --email you@example.com --smtp-server smtp.example.com:587
    viame monitor status category_models
    viame monitor stop category_models

The run is considered alive while its slurm job is queued, its process exists,
or, when neither is given, its log keeps changing (``--stale-minutes``). RF-DETR
runs are recognised by the ``metrics.csv`` they write and report validation mAP,
precision and recall; other trainers report the current stage and latest epoch
line parsed from the log. Run ``viame monitor start --help`` for the full list of
options, including the report interval and the pattern that marks a finished run.


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

Netharn RF-DETR masks, keypoints, and native checkpoints
------------------------------------------------------

Use ``train_detector_netharn_rf_detr_l_seg_kp_1728.conf`` for boxes, masks,
and head/tail keypoints. It uses 1728x960 network inputs and inherits the
netharn box recipe's data split. The recipe disables outer OpenCV scaling
(which does not transform
keypoints) and uses a separate cache; netharn samples windows and transforms
all annotation types together. When enabling keypoints in another config,
also set the outer trainer's ``mode=disabled``.
It uses the native RFDETRSegLarge architecture,
including its 12-pixel patches, five decoder layers, and 200 queries.
The netharn wrapper retains its single query group and netharn optimizer,
scheduler, augmentation, and checkpointing; this is not an identical native
RF-DETR training schedule.

To fine-tune a native RF-DETR segmentation checkpoint with netharn::

    viame train -i training_data \
        -c train_detector_netharn_rf_detr_l_seg_kp_1728.conf \
        -s detector_trainer:ocv_windowed:trainer:netharn:native_seed_model=/absolute/path/to/model.pth \
        --threshold 0.0

``native_seed_model`` accepts a native RF-DETR checkpoint (including exported
``.pth``/``.pt`` weights or a Lightning ``.ckpt``). It uses RF-DETR's weight
loader to adapt query embeddings, positional embeddings, and class-head sizes.
It starts a new netharn optimization run; optimizer, epoch, scheduler, and EMA
state are not resumed. Use a fresh training directory/identifier to avoid
netharn automatically resuming an existing run. ``seed_model`` remains the
option for netharn seeds; the two seed options are mutually exclusive, and a
missing native seed path is an error.

Match ``arch`` and ``segmentation_head`` to the seed architecture. Adding a
keypoint head to a segmentation seed is supported: the new head starts fresh.
A detection Large checkpoint and a SegLarge checkpoint have different backbones
and decoder shapes and are not interchangeable. Preserve class order and,
for an existing keypoint head, keypoint slot order. The loader adapts class
counts but does not remap class names; a class-name/order mismatch emits a warning.

The following options can also be applied to the existing netharn RF-DETR recipe
under ``detector_trainer:ocv_windowed:trainer:netharn``:

- ``segmentation_head=True`` selects a segmentation architecture and mask losses.
- ``keypoints=True`` enables keypoint coordinate and visibility losses, with or
  without segmentation.
- ``keypoint_names=head,tail`` defines ordered, case-insensitive point names.
- ``native_seed_model=/path/to/checkpoint.pth`` initializes from a native run.

Mask training requires a polygon or mask for every non-ignored object. Keypoints
use the CSV ``(kp) head x y`` / ``(kp) tail x y`` attributes (or equivalent COCO
annotations). Missing or cropped-out points get visibility zero. A run with no
visible training points matching the configured names is rejected. Regenerate
an existing augmentation cache if it predates the mask/keypoint annotations.
Use the VIAME build's COCO writer that preserves polygons and named keypoints.
Netharn's existing validation metrics remain box-based; this recipe does not add
COCO mask AP or keypoint OKS evaluation.


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

Training::

    viame train -i training_data -c train_detector_rf_detr_default.conf --threshold 0.0

For 16-bit imagery (e.g. thermal/IR), use the ``--normalize-16bit`` flag::

    viame train --input-list images.txt --input-truth annotations.csv \
        --labels labels.txt -c train_detector_rf_detr_default.conf \
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
Default (Automatic) Training
****************************

``train_detector_default.conf`` analyzes the training data and trains the single
detector that best fits it. The trainer considers:

- Annotation counts (total and per-class)
- Object sizes (mean, percentiles, distribution)
- Aspect ratios, scale variance, density, overlap
- Presence of mask/polygon and keypoint annotations
- Source image resolution
- How well the stock generic detector already covers the data

The candidates, in the order they are considered, are an SVM over the stock
detector's proposals (tiny datasets the stock detector already finds objects
in, measured by running it), a keypoint RF-DETR, a segmentation RF-DETR, a
1728x960 RF-DETR for high-resolution imagery, and a 720px RF-DETR otherwise.

Training::

    viame train -i training_data -c train_detector_default.conf --threshold 0.0

When the groundtruth contains tracks, a bytetrack tracker is also trained over
the detector's output and written out as ``tracker.pipe``.

The trainer outputs a ``training_data_statistics.json`` file with the computed
dataset statistics, including which branch was selected and why.


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

``train_rf_detr_default.sh`` / ``.bat``
    Train an RF-DETR detector with the default recipe, which auto-selects
    boxes or segmentation from the annotations.

``train_rf_detr_n_384.sh`` / ``.bat``
    Train an RF-DETR nano model at 384px resolution.

``train_rf_detr_l_728.sh`` / ``.bat``
    Train an RF-DETR large model at 728px resolution.

``train_rf_detr_default_16bit.sh`` / ``.bat``
    Train an RF-DETR detector on 16-bit (thermal/IR) imagery.

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

``train_svm_over_fish_dets.sh`` / ``.bat``
    Train an SVM classifier over fish detection proposals.

``train_svm_over_generic_dets.sh`` / ``.bat``
    Train an SVM classifier over generic detection proposals.

``continue_training_cfrnn.sh`` / ``.bat``
    Continue training a Netharn CFRNN model from an existing checkpoint.

Inference Scripts
------------------

``run_trained_model.sh`` / ``.bat``
    Run a trained detector model on new imagery. Uses ``category_models/detector.pipe``
    which loads the model from the ``category_models`` directory produced by training.
    Supports multi-GPU processing via the ``TOTAL_GPU_COUNT`` and ``PIPES_PER_GPU``
    options.
