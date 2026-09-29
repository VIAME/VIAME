
=====================
Monocular Measurement
=====================

********
Overview
********

This document corresponds to the `monocular
measurement <https://github.com/VIAME/VIAME/tree/main/examples/monocular_measurement>`__
example folder within a VIAME desktop installation.

Monocular measurement computes the real-world size of objects from a single camera. One
image holds no depth information, so the scale has to come from somewhere else: usually
the camera's height above the scene. This suits downward-looking survey cameras over a
roughly flat surface, such as a towed or autonomous vehicle imaging the seafloor.

.. list-table::
   :header-rows: 1

   * - 
     - Monocular
     - Stereo
   * - Cameras
     - One
     - A calibrated pair
   * - Scale comes from
     - Altitude above the scene, or a known ground sample distance
     - Triangulation between the two views
   * - Works for
     - Objects lying on a roughly flat surface at a known distance
     - Objects at any depth, including free-swimming animals
   * - Needs per frame
     - Altitude and orientation of the camera
     - Nothing beyond the two images
   * - What is measured
     - The width of the detection box
     - The distance between head and tail keypoints

For free-swimming animals, or wherever the distance to the object is unknown, use
`stereo
measurement <https://viame.github.io/VIAME/sections/stereo_measurement.html>`__.

************************
How Lengths Are Computed
************************

Lengths are assigned by the ``refine_measurements`` process, which sits after the
detector in a pipeline. It converts the width of each detection box from pixels to real
units using a ground sample distance (GSD), the real-world size covered by one pixel.
The GSD can come from three places:

.. list-table::
   :header-rows: 1

   * - Source
     - How it is used
   * - Camera metadata
     - The altitude and orientation of the camera, with its intrinsics, place the left
       and right edges of each box on the ground plane. The length is the distance
       between them.
   * - A supplied GSD
     - A value connected to the ``gsd`` input of the process is applied to every
       detection in the frame.
   * - Detections that already have a length
     - When some detections in a frame carry a length, the GSD is estimated from them
       and applied to the rest. The estimate is kept for following frames that have
       none.

The length is written to each detection as a ``length`` attribute. With metadata, an
altitude given in meters produces lengths in millimeters.

************
Requirements
************

- **Camera intrinsics**: the focal length and image center of the camera, found by
  calibrating it once.
- **Altitude and orientation for every frame**: read from metadata recorded with the
  imagery. HabCam images carry theirs inside the image file, which the
  ``read_habcam_metadata`` process reads. Imagery from other systems needs its own
  reader, or a GSD supplied another way.

***************************
Calibrating a Single Camera
***************************

``utility_calibrate_single_camera.pipe`` computes the intrinsics and distortion
coefficients of one camera from images or video of a calibration target. It first looks
for a checkerboard and, failing that, a grid of bright dots.

.. code-block:: bash

   ./calibrate_single_camera.sh calibration_images.txt 25.0

or, calling the pipeline directly:

.. code-block:: bash

   source /path/to/VIAME/install/setup_viame.sh
   viame configs/pipelines/utility_calibrate_single_camera.pipe \
     -s input:video_filename=calibration_images.txt \
     -s global:square_size=25.0

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Purpose
   * - ``global:square_size``
     - 80
     - Real-world size of one checkerboard square, or the dot spacing
   * - ``global:frame_count_threshold``
     - 50
     - Number of frames with a detected target to use
   * - ``global:output_json_file``
     - ``calibration.json``
     - Name of the file written

When run from DIVE, the pipeline prompts for the square size. The output file holds the
image size, the calibration error, the intrinsics ``fx``, ``fy``, ``cx`` and ``cy``, and
the distortion coefficients ``k1``, ``k2``, ``p1``, ``p2`` and ``k3``.

***********************
Measuring from Metadata
***********************

The HabCam add-on includes two pipelines that detect scallops in downward-looking
seafloor imagery and measure them from the metadata in each image:

.. list-table::
   :header-rows: 1

   * - Pipeline
     - Detects
   * - ``detector_habcam_measure_scallops_one_class_metadata.pipe``
     - Scallops as a single class
   * - ``detector_habcam_measure_scallops_four_class_metadata.pipe``
     - Scallops in four classes

With the add-on installed, run the first on the example imagery with:

.. code-block:: bash

   ./measure_via_habcam_metadata.sh

The results are written to ``computed_detections_left.csv``. The measurement part of the
pipeline is two processes, one reading the metadata and one applying it:

::

   process measurer_metadata_parser
     :: read_habcam_metadata

   process measurer_pass1_left
     :: refine_measurements
     :recompute_all                               true
     :min_valid                                   10.0
     :max_valid                                   230.0
     :intrinsics               2518.80862 0 680 0 2518.80862 512 0 0 1

To measure with a different camera, replace the ``intrinsics`` with the values from its
calibration, in the order ``fx 0 cx 0 fy cy 0 0 1``.

********
Settings
********

These are the settings of ``refine_measurements``:

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Purpose
   * - ``intrinsics``
     - identity
     - Camera matrix as nine numbers, row by row
   * - ``recompute_all``
     - false
     - Recomputes the length of every detection, including those that already have one
   * - ``min_valid``, ``max_valid``
     - off
     - Lengths outside this range are not written. Set these to the plausible size range
       of the target.
   * - ``border_factor``
     - 0
     - Detections within this many pixels of the image edge are treated as unreliable
       when estimating the GSD, since the object may be cut off
   * - ``percentile``
     - 0.45
     - Which of the GSD estimates in a frame to use, as a fraction from smallest to
       largest
   * - ``output_conf_level``
     - false
     - Adds a ``length_conf`` note to each detection: ``none``, ``very_low``, ``low``,
       ``medium`` or ``high``
   * - ``output_multiple``
     - false
     - Keeps existing length notes alongside the new one

***********
Limitations
***********

- The length is the width of the detection box. It matches the size of the object only
  when the object is roughly round, like a scallop, or lies along the horizontal axis of
  the image.
- The surface is assumed to be flat, with every object resting on it. Objects above the
  surface appear larger than they are.
- Accuracy depends directly on the altitude. An error of ten percent in altitude gives
  an error of ten percent in length.
- Objects cut off by the edge of the image are measured too short.


.. dive-crosslink

******************
DIVE Documentation
******************

* `DIVE frame metadata <https://viame.github.io/VIAME/sections/dive/Frame-Metadata.html>`__ covers attaching per-frame metadata, such as altitude, to a dataset
* `DIVE pipelines and training <https://viame.github.io/VIAME/sections/dive/Pipeline-Documentation.html>`__ covers running a pipeline on a dataset


********************
Code and Build Flags
********************

Flags to enable when building VIAME from source for this example:

* ``VIAME_ENABLE_ONNX``
* ``VIAME_ENABLE_OPENCV``
* ``VIAME_ENABLE_PYTHON``
* ``VIAME_ENABLE_VXL``

Add-ons providing the pipelines or models used: ``habcam``.

Pipeline and configuration files:

* configs/pipelines/utility_calibrate_single_camera.pipe
* configs/add-ons/habcam/detector_habcam_measure_scallops_one_class_metadata.pipe
* configs/add-ons/habcam/detector_habcam_measure_scallops_four_class_metadata.pipe

Source code:

* plugins/core/accumulate_object_tracks_process.cxx
* plugins/core/merge_detections_nms_fusion.py
* plugins/core/merge_detections_simple.py
* plugins/core/read_habcam_metadata_process.cxx
* plugins/core/refine_detections_nms.cxx
* plugins/core/refine_measurements_process.cxx
* plugins/onnx/onnx_detector.py
* plugins/onnx/onnx_refiner.py
* plugins/opencv/calibrate_single_camera.cxx
* plugins/opencv/calibrate_single_camera_process.cxx
* plugins/opencv/debayer_filter.cxx
* plugins/opencv/detect_calibration_targets.cxx
* plugins/opencv/enhance_images.cxx
* plugins/opencv/split_image_habcam.cxx
* plugins/opencv/windowed_detector.cxx
