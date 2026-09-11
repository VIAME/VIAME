
===========================
Size Measurement Examples
===========================

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/fish_measurement_example.png
   :width: 70%
   :align: center

Running the Demo
================

This section corresponds to the `size measurement
<https://github.com/VIAME/VIAME/tree/master/examples/size_measurement>`_ example folder
within a VIAME desktop installation. This folder contains examples and pipelines for
computing real-world size measurements of objects (e.g., fish length) from imagery.

VIAME supports two primary approaches to size measurement:

**Stereo-Based Measurement**
  Uses a calibrated pair of stereo cameras to triangulate 3D positions and compute
  real-world distances. This is the most accurate approach and works at varying depths
  and distances. It requires a stereo calibration file containing camera intrinsic and
  extrinsic parameters (see the `Calibration Pipelines`_ section). Stereo measurement
  pipelines detect or accept annotated objects in both left and right camera views,
  establish correspondences between them, and triangulate keypoints (e.g., head and
  tail) to compute lengths. The demo data and scripts in this folder use stereo-based
  measurement.

**Metadata-Based Measurement**
  Uses camera metadata -- such as altitude above the seafloor, camera intrinsics, and
  orientation angles (yaw, pitch, roll) -- to compute a ground sample distance (GSD)
  and convert pixel measurements to real-world units. This approach requires only a
  single camera but depends on accurate metadata being available for each frame. It is
  well-suited for downward-looking survey cameras at a known or measured altitude, such
  as the HabCam benthic survey system. Examples of metadata-based measurement can be
  found in the HabCam add-on (e.g., ``detector_habcam_measure_scallops_one_class_metadata.pipe``),
  which reads altitude and orientation from image metadata and applies a GSD calculation
  using the camera intrinsics matrix.

Run CMake to automatically download the demo data into this example folder.
Alternatively you can download the demo data `directly`_.

.. _directly: https://data.kitware.com/#item/5a8607858d777f068578345e`

Setup:
------

Make sure you build VIAME with `VIAME_ENABLE_PYTHON=True` and
`VIAME_ENABLE_OPENCV=True`.

For simplicity this tutorial will assume that the VIAME source directory is
`[viame-source]` and the build directory is `[viame-build]`. Please modify
these as needeed to match your system setup. We also assume that you have built
VIAME.

Additionally this example requires an extra python dependency to be installed.
On Linux or Windows, 'pip install ubelt'.


Available Scripts
-----------------

This example folder contains several scripts for different measurement workflows.
Each script is available in both Linux (.sh) and Windows (.bat) versions.

**calibrate_cameras**
  Runs the camera calibration tool to compute stereo camera calibration parameters
  from a video or set of images containing a chessboard calibration pattern. Outputs
  a JSON file (calibration_matrices.json) compatible with the VIAME measurement pipelines.
  Usage: ``./calibrate_cameras.sh <video_file_or_image_glob>``

**measure_via_gmm_oriented_boxes**
  Runs the automatic GMM (Gaussian Mixture Model) motion-based measurement pipeline.
  This pipeline uses background subtraction to detect moving objects and computes
  oriented bounding boxes for measurement. Best suited for scenarios with stationary
  cameras and moving fish.

**measure_via_default_fish**
  Runs the default automatic fish measurement pipeline using a neural network-based
  fish detector. This pipeline detects fish using a trained model and performs
  stereo measurement on the detected objects.

**measure_over_manual_annotations**
  Runs measurement on user-provided annotations. Use this when you have manually
  annotated fish locations and want to compute measurements from those annotations
  rather than using automatic detection.

**gmm_standalone_tool**
  A standalone tool for GMM-based measurement that includes disparity computation.
  This script provides more control over the measurement process and can be used
  for debugging or custom workflows.

**compute_depth_maps**
  Computes stereo disparity/depth maps from the calibrated stereo camera imagery.
  Useful for visualizing the depth information or for custom processing workflows.


Running via the pipeline runner
-------------------------------

To run the process using the sprokit C++ pipeline we use the the pipeline
runner:

::

    # First move to the example directory
    cd [viame-build]/install/examples/size_measurement

    # The below script runs pipeline runner on the GMM motion-based measurement
    bash measure_via_gmm_oriented_boxes.sh


This example runs at about 4.0Hz, and takes 13.3 seconds to complete on a 2017
i7 2.8Ghz Dell laptop.


Running via installed opencv python module 
--------------------------------------------

The above pipeline can alternatively be run as a python script.

::

    # move to your VIAME build directory
    cd [viame-build]
    # Run the setup script to setup the proper paths and environment variables
    source install/setup_viame.sh

    # you may also want to set these environment variables
    # export KWIVER_DEFAULT_LOG_LEVEL=debug
    export KWIVER_DEFAULT_LOG_LEVEL=info
    export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes


You should be able to run the help command

:: 

    python -m viame.opencv.stereo_demo --help

The script can be run on the demodata via

::

    python -m viame.opencv.stereo_demo \
        --left=camtrawl_demodata/left --right=camtrawl_demodata/right \
        --cal=camtrawl_demodata/cal.npz \
        --out=out --draw -f


Running via the standalone script
---------------------------------

Alternatively you can run by specifying the path to opencv module (if you
have a python environment you should be able to run this without even building
VIAME)



::

    # First move to the example directory
    cd [viame-source]/examples/size_measurement

    # Run the stereo_demo module directly via the path
    python ../../plugins/opencv/stereo_demo.py \
        --left=camtrawl_demodata/left --right=camtrawl_demodata/right \
        --cal=camtrawl_demodata/cal.npz \
        --out=out --draw -f

Without the `--draw` flag the above example, this example runs at about 2.5Hz,
and takes 20 seconds to complete on a 2017 i7 2.8Ghz Dell laptop.

With `--draw` it takes significantly longer (it runs at 0.81 Hz and takes over
a minute to complete), but will output images like the one at the top of this
readme as well as a CSV file.

Note that the KWIVER C++ Sprokit pipline offers a significant speedup (4Hz vs
2.5Hz), although it currently does not have the ability to output the algorithm
visualization.

.. _Calibration Pipelines:

Calibration Pipelines
---------------------

VIAME provides several calibration pipelines for computing camera parameters from
images or video of a calibration target. The pipelines first attempt to detect a
checkerboard (chessboard) pattern, and if that fails, fall back to detecting a grid
of bright dots (circle grid). Detected corners or centers are accumulated across
frames and used to solve for the camera intrinsics, distortion coefficients, and
(for stereo) extrinsic parameters. The ``square_size`` parameter must be set to the
real-world size of a checkerboard square (or dot spacing) in your chosen unit (e.g.,
millimeters) -- this value determines the scale of all subsequent measurements. When
running from the DIVE interface, the pipeline will prompt for the checkerboard square
size in real units before running. The output calibration file can then be used by the
measurement pipelines.

.. image:: https://www.viametoolkit.org/wp-content/uploads/2026/04/Calibration-Query-User.jpg
   :width: 80%
   :align: center

*The DIVE calibration dialog prompts for the checkerboard square size before running
the calibration pipeline.*

|

.. image:: https://www.viametoolkit.org/wp-content/uploads/2026/04/Calibration-Show-Features-On-Success1.jpg
   :width: 80%
   :align: center

*Successful stereo calibration showing detected feature correspondences between left
and right camera views.*

|

**stereo_calibrate_cameras_default.pipe**
  Stereo camera calibration from separate left and right camera inputs. Detects
  chessboard corners in both views, accumulates correspondences across frames, and
  computes stereo calibration matrices. Outputs ``calibration_matrices.json``. This
  is the recommended pipeline for most stereo calibration tasks.

**stereo_calibrate_cameras_fast.pipe**
  A faster variant of the stereo calibration pipeline that uses fewer frames
  (threshold of 25 vs. the default). Use this when you have a large number of
  calibration frames and want quicker results at the cost of slightly reduced accuracy.

**utility_calibrate_single_camera.pipe**
  Monocular (single camera) calibration from images of a chessboard target. Computes
  intrinsic parameters and distortion coefficients for a single camera. Outputs
  ``calibration.json``. Useful when you only need to undistort imagery from one camera
  or as a preliminary step before stereo calibration.

**utility_calibrate_stitched_stereo_pair.pipe**
  Calibrates a stereo pair from a single video or image input where left and right
  frames are horizontally concatenated (stitched side-by-side). The pipeline splits
  each frame, detects chessboard corners in both halves, and computes stereo calibration.
  Outputs ``calibration_matrices.json``. Useful for cameras that record both views into
  a single file.

To run a calibration pipeline from the command line, for example::

  source /path/to/VIAME/install/setup_viame.sh
  kwiver runner configs/pipelines/utility_calibrate_single_camera.pipe \
    -s downsampler:input_file_name=calibration_images.txt \
    -s global:square_size=25.0

For stereo calibration with separate camera inputs::

  kwiver runner configs/pipelines/stereo_calibrate_cameras_default.pipe \
    -s input1:video_filename=cam1_images.txt \
    -s input2:video_filename=cam2_images.txt \
    -s global:square_size=25.0


Stereo Disparity and Depth Pipelines
-------------------------------------

.. image:: https://www.viametoolkit.org/wp-content/uploads/2026/04/Stereo-Epipolar-Search1.jpg
   :width: 80%
   :align: center

*Stereo epipolar matching: a point selected in the left camera (cyan) is matched to
its correspondence in the right camera (green/red) using epipolar geometry.*

|

VIAME includes several methods for computing stereo disparity and depth maps, which
are used internally by the measurement pipelines and can also be run standalone for
visualization or custom processing. Stereo measurement requires a calibration file
containing the camera intrinsic and extrinsic parameters. This calibration file can
either be computed within VIAME using one of the calibration pipelines described above,
or imported from an external source (e.g., OpenCV, MATLAB Camera Calibrator, or other
third-party calibration tools). Supported calibration file formats include JSON (as
output by the VIAME calibration pipelines), NPZ (numpy archive), MAT (MATLAB), and
``.CamCAL`` (SEAGIS) files. See the `Calibration File Format`_ section below for
details on the expected contents of each format.

**stereo_compute_rectified_disparity.pipe**
  Computes rectified stereo disparity maps using the SGBM (Semi-Global Block Matching)
  algorithm with WLS filtering. Requires a pre-computed camera calibration file.
  Useful for visualizing depth or feeding into custom measurement workflows.

**filter_stereo_depth_map.pipe**
  Filters and enhances stereo depth maps from horizontally concatenated stereo images.
  Applies CLAHE contrast enhancement and denoising before computing OCV stereo
  disparity. Outputs filtered depth map images.

**Foundation Stereo (add-on)**
  The Foundation Stereo add-on provides a deep learning-based stereo disparity model
  that produces higher quality depth estimates than traditional SGBM. It is available
  in three model sizes: ``vits`` (small, faster), ``vitb`` (base), and ``vitl`` (large,
  more accurate). The small variant is recommended for most use cases. Foundation Stereo
  is used by the ``stereo_measure_current_annots_fdn_stereo_s.pipe`` pipeline and can be
  enabled by installing the Foundation Stereo add-on.


Measurement from Annotations Pipelines
---------------------------------------

These pipelines compute stereo measurements from user-provided annotations (e.g.,
head/tail keypoints on fish). They read left and right camera track files, match
detections between cameras, and triangulate 3D positions to compute lengths.

.. image:: https://www.viametoolkit.org/wp-content/uploads/2026/04/Interactive-Stereo0.jpg
   :width: 80%
   :align: center

*Setting up interactive stereo measurement in DIVE with multi-camera settings.*

|

.. image:: https://www.viametoolkit.org/wp-content/uploads/2026/04/Interactive-Stereo1.jpg
   :width: 80%
   :align: center

*Creating stereo annotations interactively -- the user draws on one camera view.*

|

.. image:: https://www.viametoolkit.org/wp-content/uploads/2026/04/Interactive-Stereo2.jpg
   :width: 80%
   :align: center

*Stereo correspondences with epipolar geometry shown across left and right views.*

|

.. image:: https://www.viametoolkit.org/wp-content/uploads/2026/04/Stereo-Seamap-Short1.png
   :width: 80%
   :align: center

*Stereo measurement results in DIVE showing detected objects with computed lengths
displayed in the Track Details panel.*

|

**stereo_measure_current_annots_default.pipe**
  The default measurement-from-annotations pipeline. Uses Foundation Stereo (if the
  add-on is installed) for disparity estimation and ORB feature matching for stereo
  correspondence. Reads annotation files for both cameras, pairs detections, and
  outputs measured tracks to ``computed_tracks2.csv``.

**stereo_measure_current_annots_fdn_stereo_s.pipe (Foundation Stereo add-on)**
  Uses the Foundation Stereo deep learning model (small variant) for high-quality
  disparity estimation. Recommended when the Foundation Stereo add-on is installed,
  as it generally produces more accurate measurements than traditional methods.

**stereo_measure_current_annots_ncc_dino.pipe (DINO add-on)**
  Uses DINO visual features for template matching between left and right camera views,
  with NCC (Normalized Cross-Correlation) as a secondary matching stage. Can produce
  better correspondence in challenging cases where ORB features are insufficient.

**stereo_measure_current_annots_seagis.pipe (SEAGIS add-on)**
  Uses the SEAGIS StereoLibLX library with ``.CamCAL`` calibration files for stereo
  measurement. Supports epipolar template matching as a fallback when only one camera
  has annotated keypoints. Use this pipeline if your calibration data is in the SEAGIS
  format.


Fully Automatic Measurement Pipelines
--------------------------------------

These pipelines perform end-to-end automatic detection and measurement without
requiring any manual annotations.

**stereo_track_and_measure_default_fish.pipe**
  Fully automatic fish detection, tracking and measurement pipeline. Runs the
  default fish detector (boxes, masks and head/tail keypoints) and a tracker on
  both stereo cameras, pairs left and right tracks, averages each pair's
  classification, and triangulates the head/tail keypoints into lengths that
  are aggregated per track. Outputs measured tracks to ``computed_tracks1.csv``
  and ``computed_tracks2.csv``.

**stereo_detect_and_measure_default_fish.pipe**
  Same detector without a tracker: left and right detections are paired and
  measured independently on every frame, each pair sharing one track ID and an
  averaged classification.

**stereo_detect_and_measure_gmm_motion.pipe**
  Automatic measurement pipeline using GMM (Gaussian Mixture Model) background
  subtraction to detect moving objects. Computes oriented bounding boxes for
  measurement. Best suited for stationary camera setups where fish swim through
  the field of view.


.. _Calibration File Format:

Calibration File Format
-----------------------

For the npz file format the root object should be a python dict with the
following keys and values:

|
|    R: extrinsic rotation matrix
|    T: extrinsic translation
|    cameraMatrixL: dict of intrinsict parameters for the left camera
|        fc: focal length
|        cc: principle point
|        alpha_c: skew
|    cameraMatrixR: dict of intrinsict parameters for the right camera
|        fc: focal length
|        cc: principle point
|        alpha_c: skew
|    distCoeffsL: distortion coefficients for the left camera
|    distCoeffsR: distortion coefficients for the right camera
|

For the mat file, format the root structure should be a dict with the key
`Cal` whose value is a dict with the following items:

|
|    om: extrinsic rotation vector (note rotation matrix is rodrigues(om))
|    T: extrinsic translation
|    fc_left: focal length of the left camera
|    cc_left: principle point
|    alpha_c_left: skew
|    kc_left: distortion coefficients for the left camera
|    fc_right: focal length of the right camera
|    cc_right: principle point
|    alpha_c_right: skew
|    kc_right: distortion coefficients for the right camera
|

Curved fish measurement (opt-in)
-------------------------------

The Python stereo service supports a separate ``measure_curve`` command. It
uses dense disparity from FoundationStereo, Fast FoundationStereo, or another
configured disparity backend. Existing line measurements and pipelines keep
working as before. This is a service/API and offline utility; the annotation UI
does not automatically generate or display these curves.

Inputs must already be rectified, with masks, curves, disparity and calibration
all on the same full-resolution image grid. Use the intrinsics produced by
rectification, not the original distorted camera intrinsics. The supported
rectified model has shared ``fx``, ``fy`` and ``cy``, potentially different
horizontal principal points, and a positive horizontal baseline. Lengths use
the baseline's units. Underwater measurements require calibration appropriate
to the camera housing and medium.

After enabling the dense stereo service and waiting for ``disparity_ready``,
send a request such as::

  {
    "command": "measure_curve",
    "left_image_path": "/data/left.png",
    "right_image_path": "/data/right.png",
    "rectified_calibration": {
      "rectified": true,
      "fx": 1000, "fy": 1000,
      "cx_left": 640, "cx_right": 640, "cy": 360,
      "baseline": 0.12
    },
    "left_curve": [[300, 300], [340, 280], [390, 285], [440, 320]],
    "right_curve": [[260, 300], [300, 280], [350, 285], [400, 320]],
    "options": {
      "mode": "bidirectional",
      "samples": 32,
      "smoothing": 0.5,
      "consistency_px": 1.5,
      "centerline_tolerance_px": 5.0,
      "max_length_disagreement": 0.1,
      "max_depth_ratio": 2.0
    }
  }

Frame paths must identify the current frame; for video, also supply its current
``frame_time``. Requests before disparity is ready fail and can be retried.
The command is also routed by the unified interactive service.

The modes are:

* ``left`` (default): resample the left centerline and transfer each point using
  dense disparity. A supplied right centerline constrains the transfer by
  proximity; equal fractions of two independently drawn curves are not assumed
  to correspond.
* ``bidirectional``: additionally compute independent right-reference disparity
  by swapping and horizontally flipping the stereo images, then unflipping the
  result. Both centerlines are sampled independently, with a round-trip disparity
  check at every sample. Average the two reconstructed lengths only if their
  relative disagreement is within the configured threshold. This adds one model
  inference per request; the reverse result is not cached. The backend must
  support normal horizontal rectified stereo and output disparity, not depth.

Instead of ``left_curve`` or ``right_curve``, supply ``left_mask_path`` or
``right_mask_path`` and the corresponding ``left_endpoints`` or
``right_endpoints`` as ``[[head_x, head_y], [tail_x, tail_y]]``. Masks are binary
full-image rasters (nonzero means fish), for example exported SAM2 masks;
bounding-box-local masks must first be placed on the full-image grid. Endpoint
anchors can come from SLEAP. Mask extraction requires scikit-image, included in
PyTorch builds. The shortest connected skeleton path between anchor locations
removes side branches such as fins. Disconnected anchor paths fail. Masks can
also accompany explicit curves to check that every matched sample stays inside
the fish.

``smoothing`` is the 2D spline fit tolerance in pixels; endpoints are retained.
The reported length is the sum of 3D segment lengths between sampled points,
not a spline integral. More samples are not necessarily more accurate: depth
noise can inflate length, while too few samples underestimate sharp bends.
Defaults are starting points for validation, not calibrated uncertainty bounds.
Silhouette centerlines and visible surface disparity approximate the anatomical
midline, especially when fish roll or bend out of the image plane. Bidirectional
agreement alone does not establish anatomical accuracy. Define the tail landmark
consistently (base, fork or tip), and validate against known lengths and poses.

Responses contain ``curved_length``, ``straight_length``, ``curvature_ratio``,
per-direction 3D points and transferred image points, and round-trip/length
agreement diagnostics. Invalid or out-of-image disparity, mask violations,
inconsistent matches or excessive depth range produce ``success: false`` with
no top-level curved length. Missing samples are never bridged or replaced by a
straight-line result. Existing detection length fields are not overwritten.

Offline processing of saved disparity maps is also available::

  python -m viame.core.curved_measurement request.json --output measurement.json

Use the same JSON fields as above, adding ``left_disparity_path`` and, for
bidirectional mode, ``right_disparity_path`` to NumPy ``.npy`` maps. Offline maps
must already be computed; this utility does not load a model. Set
``disparity_scale`` to 256 for VIAME uint16 disparity, or 1 (default) for floating
point pixel disparity. Both maps represent the positive quantity
``x_left - x_right``, indexed on their respective source image grids; a negated
right disparity map must be converted first. All paths are resolved against the
working directory. Independent right disparity also enables round-trip checking
in offline ``left`` mode.

Editable DIVE centerlines
~~~~~~~~~~~~~~~~~~~~~~~~

The matching DIVE ``feature/curved-headtail-lines`` branch extends the existing
line editor with segment midpoint handles. Interior vertices use named keypoints
``spine_001``, ``spine_002``, etc., between ``head`` and ``tail``. VIAME's CSV and
DIVE JSON readers/writers preserve these markers and subpixel coordinates; JSON
uses the existing ``HeadTails`` LineString. No new KWIVER geometry type is required.

``measure_curve`` responses also provide ``left_keypoints`` and
``right_keypoints`` dictionaries containing named points for export through these
writers. In DIVE, editing a multi-point line invokes ``measure_line`` with an
ordered array of vertices per camera. The service detects this form and rematches
samples along the polyline before summing triangulated segment lengths. Vertex
indices across independently edited camera views are never assumed to match.
The original two-endpoint request keeps its existing behavior. The editor path
uses a polyline (no spline smoothing), while ``measure_curve`` retains its
configurable smoothing and bidirectional options.
