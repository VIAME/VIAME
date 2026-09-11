
======================================
Detection File Formats and Conversions
======================================

This document corresponds to the 'Detection File Conversions' example folder within a
VIAME desktop installation. This folder contains examples of different formats which VIAME
supports, and additionally how to convert between textual formats representing object
detections, tracks, results, etc. Conversions are performed with the ``viame convert``
tool, which reads any registered annotation format and writes any other, one file or a
whole folder at a time::

    viame convert annotations.csv annotations.json          # VIAME CSV to COCO
    viame convert annotations.csv annotations.dive.json     # VIAME CSV to DIVE
    viame convert results.json results.csv                  # COCO or DIVE to VIAME CSV
    viame convert training_data output_folder -o coco       # every file under a folder

The input format is recognised from each file's extension and content, and the output
format from the output extension or the ``-o`` flag. When the imagery an annotation file
belongs to sits next to it (images in the same folder, or a video), the tool uses it for
the frame names, frame count and timing of the output; ``--no-images`` converts from the
annotation files alone, and ``--images`` points at imagery kept elsewhere. See the
`Example Conversions`_ section below and the ``bulk_convert`` scripts in this folder.

.. _Detection File Conversions: https://github.com/VIAME/VIAME/tree/master/examples/detection_file_conversions

A subset of the output ASCII formats already integrated into VIAME is listed below.
New formats can be integrated to the system by implementing a derived version of the
vital::detected_object_set_input or vital::read_object_track_set classes in C++ or
python, which produce either detected_object_sets or object_track_sets, respectively.

**************************
VIAME CSV - Default Format
**************************

There are 3 parts to a VIAME csv. First, 9 required fields comma seperated, with
a single line for either each detection, or each detection state, in a track:

- 1: Detection or Track Unique ID
- 2: Video or Image String Identifier
- 3: Unique Frame Integer Identifier
- 4: TL-x (top left of the image is the origin: 0,0)
- 5: TL-y
- 6: BR-x
- 7: BR-y
- 8: Auxiliary Confidence (how likely is this actually an object)
- 9: Target Length

Where detections can be linked onto tracks on multiple frames via sharing the
same track ID field. Depending on the context (image or video) the second field
may either be video timestamp or an image filename. Field 3 is a unique frame
identifier for the frame in the given video or loaded sequence, starting from 0
not 1. Fields 4 through 7 represent a bounding box for the target in the imagery.
Depending on the context, auxiliary confidence may represent how likely this
detection is an object, or it may be the confidence in the length measurement,
if present. If length measurement is not present, it can be specified with a
value less than 0, most commonly "-1".

Next, a sequence of optional species <=> score pairs, also comma seperated:

- 10,11+  : class-name, score (this pair may be omitted or repeated)

There can be as many class, score pairs as necessary (e.g. fields 12 and 13, 14
and 15, etc...). In the case of tracks, which may span multiple lines and thus
have multiple probabilities per line, the probabilities from the last state in
the track should be treated as the aggregate probability for the track and it's
okay for prior states to have no probability to prevent respecifying it. In the
class and score list, the highest scoring entries should typically be listed first.

Lastly, optional categorical values associated with each detection in the file
after species/class pairs. Attributes are given via a keyword followed by any
space seperate values the attribute may have. Possible attributes are:

 (kp) head 120 320            [optional head, tail, or arbitrary keypoints]

 (atr) is_diseased true       [attribute keyword then boolean or numeric value]

 (note) this is a note        [notes take no form just can't have commas]

 (poly) 12 455 40 515 25 480  [a polygon for the detection]

 (hole) 38 485 39 490 37 470  [a hole in a polygon for a detection]

 (mask) ./masks/mask02393.png [a reference to an external pixel mask image]

Throwing together all of these components, an example line might look like:

1,image.png,0,104,265,189,390,0.32,1.5,flounder,0.32,(kp) head 120 320

This file format is supported by most GUIs and detector training tools. It can
be used via specifying the 'viame_csv' keyword in any readers or writers

*********
COCO JSON
*********

COCO (Common Objects in Context) jsons are a json schema popularized by the
COCO academic computer vision competitions, but are now also used in other
applications more widely, for example in the cvat annotation tool. The base
format is defined at https://cocodataset.org

Compared to the CSV format they are typically larger but much more extensible,
structured, and have more capacities for optional fields.

The COCO JSON reader/writer can be specified in config files using 'coco'.

VIAME's COCO writers produce output compatible with kwcoco_, an extended COCO
format from Kitware that adds support for video sequences, object tracks, and
richer image metadata. The full informal specification for these extensions is
available at:

https://github.com/Kitware/kwcoco/blob/main/kwcoco/coco_schema_informal.rst

.. _kwcoco: https://github.com/Kitware/kwcoco

The following kwcoco extensions are used by VIAME beyond the base COCO format:

- Top-level ``videos`` table — groups images into video sequences. Each video
  entry has an ``id`` and ``name``. The track writer emits a single video entry
  per output file. This table is not present in base COCO.
- Top-level ``tracks`` table — defines named track identities that annotations
  reference. Each track entry has an ``id`` and ``name``. In base COCO there is
  no concept of tracks; annotations are independent per-image.
- Per-annotation ``track_id`` field — links an annotation to an entry in the
  ``tracks`` table, associating detections of the same object across frames.
- Per-image ``video_id`` field — links an image to its parent video entry in
  the ``videos`` table.
- Per-image ``frame_index`` field — integer giving the temporal ordering of the
  image within its video. Used for sequential playback and streaming reads.
- Per-image ``timestamp`` field — numeric timestamp (seconds) for the frame.
  The base COCO format has no per-image timing information.
- ``auxiliary`` on images — list of auxiliary image assets with ``file_name``
  and ``channels`` fields, used for multi-spectral or multi-file image
  composition.

Files produced by the detection writer (without tracks) include ``frame_index``
on each image but omit the ``videos`` and ``tracks`` tables, remaining
compatible with standard COCO readers that ignore unknown fields.

**************
HABCAM CSV/SSV
**************

Space or comma seperated annotation format used by the HabCam project

A typical habcam annotation looks like:

 201503.20150517.png 527 201501 boundingBox 458 970 521 1021

Which corresponds to image_name, species_id (species id to labels seperate),
date, annot_type [either boundingBox, line, or point], tl_x, tl_y, bl_x, bl_y

For the point type, only 1 set of coordinate is provided

An alternative format, that the reader also supports, looks like:

 201503.20150517.png,527,scallop,"""line"": [[458, 970], [521, 1021]]"

which is more or less the same as the prior, just formatted differently.

The habcam reader/writer can be specified in config files using 'habcam'.

*********
DIVE JSON
*********

DIVE JSON is the native annotation format used by the DIVE annotation
interface. It stores tracks as top-level objects keyed by track ID, with each
track containing temporal features (one per frame), confidence pairs for class
labels, and optional per-detection attributes, keypoints, and GeoJSON polygon
geometry.

The full format specification is available at:

https://kitware.github.io/dive/DataFormats/

The DIVE JSON reader can be specified in config files using 'dive'.

Both a reader and a writer are provided. The writer emits version 2 documents
with one feature per frame carrying the box, keyframe flag, head and tail
points, polygon geometry, fish length, notes and per-detection attributes;
class labels become the track's confidence pairs. Files named ``*.dive.json``
are recognised as DIVE by the auto reader and the convert tool, and plain
``.json`` files are told apart from COCO by their content.

****************************
Auto - Format Auto-Detection
****************************

The 'auto' reader inspects the file extension and content to automatically
select the correct format reader, removing the need to specify it manually.

The auto reader can be specified in config files using 'auto'.

For detection reading (detected_object_set_input), it selects between:

- DIVE JSON — files ending in ``.dive.json``, or ``.json`` files whose content
  contains DIVE-specific keys (``tracks``, ``features``, ``confidencePairs``)
- COCO JSON — files ending in ``.coco.json``, or ``.json`` files whose content
  contains COCO-specific keys (``images``, ``annotations``, ``categories``)
- VIAME CSV — ``.csv`` files
- YOLO — ``.txt`` files (image lists with per-image label files)
- CVAT XML — ``.xml`` files

For track reading (read_object_track_set), it selects between:

- DIVE JSON — files ending in ``.dive.json``, or ``.json`` files whose content
  contains DIVE-specific keys (``tracks``, ``features``, ``confidencePairs``)
- COCO JSON — files ending in ``.coco.json``, or ``.json`` files whose content
  contains COCO-specific keys (``images``, ``annotations``, ``categories``)
- VIAME CSV — ``.csv`` files

When a ``.json`` file does not match either DIVE or COCO patterns, both
the detection and track readers default to COCO.

*****************
KW18 - Deprecated
*****************

KW18, or Kitware KW18 Column Seperated Track Format, are a space seperated
file format for representing detections or tracks.

Each KW18 file has a header stating its contents, as follows:

# 1:Track-id 2:Track-length 3:Frame-number 4:Tracking-plane-loc(x) 5:Tracking-plane-loc(y)
6:velocity(x) 7:velocity(y) 8:Image-loc(x) 9:Image-loc(y) 10:Img-bbox(TL_x)
11:Img-bbox(TL_y) 12:Img-bbox(BR_x) 13:Img-bbox(BR_y) 14:Area 15:World-loc(x)
16:World-loc(y) 17:World-loc(z) 18:timestamp 19:track-confidence

The kw18 reader/writer can be specified in config files using 'kw18'.

***********************
KWIVER CSV - Deprecated
***********************

A detection only CSV format contains 1 detection per line, with each line as follows:

- 1: frame number
- 2: file name
- 3: TL-x (top left of the image is the origin: 0,0)
- 4: TL-y
- 5: BR-x
- 6: BR-y
- 7: detection confidence
- 8,9+  : class-name  score (this pair may be omitted or repeated)

The kwiver reader/writer can be specified in config files using 'csv'. We reccomend
you don't use it for anything.

*******************
Example Conversions
*******************

The ``viame convert`` tool converts between every registered reader and writer
directly, without a pipeline. Tracks are carried across when both formats hold
them, otherwise per-frame detections. Run ``viame convert --list-formats`` to see
what is available; ``viame convert --help`` lists the options.

Single files::

    viame convert groundtruth.csv groundtruth.json           # to COCO
    viame convert groundtruth.csv groundtruth.dive.json      # to DIVE
    viame convert groundtruth.kw18 groundtruth.csv           # KW18 to VIAME CSV
    viame convert habcam.csv habcam_viame.csv -i habcam      # HabCam CSV to VIAME CSV
    viame convert results.json results.csv -o viame_csv      # COCO or DIVE to VIAME CSV

Folders, mirroring the input layout into the output folder with the new extension::

    viame convert training_data converted -o coco
    viame convert training_data converted -o dive --no-images

Imagery alongside the annotations is used automatically: an image folder gives the
frame names and count (so empty frames are recorded too, as COCO expects), and a
video gives the frame count and timestamps. ``--frame-rate`` sets the rate the
frames of a video are numbered at, matching the ``-frate`` used when the
annotations were produced, and applies timestamps to image sequences::

    viame convert clip.csv clip.json --frame-rate 5          # clip.mp4 found alongside
    viame convert annotations.csv out.json --images frames/  # imagery kept elsewhere

Reader and writer settings are passed as ``-s key=value``, prefixed by ``reader:``
or ``writer:`` when the two share a key::

    viame convert in.csv out.csv -s writer:tot_option=average

``viame run --gt-only`` also converts annotation folders through the same tool, for
batch runs that already use the run applet, and the ``bulk_convert`` scripts in this
folder show both the with-data and annotation-only forms.

The ``standalone_utils`` folder keeps older single-purpose scripts for formats that
are not registered readers (Scallop-TK, PVO and similar).
