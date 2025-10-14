
======================================
Detection File Formats and Conversions
======================================

This document corresponds to the 'Detection File Conversions' example folder within a
VIAME desktop installation. This folder contains examples of different formats which VIAME
supports, and additionally how to convert between textual formats representing object
detections, tracks, results, etc. There are multiple ways to perform format conversions,
either using KWIVER pipelines with reader/writer nodes (e.g. see bulk_convert_using_pipe
script) or using quick standalone scripts (see standalone_utils). Conversion pipelines
are simple, containing a detection input node (reader) and output node (writer).

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
applications more widely, for example in the cvat annotation tool. It is
defined at https://cocodataset.org

Compared to the CSV format they are typically larger but much more extensible,
structured, and have more capacities for optional fields.

The COCO JSON reader/writer can be specified in config files using 'coco'.

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

*******************
DIVE JSON - Limited
*******************

An alternative JSON schema export from the DIVE interface. Unlike COCO, this
is currently only supported by the DIVE tool, not by other scripts and CLIs
within VIAME.

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

There are multiple ways to perform format conversions, either using KWIVER pipelines
with reader/writer nodes (e.g. see pipelines directory in this example directory) or
using quick standalone scripts (see scripts). Conversion pipelines are simple,
containing a detection input node (reader) and output node (writer) and can be run 
with the 'kwiver runner' command line tool.
