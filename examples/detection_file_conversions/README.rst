
======================================
Detection File Formats and Conversions
======================================

This document corresponds to `this example online`_, in addition to the
detection_file_conversions example folder in a VIAME installation.

.. _this example online: https://github.com/VIAME/VIAME/tree/master/examples/detection_file_conversions

This folder contains examples of how to convert between textual formats representing object
detections, tracks, results, etc. There are multiple ways to perform format conversions,
either using KWIVER pipelines with reader/writer nodes (e.g. see pipelines directory) or
using quick standalone scripts (see scripts). Conversion pipelines are simple, containing
a detection input node (reader) and output node (writer).

****************************
Integrated Detection Formats
****************************

A subset of the output ASCII formats already integrated into VIAME is listed below.
New formats can be integrated to the system by implementing a derived version of the
vital::algo::detected_object_set_input class in C++, or via making a python process which
produces detected_object_sets or object_track_sets.

| **Default CSV - Default Comma Seperated Value Detection Format**
| 
|  The default CSV format contains 1 detection per line, with each line as follows:
|
|   - 1: Detection or Track Unique ID
|   - 2: Video or Image String Identifier
|   - 3: Unique Frame Integer Identifier
|   - 4: TL-x (top left of the image is the origin: 0,0)
|   - 5: TL-y
|   - 6: BR-x
|   - 7: BR-y
|   - 8: Detection Confidence (how likely is this actually an object)
|   - 9: Target Length
|   - 10,11+  : class-name  score (this pair may be omitted or repeated)
|
|  This file format is supported by most GUIs and detector training tools.
|
| **KW18 - Kitware KW18 Column Seperated Track Format**
|
|   KW18s are a space seperated file format for representing detections or tracks.
|
|   Each KW18 file has a header stating its contents, as follows: # 1:Track-id
|   2:Track-length 3:Frame-number 4:Tracking-plane-loc(x) 5:Tracking-plane-loc(y)
|   6:velocity(x) 7:velocity(y) 8:Image-loc(x) 9:Image-loc(y) 10:Img-bbox(TL_x)
|   11:Img-bbox(TL_y) 12:Img-bbox(BR_x) 13:Img-bbox(BR_y) 14:Area 15:World-loc(x)
|   16:World-loc(y) 17:World-loc(z) 18:timestamp 19:track-confidence
|
| **HABCAM - Annotation format used by the HabCam project**
|
|   A typical habcam annotation looks like:
|
|     201503.20150517.png 527 201501 boundingBox 458 970 521 1021
|
|   Which corresponds to image_name, species_id (species id to labels seperate),
|   date, annot_type [either boundingBox, line, or point], tl_x, tl_y, bl_x, bl_y
|
|   For the point type, only 1 set of coordinate is provided
|
|   An alternative format, that the reader also supports, looks like:
|
|     201503.20150517.png,527,scallop,"""line"": [[458, 970], [521, 1021]]"
|
|   which is more or less the same as the prior, just formatted differently.
|
| **Detection CSV (Deprecated) - Additional Comma Seperated Value Detection Format**
|
|  A detection only CSV format contains 1 detection per line, with each line as follows:
|
|   - 1: frame number
|   - 2: file name
|   - 3: TL-x (top left of the image is the origin: 0,0)
|   - 4: TL-y
|   - 5: BR-x
|   - 6: BR-y
|   - 7: detection confidence
|   - 8,9+  : class-name  score (this pair may be omitted or repeated)
|

*******************
Example Conversions
*******************

There are multiple ways to perform format conversions, either using KWIVER
pipelines with reader/writer nodes (e.g. see pipelines directory) or
using quick standalone scripts (see scripts). Conversion pipelines
are simple, containing a detection input node (reader) and output
node (writer).
