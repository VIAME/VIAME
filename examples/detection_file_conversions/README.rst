Detection File Conversions Examples
-----------------------------------

A folder containing misc examples of how to convert between textual
formats representing object detections, tracks, results, etc. There
are multiple ways to perform format conversions, either using KWIVER
pipelines with reader/writer nodes (e.g. see pipelines directory) or
using quick standalone scripts (see scripts). Conversion pipelines
are simple, containing a detection input node (reader) and output
node (writer).


VIAME Integrated Detection Formats
----------------------------------
|
| CSV - Comma Seperated Value Detection Formats
| 
|   detection1, detection2, detection3, 
|
| KW18 - Kitware KW18 Column Seperated Track Format
|
|   detection2, detection3, 
|
| HABCAM - Annotation format used by the HabCam project
|
|   v1
|
|
