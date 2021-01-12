
=====================================
GUIs for Visualization and Annotation
=====================================

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/annotation_example_painted.png
   :scale: 30
   :align: center
   :target: https://github.com/VIAME/VIAME/blob/master/examples/annotation_and_visualization/README.rst
|
This document corresponds to `this example online`_, in addition to the
annotation_and_visualization example folder in a VIAME installation.

.. _this example online: https://github.com/VIAME/VIAME/blob/master/examples/annotation_and_visualization

There are a number of GUIs in the system. As part of the VIVIA package the vpView GUI, the current
default desktop annotator, is useful for displaying detections, their respective probabilities,
for running existing automated detectors, and for making new annotations in video. There are
additionally simpler GUIs which can be enabled in .pipe files. vpView can either be pointed directly
to imagery, pointed to a compressed video file (see [install-dir]/configs/prj-*/for_videos) or given
an input prj file that points to the location of input imagery and any optional settings (e.g.
groundtruth, computed detections, and/or homographies for the input data). If you just want to
use the tool to make annotations you don't need to specify the later three, and just need to
set a DataSetSpecifier or [reccommended] use the File->New Project option to load imagery directly
without a prj file. Also, see the below example guide and videos.

There are 2 default run scripts in this folder. "launch_view_interface" launches the main
vpview annotation and results display GUI while "run_display_pipe" runs the simpler in-pipeline
display GUI. Lastly, "run_chip_pipe" creates image chips and "run_draw_pipe" does the same
as display, only writing out images with boxes drawn on top of them to file.

**********************************
vpView Annotation Process Overview
**********************************

| `Manual Annotation Guide (PDF)`_
| `Example Video Overviews (Youtube)`_

.. _Manual Annotation Guide (PDF): https://data.kitware.com/api/v1/item/5c6574668d777f072b47cbd6/download
.. _Example Video Overviews (YouTube): https://www.youtube.com/channel/UCpfxPoR5cNyQFLmqlrxyKJw

************************************
Notable Annotation GUI Shortcut Keys
************************************

* *r* = Zoom back to the full image
* *hold ctrl + drag* = create a box in annotation mode (create detection/track)

********************************
vpView GUI Project File Overview
********************************

Examples of the optional contents of loadable prj files are listed below for quick reference.
For those not familiar with the tool, downloading the above manual is best. Project files are
no longer required to be used (imagery can be opened directly via the 'New Project' dropdown),
however, these are listed here for advanced users who may want to configure with multiple
homographies.

Note: The list is not complete, but currently focusing on the most used (and new) parameters

* DataSetSpecifier = filename(or glob)  
  Filename with list of images for each frame or glob for sequence of images  
* TracksFile = filename  
  Filename containing the tracks data.  
* TrackColorOverride = r g b  
  rgb color, specified from 0 to 1, overrides the default vpView track color for this
  project only  
* ColorMultiplier = x  
  Event and track colors are scaled by the indicated value.  Can be used in conjunction
  with the TrackColorOverride  
* EventsFile = filename  
  Filename containing the events data.  
* ActivitiesFile = filename  
  Filename containing the activity data.  
* SceneElementsFile= filename  
  Filename containing the scene elements (in json file).  
* AnalysisDimensions = W H  
  Dimension (in pixel/image coordinates) of AOI.  Ignored when using a mode that leverages
  image corner points for imagetransformation.  
* OverviewOrigin = U V  
  Offset of image data such that "0, 0" occurs at the AOI origin. Should always be negative
  #'s.  Like AnalysisDimensions, unused when image corner points are used.  
* AOIUpperLeftLatLon = lat lon  
  Required for "Translate image" mode of corner point usage (Tools->Configure->Display);
  also required for displaying an AOI when the source imagery isn't ortho-stabilized  
* AOIUpperRightLatLon = lat lon  
* AOILowerLeftLatLon = lat lon  
* AOILowerRightLatLon = lat lon  
  If the UpperLeft and LowerRight are specified, an AOI "box" can be displayed.  Depending
  on the nature of the homography controlling image display / transformation, additional
  corner point may improve the designation of the region.  
* FrameNumberOffset = N  
  Positive value to offset imagery relative to the track/event data.  A value of 3 would
  mean that the 1st image would correspond to track frame 2 (0-based numbering)  
* ImageTimeMapFile = filename  
  Specifies file containing map of "filename <space> timestamp (in seconds)"
  one line per frame.  The file can be created via File->Export Image Time Stamps  
* HomographyIndexFile = filename  
  Specifies file containing frame number/timestamp/homography sequence for all frames
  specified by the DataSetSpecifier.  If the tag is set and the number of homographies
  match the image source count, the "Image-loc"'s of the tracks (not the "Img-bbox") are
  stored in coordinate frame mapped to by the homographies.  This enables track trails
  during playback (for source imagery that isn't stored in stabilized form)  
* HomographyReferenceFrame = frame index  
  Specifies the frame to use as the reference homography frame for stabilizing the video
  (if homographies are present). If, instead of stabilizing the video, the homographies should
  be used to stabilize the tracks, set the HomographyReferenceFrame to -1 (defaults to 0).  
* FiltersFile = filename  (note, support not yet in master)  
  Specifies file containing definitions of spatial filters for the project. The coordinate
  system (lat/lon or image/pixels) must be consistent between states when the filters are
  saved versus when the project / filter file is loaded.  
* IgnoreImageCoords = true/false (note, support not yet in master)  
  In the world mode, ignores the image bounding box data and shows a head (point/dot) at the end
  of the track tail. This parameter only affects the track head and not the tail.  
* ColorWindow = W (defaults to 255)  
  Window / range of input color values that will be mapped. The value gives the total range,
  not the distance from the median.  
* ColorLevel = L (defaults to 127)  
  Input color value that will be mapped to the median output value, and also serves as the
  median value of the input color range.  
