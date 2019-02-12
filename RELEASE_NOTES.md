
v0.9.9.2 - 2/12/2019
====================


-Add train deep model from GUI dropdown


v0.9.9.1 - 2/6/2019
===================


-Fix issues in prior release dealing with default indexing option and finding local models


v0.9.9.0 - 2/4/2019
===================


-Add YOLOv3, YOLOv3 with Temporal Features detectors


-GUI bug fixes: issues with running embedded pipes and detections not always showing up


-Configuration file cleanup


-Remove multiple models from default installers, move to seperate .zips



v0.9.8.11 - 12/4/2018
=====================


-Improved timeline plotting for multi-video inputs


-Fixed issue in image list project files introduced in prior


-More intelligent default track classification probabilities which make use of keyframes



v0.9.8.10 - 11/26/2018
======================


-Add automatic dynamic range compute for 16-bit imagery in GUI


-Expose video rate for image list processing in default project files


-Added extra error checking on indexing scripts involving databases, improved error reporting



v0.9.8.9 - 11/12/2018
=====================


-Allow video projects to process directory of directory of images


-Additional error checking on index formation


-Improvements to launch_timeline_viewer script


-Added GUI display dynamic range parameters for 16-bit imagery



v0.9.8.8 - 10/29/2018
=====================


-Fix issue with measurement example after config refactor



v0.9.8.7 - 10/25/2018
=====================


-Additional bug fixes and parameter tuning relating to tile boundaries when chipping enabled in detectors



v0.9.8.6 - 10/24/2018
=====================


-Fixed a bug in debayer and color correction filter .pipe which was causing garbled results in the example


-Added RELEASE_NOTES to binaries


-Fixes to Windows CPU-only binaries


-Fixes for properly processing greyscale imagery


-Non-maximum suppression tuning


-If there is a corrupt frame in the middle of a sequence, keep on processing in both GUI and CLI



v0.9.8.5 - 10/18/2018
=====================


-Improved video support in default project file run scripts


-Configuration files have been refactored and their structured cleaned up


-Additional annotation GUI embedded pipelines added, directory structure improved
