v0.11.3 - 10/14/2020
====================


-Add extra log comments to training log for verifying image reads


-Fix filename parsing in seal-tk GUI for unknown filename types



v0.11.2 - 10/4/2020
===================


-Do not write out polygons when just rectangles drawn in GUI


-Support polys in annotation csv in model training tool


-Support training on folders with no labels.txt present


-Improved CFRNN WTF training pipeline with improved features



v0.11.1 - 9/16/2020
===================


-Upgrade multiple dependencies: pytorch, cuda, nvidia driver, mmdet


-Additional support for full frame classifiers


-Support viame csv for polygon saveout from desktop annotator


-Seal multi-view GUI fix zooming issue, allow scoring editing



v0.10.10 - 7/30/2020
====================


-Fix issue on windows running CFRNN training from pipelines dropdown


-Fix issue in C++ hello world example


-Fix issue in linux running IQR search and rapid model generation



v0.10.9 - 4/30/2020
===================


-Fix issue on windows with running CFRNN detectors at test time


-Fix issue with running detector downloading excess un-needed model files



v0.10.9 - 4/06/2020
===================


-Fix issue with CFRNN training on windows


-Add continue training from prior model CFRNN script


-Fix issue in habcam model pack



v0.10.8 - 3/20/2020
===================


-Fix issue with timeout in latest CFRNN training



v0.10.7 - 3/19/2020
===================


-Fix issue with labels.txt file not being used in latest CFRNN training



v0.10.6 - 3/18/2020
===================


-Fix scoring examples which became broken in last


-Add tracker training example for windows binaries


-Fix issue in running test-time multi-GPU from an image input list


-Add extra checks for project file scripts with early exits



v0.10.5 - 2/26/2020
===================


-Add improved CFRNN training proceedure using netharn



v0.10.4 - 1/31/2020
===================


-Fix issue in centos binaries backing X11 library



v0.10.3 - 1/13/2020
===================


-Refine cascade faster rcnn training proceedure first pass


-Add tracker-assisted annotation for video


-Fix split track option in default annotator


-New seal multi-view GUI release



v0.10.2 - 12/1/2019
===================


-Fix CentOS issues with unable to run pipelines (Ubuntu and Windows fine)



v0.10.2 - 11/25/2019
====================


-Updated KWIVER in VIAME causing all add-ons to break and need to be regenerated


-Added in support for object tracking solely based on IOU and image stabilization


-Fixed issue in installers with measurement code after upgrading numpy versions


-Add simple mosaic generation script and related pipelines


-Add multiple seal tracking example pipelines


-Fix issues in prior v0.10.* releases on systems with no CUDA 10s installed



v0.10.1 - 11/16/2019
====================


-Remove ball-tree indexing in favor of full LSH hash searches due to issues


-Begin packaging python in binaries for linux in addition to windows



v0.10.0 - 11/15/2019
====================


-Baseline speed optimizations and bug fixes for IQR, significant database speedups


-Begin packaging python in binaries for windows, will do this for linux soon



v0.9.18 - 11/01/2019
===================


-Fixed windows issues with running and training detectors introduced in last


v0.9.17 - 10/18/2019
====================


-Quick fixes to make IQR GUI not crash on images of different size and large images


-Add SVM training tool example, for training SVMs from large amounts of annotations


-Misc tuning and bug fixes



v0.9.16 - 7/24/2019
===================


-Lots of bug fixes (windows project files)


-Learning rate tuning for latest detection models


-CUDA 10.0 support



v0.9.13 - 5/30/2019
===================


-Add missing .so files into CentOS and Ubuntu binaries to not require external packages



v0.9.12 - 5/21/2019
===================


-Support local detection running for multiple variants of motion detectors


-All detector training routines now also generate KWIVER pipelines


-Remove index_existing and merged changes into index_defaults to reduce confusion


-Additional detection models



v0.9.11 - 5/16/2019
===================


-New default and generic object detection models


-Improved IQR detector performance


-The default box detector training routine is now cascade rnn instead of yolo



v0.9.10.0 - 4/22/2019
=====================


-New detection techniques and default models: cascade faster rcnn, cfrnn with motion, YOLO-WTF


-Additional support for frame-level classification in the system


-Image registration example added, arctic seal examples



v0.9.9.7 - 3/19/2019
====================


-Fixed a few issues in last relating to detector optimizations which broke things in last


-More intuitive final model saveouts


v0.9.9.6 - 2/25/2019
====================


-Windows 7 runtime training tool fixes


-Better automatic video detection, don't process hidden files, dirs recursively or non-videos


-Show correct time offsets in query GUI from video start


v0.9.9.5 - 2/21/2019
====================


-Fix bug added to deep training tool in last version


-Fix training tool windows error reporting


v0.9.9.4 - 2/18/2019
====================


-Fix model saveout in windows binaries broken in last


-Updated default model training parameters


v0.9.9.3 - 2/13/2019
====================


-Minor python change to drop dependency in standard use case


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
