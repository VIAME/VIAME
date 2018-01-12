This folder contains an example for video search. <br>

WARNING: This example is a work in progress, and should only be attempted by advanced users
for the time being. <br>

Building and running this examples requires: <br>

  (a) The python packages: <br>
  (b) A VIAME build with VIAME_ENABLE_SMQTK, BURNOUT, YOLO, OPENCV, VXL and VIVIA. <br>

An arbitrary tracking pipeline is used to first generate spatio-temporal object tracks
representing object candidate locations in video. Descriptors are generated around these
object tracks, which get indexed into a database and can be queried upon. By indicating
which query results are correct, a model can be trained for a new object category and
saved to an output file to be reused again.
