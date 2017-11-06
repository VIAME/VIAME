  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "frame_time", "0.03333333", "NO", "Inter frame time in seconds. If the input video stream does not supply frame |br|\ times, this value is used to create a default timestamp. If the video stream has |br|\ frame times, then those are used."
   "video_filename", "(no default value)", "NO", "Name of video file."

Input Ports
-----------

There are no input ports for this process.


Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "image", "kwiver:image", "(none)", "Single frame image."
   "timestamp", "kwiver:timestamp", "(none)", "Timestamp for input image."
   "video_metadata", "kwiver:video_metadata", "(none)", "Video metadata vector for a frame."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: video_input
 # Inter frame time in seconds. If the input video stream does not supply frame
 # times, this value is used to create a default timestamp. If the video stream
 # has frame times, then those are used.
   frame_time = 0.03333333
 # Name of video file.
   video_filename = <value>
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # There are no input port's for this process


The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.image
          to   <downstream-proc>.image
 connect from <this-proc>.timestamp
          to   <downstream-proc>.timestamp
 connect from <this-proc>.video_metadata
          to   <downstream-proc>.video_metadata

Class Description
-----------------

.. doxygenclass:: class kwiver::video_input_process
   :project: kwiver
   :members:

