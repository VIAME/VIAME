  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "base_filename", "(no default value)", "NO", "Base file name (no extension) for KWA component files"
   "compress_image", "true", "NO", "Whether to compress image data stored in archive"
   "mission_id", "(no default value)", "NO", "Mission ID to store in archive"
   "output_directory", ".", "NO", "Output directory where KWA will be written"
   "separate_meta", "true", "NO", "Whether to write separate .meta file"
   "static/corner_points", "(no default value)", "NO", "A default value to use for the 'corner_points' port if it is not connected."
   "static/gsd", "(no default value)", "NO", "A default value to use for the 'gsd' port if it is not connected."
   "stream_id", "(no default value)", "NO", "Stream ID to store in archive"

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "corner_points", "corner_points", "_static", "Four corner points for image in lat/lon units, ordering ul ur lr ll."
   "gsd", "kwiver:gsd", "_static", "GSD for image in meters per pixel."
   "homography_src_to_ref", "kwiver:s2r_homography", "_required", "Source image to ref image homography."
   "image", "kwiver:image", "_required", "Single frame image."
   "timestamp", "kwiver:timestamp", "_required", "Timestamp for input image."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto


Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: kw_archive_writer
 # Base file name (no extension) for KWA component files
   base_filename = <value>
 # Whether to compress image data stored in archive
   compress_image = true
 # Mission ID to store in archive
   mission_id = <value>
 # Output directory where KWA will be written
   output_directory = .
 # Whether to write separate .meta file
   separate_meta = true
 # A default value to use for the 'corner_points' port if it is not connected.
   static/corner_points = <value>
 # A default value to use for the 'gsd' port if it is not connected.
   static/gsd = <value>
 # Stream ID to store in archive
   stream_id = <value>
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.corner_points
          to   <upstream-proc>.corner_points
 connect from <this-proc>.gsd
          to   <upstream-proc>.gsd
 connect from <this-proc>.homography_src_to_ref
          to   <upstream-proc>.homography_src_to_ref
 connect from <this-proc>.image
          to   <upstream-proc>.image
 connect from <this-proc>.timestamp
          to   <upstream-proc>.timestamp

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports

Class Description
-----------------

.. doxygenclass:: kwiver::kw_archive_writer_process
   :project: kwiver
   :members:

