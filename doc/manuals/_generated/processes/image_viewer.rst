  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "annotate_image", "false", "NO", "Add frame number and other text to display."
   "footer", "(no default value)", "NO", "Footer text for image display. Displayed centered at bottom of image."
   "header", "(no default value)", "NO", "Header text for image display."
   "pause_time", "0", "NO", "Interval to pause between frames. 0 means wait for keystroke, Otherwise interval |br|\ is in seconds (float)"
   "title", "Display window", "NO", "Display window title text.."

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "image", "kwiver:image", "_required", "Single frame image."
   "timestamp", "kwiver:timestamp", "(none)", "Timestamp for input image."

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
   :: image_viewer
 # Add frame number and other text to display.
   annotate_image = false
 # Footer text for image display. Displayed centered at bottom of image.
   footer = <value>
 # Header text for image display.
   header = <value>
 # Interval to pause between frames. 0 means wait for keystroke, Otherwise
 # interval is in seconds (float)
   pause_time = 0
 # Display window title text..
   title = Display window
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
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

.. doxygenclass:: kwiver::image_viewer_process
   :project: kwiver
   :members:

