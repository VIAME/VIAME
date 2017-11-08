  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "alpha_blend_prob", "true", "YES", "If true, those who are less likely will be more transparent."
   "clip_box_to_image", "false", "YES", "If this option is set to true, the bounding box is clipped to the image bounds."
   "custom_class_color", "(no default value)", "YES", "List of class/thickness/color seperated by semicolon. For example: person/3/255 |br|\ 0 0;car/2/0 255 0. Color is in RGB."
   "default_color", "0 0 255", "YES", "The default color for a class (RGB)."
   "default_line_thickness", "1", "YES", "The default line thickness for a class, in pixels."
   "draw_text", "true", "YES", "If this option is set to true, the class name is drawn next to the detection."
   "select_classes", "*ALL*", "YES", "List of classes to display, separated by a semicolon. For example: |br|\ person;car;clam"
   "text_scale", "0.4", "YES", "Scaling for the text label."
   "text_thickness", "1.0", "YES", "Thickness for text"
   "threshold", "-1", "YES", "min threshold for output (float). Detections with confidence values below this |br|\ value are not drawn."

Input Ports
-----------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "detected_object_set", "kwiver:detected_object_set", "_required", "Set of detected objects."
   "image", "kwiver:image", "_required", "Single frame image."

Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "image", "kwiver:image", "(none)", "Single frame image."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: draw_detected_object_boxes
 # If true, those who are less likely will be more transparent.
   alpha_blend_prob = true
 # If this option is set to true, the bounding box is clipped to the image
 # bounds.
   clip_box_to_image = false
 # List of class/thickness/color seperated by semicolon. For example:
 # person/3/255 0 0;car/2/0 255 0. Color is in RGB.
   custom_class_color = <value>
 # The default color for a class (RGB).
   default_color = 0 0 255
 # The default line thickness for a class, in pixels.
   default_line_thickness = 1
 # If this option is set to true, the class name is drawn next to the detection.
   draw_text = true
 # List of classes to display, separated by a semicolon. For example:
 # person;car;clam
   select_classes = *ALL*
 # Scaling for the text label.
   text_scale = 0.4
 # Thickness for text
   text_thickness = 1.0
 # min threshold for output (float). Detections with confidence values below
 # this value are not drawn.
   threshold = -1
 # ================================================================

Process connections
~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will consume the following input ports
 connect from <this-proc>.detected_object_set
          to   <upstream-proc>.detected_object_set
 connect from <this-proc>.image
          to   <upstream-proc>.image

The following Output ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will produce the following output ports
 connect from <this-proc>.image
          to   <downstream-proc>.image

Class Description
-----------------

.. doxygenclass:: kwiver::draw_detected_object_boxes_process
   :project: kwiver
   :members:

