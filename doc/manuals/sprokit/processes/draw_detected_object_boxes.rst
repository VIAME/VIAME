draw_detected_object_boxes
==========================
.. _draw_detected_object_boxes:

Pipefile Usage
--------------
The following sections describe the blocks needed to use this process in a pipe file

Pipefile block
~~~~~~~~~~~~~~

.. code::

 # ================================================================
 process draw # This name can be whatever you want
  :: draw_detected_object_boxes
  :default_line_thickness 3
 # ================================================================

Pipefile connections
~~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # ProcessX will provide a detected_object_set
 connect from processX.detected_object_set
        to   draw.detected_object_set
 # ProcessY will provide an image
 connect from processY.image
        to draw.image

        
The follwing Output ports are available from this process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will provide an image with boxes to any processes
 connect from draw.image
        to   processZ.image

Class Description
-----------------
        
..  doxygenclass:: kwiver::draw_detected_object_boxes_process
    :project: kwiver
    :members: