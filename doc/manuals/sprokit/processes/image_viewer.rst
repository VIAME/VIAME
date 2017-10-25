image_viewer
============

.. _image_viewer:

Display input image and delay

..  doxygenclass:: kwiver::image_viewer_process
    :project: kwiver

Configuration
-------------
**annotate_image** = false       *Not tunable*
Add frame number and other text to display.

**footer** = (no default value)       *Not tunable*
Footer text for image display. Displayed centered at bottom of image.

**header** = (no default value)       *Not tunable*
Header text for image display.

**pause_time** = 0       *Not tunable*
Interval to pause between frames. 0 means wait for keystroke, Otherwise interval
is in seconds (float)

**title** = Display window       *Not tunable*
Display window title text..

Input Ports
-----------

**image**
Single frame image.

Data type  : kwiver:image
Flags      : _required

**timestamp**
Timestamp for input image.

Data type  : kwiver:timestamp
Flags      : (none)

Output Ports
------------

Pipefile Usage
--------------
The following sections describe the blocks needed to use this process in a pipe file

Pipefile block
~~~~~~~~~~~~~~

.. code::

 # ================================================================
 process disp
   :: image_viewer
 :annotate_image         true
 :pause_time             2.0
 :footer                 footer_footer
 :header                 header-header
 # ================================================================

Pipefile connections
~~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 connect from processX.timestamp
         to   disp.timestamp
 connect from processX.image
         to   disp.image

        
The follwing Output ports are available from this process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # There are no output port's for this process

Class Description
-----------------
        
..  doxygenclass:: kwiver::image_viewer_process
    :project: kwiver
    :members: