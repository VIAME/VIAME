image_viewer
============

.. _image_viewer:

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
        
..  doxygenclass:: kwiver::frame_list_process
    :project: kwiver
    :members: