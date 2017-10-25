frame_list_input
================

.. _frame_list_input:

Pipefile Usage
--------------
The following sections describe the blocks needed to use this process in a pipe file

Pipefile block
~~~~~~~~~~~~~~

.. code::

 # ================================================================
 process input
   :: frame_list_input
 # Input file containing new-line separated paths to sequential image files.
   image_list_file = C:/Programming/KWIVER/builds/release/examples/pipelines/image_list.txt
   frame_time = .9
 # image_io algorithm to use for 'image_reader'.
  image_reader:type = ocv
 # ================================================================

Available :ref:`image_io impl_name options<algo_image_io>` for image_reader:type

Pipefile connections
~~~~~~~~~~~~~~~~~~~~

The following Input ports will need to be set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # There are no input port's for this process

        
The follwing Output ports are available from this process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

 # This process will provide these vital objects to any processes
 connect from input.timestamp
         to   processX.timestamp
 connect from input.image
         to   processX.image

Class Description
-----------------
        
..  doxygenclass:: kwiver::frame_list_process
    :project: kwiver
    :members: