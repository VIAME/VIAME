  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "wait_on_queue_full", "TRUE", "NO", "When the output queue back to the application is full and there is more data to |br|\ add, should new data be dropped or should the pipeline block until the data can |br|\ be delivered. The default action is to wait until the data can be delivered."

Input Ports
-----------

There are no input ports for this process.


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
   :: output_adapter
 # When the output queue back to the application is full and there is more data
 # to add, should new data be dropped or should the pipeline block until the
 # data can be delivered. The default action is to wait until the data can be
 # delivered.
   wait_on_queue_full = TRUE
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

Class Description
-----------------

.. doxygenclass:: kwiver::output_adapter_process
   :project: kwiver
   :members:

