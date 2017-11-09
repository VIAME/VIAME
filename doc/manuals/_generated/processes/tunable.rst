  .. |br| raw:: html

   <br />

Configuration
-------------

.. csv-table::
   :header: "Variable", "Default", "Tunable", "Description"
   :align: left
   :widths: auto

   "non_tunable", "(no default value)", "NO", "The non-tunable output."
   "tunable", "(no default value)", "YES", "The tunable output."

Input Ports
-----------

There are no input ports for this process.


Output Ports
------------

.. csv-table::
   :header: "Port name", "Data Type", "Flags", "Description"
   :align: left
   :widths: auto

   "non_tunable", "string", "(none)", "The non-tunable output."
   "tunable", "string", "(none)", "The tunable output."

Pipefile Usage
--------------

The following sections describe the blocks needed to use this process in a pipe file.

Pipefile block
--------------

.. code::

 # ================================================================
 process <this-proc>
   :: tunable
 # The non-tunable output.
   non_tunable = <value>
 # The tunable output.
   tunable = <value>
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
 connect from <this-proc>.non_tunable
          to   <downstream-proc>.non_tunable
 connect from <this-proc>.tunable
          to   <downstream-proc>.tunable

Class Description
-----------------

.. doxygenclass:: sprokit::tunable_process
   :project: kwiver
   :members:

