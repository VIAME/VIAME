
===========================
Length Measurement Examples
===========================

.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/fish_measurement_example.png
   :scale: 60
   :align: center

This section corresponds to `this example online`_, in addition to the
measurement_using_stereo example folder in a VIAME installation. This folder contains
examples covering fish measurement using stereo.

.. _this example online: https://github.com/Kitware/VIAME/tree/master/examples/measurement_using_stereo


`Coming Soon`_  on Pull Request 25

.. _Coming Soon: https://github.com/Kitware/VIAME/pull/25


Running the Demo (WIP)
======================


Run CMake to automatically download the demo data into this example folder.
Alternatively you can download the demo data `directly`_.

.. _directly: https://data.kitware.com/#item/5a8607858d777f068578345e`

Building:
---------

Make sure you build VIAME with `VIAME_ENABLE_PYTHON=True` and
`VIAME_ENABLE_CAMTRAWL=True`.  (For development it is useful to set
`VIAME_SYMLINK_PYTHON=True`)


Remember to source the setup VIAME script. Then change directory to this example folder.

::

    cd /path/to/your/viame/build

    source install/setup_viame.sh

    # you may also want to set these environment variables
    # export KWIVER_DEFAULT_LOG_LEVEL=debug
    export KWIVER_DEFAULT_LOG_LEVEL=info
    export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes


    cd /path/to/your/viame/examples/measurement_using_stereo


Running via the pipeline runner
-------------------------------

To run the process using the sprokit C++ pipeline we use the the pipeline
runner use the command: (Note this method may not be stable and is under
development)

::

    pipeline_runner -p camtrawl_demo.pipe -S pythread_per_process


Running via installed camtrawl python module 
--------------------------------------------

The above pipeline can alternatively be run as a python script.

You should be able to run the help command

:: 

    python -m viame.processes.camtrawl.demo --help

The script can be run on the demodata via

::

    python -m viame.processes.camtrawl.demo \
        --left=camtrawl_demodata/left --right=camtrawl_demodata/right \
        --cal=camtrawl_demodata/cal.npz \
        --out=out --draw -f


Running via the standalone script
---------------------------------

Alternatively you can run by specifying the path to camtrawl module (if you
have a python environment you should be able to run this without even building
VIAME)

::

    python ../../plugins/camtrawl/python/viame/processes/camtrawl \
        --left=camtrawl_demodata/left --right=camtrawl_demodata/right \
        --cal=camtrawl_demodata/cal.npz \
        --out=out --draw -f
