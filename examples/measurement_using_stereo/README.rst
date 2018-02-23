
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


.. _PR1: https://github.com/Kitware/VIAME/pull/25
.. _PR2: https://github.com/Kitware/VIAME/pull/38


Running the Demo (WIP)
======================


Run CMake to automatically download the demo data into this example folder.
Alternatively you can download the demo data `directly`_.

.. _directly: https://data.kitware.com/#item/5a8607858d777f068578345e`

Setup:
------

Make sure you build VIAME with `VIAME_ENABLE_PYTHON=True` and
`VIAME_ENABLE_CAMTRAWL=True`.  (For development it is useful to set
`VIAME_SYMLINK_PYTHON=True`)

For simplicity this tutorial will assume that the VIAME source directory is
`~/code/VIAME` and the build directory is `~/code/VIAME/build`. Please modify
these as needeed to match your system setup. We also assume that you have built
VIAME.

After you build viame, remember to source the setup VIAME script. Then change directory to this example folder.

::

    # move to your VIAME build directory
    cd ~/code/VIAME/build
    # Run the setup script to setup the proper paths and environment variables
    source install/setup_viame.sh

    # you may also want to set these environment variables
    # export KWIVER_DEFAULT_LOG_LEVEL=debug
    export KWIVER_DEFAULT_LOG_LEVEL=info
    export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes


Running via the pipeline runner
-------------------------------

To run the process using the sprokit C++ pipeline we use the the pipeline
runner use the command: (Note this method may not be stable and is under
development)

::

    # First move to the example directory
    cd ~/code/VIAME/examples/measurement_using_stereo

    # Then run the pipeline file
    pipeline_runner -p camtrawl_demo.pipe -S pythread_per_process


This example runs at about 4.0Hz, and takes 13.3 seconds to complete on a 2017
i7 2.8Ghz Dell laptop.


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

    # First move to the example directory
    cd ~/code/VIAME/examples/measurement_using_stereo

    # Run the camtrawl module directly via the path
    python ../../plugins/camtrawl/python/viame/processes/camtrawl \
        --left=camtrawl_demodata/left --right=camtrawl_demodata/right \
        --cal=camtrawl_demodata/cal.npz \
        --out=out --draw -f

Without the `--draw` flag the above example, this example runs at about 2.5Hz,
and takes 20 seconds to complete on a 2017 i7 2.8Ghz Dell laptop.

With `--draw` it takes significantly longer (it runs at 0.81 Hz and takes over
a minute to complete), but will output images like the one at the top of this
readme as well as a CSV file.

Note that the KWIVER C++ Sprokit pipline offers a significant speedup (4Hz vs
2.5Hz), although it currently does not have the ability to output the algorithm
visualization.
