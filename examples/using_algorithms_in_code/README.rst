
========================
Using Algorithms in Code
========================

********
Overview
********

This document corresponds to the `using algorithms in
code <https://github.com/VIAME/VIAME/tree/main/examples/using_algorithms_in_code>`__
example folder within a VIAME desktop installation.

Any algorithm available to a pipeline can also be called directly from C++ or python. This
example uses a Hough circle detector to show the steps involved in each. The folder contains
these programs:

.. list-table::
   :header-rows: 1

   * - File
     - Shows
   * - ``detector1.cxx``
     - Running one named detector, with an optional configuration file
   * - ``detector3.cxx``
     - Choosing the detector at run time from a configuration file
   * - ``detector4.cxx``
     - The same as ``detector3.cxx``, written against the older KWIVER interface
   * - ``detector1.py``
     - ``detector1.cxx`` in python
   * - ``detector3.py``
     - ``detector3.cxx`` in python

The C++ programs ``detector1`` and ``detector3`` are built along with VIAME. The python ones
run as they are.

***
C++
***

Running a Single Detector
=========================

Detectors accept an image and return detections. The types used to pass data in and out
come from the vital part of KWIVER: an ``image_container`` holds the input image, and a
``detected_object_set`` holds the results.

Vital provides an algorithm for loading an image from a file:

.. code-block:: cpp

   kwiver::vital::image_container_sptr load( std::string const& filename ) const;

The image is then passed to the detector:

.. code-block:: cpp

   virtual vital::detected_object_set_sptr detect( vital::image_container_sptr image_data ) const;

Put together, a complete program is short:

.. code-block:: cpp

   #include <arrows/ocv/image_container.h>
   #include <arrows/ocv/algo/image_io.h>
   #include <arrows/ocv/algo/hough_circle_detector.h>

   #include <string>

   int main( int argc, char* argv[] )
   {
     // Get file name for input image
     std::string filename = argv[1];

     // Create image reader
     kwiver::vital::algo::image_io_sptr image_reader( new kwiver::arrows::ocv::image_io() );

     // Read the image
     kwiver::vital::image_container_sptr the_image = image_reader->load( filename );

     // Create the detector
     kwiver::vital::algo::image_object_detector_sptr detector(
       new kwiver::arrows::ocv::hough_circle_detector() );

     // Send image to detector and get detections
     kwiver::vital::detected_object_set_sptr detections = detector->detect( the_image );

     // See what was detected
     std::cout << "There were " << detections->size() << " detections in the image." << std::endl;

     return 0;
   }

Logging
=======

Vital provides logging through macros that format and report messages:

.. code-block:: cpp

   #include <vital/logger/logger.h>

   kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger( "detector_test" ) );

   LOG_INFO( logger, "There were " << detections->size() << " detections in the image." );

Each logger has a name, which can be used to control which messages are shown. There is
a macro for each severity: error, warning, info, debug and trace. The message is written
as an output stream expression, so any type with an output operator can be logged, and
no end of line is needed.

Configuration Support
=====================

The program above runs the detector with its default settings. Most algorithms need
adjusting for good results, which is done through configuration.

Vital stores settings as keys and values. They are usually read from a file, and can
also be set in code. Every algorithm has a ``get_configuration()`` method that returns
its settings with their defaults, and a ``set_configuration()`` method that applies new
values. The Hough circle detector has these settings:

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Description
   * - ``dp``
     - 1
     - Inverse ratio of the accumulator resolution to the image resolution. With 1 the
       accumulator matches the image; with 2 it has half the width and height.
   * - ``min_dist``
     - 100
     - Minimum distance between the centers of detected circles. Too small and
       neighbouring circles are falsely detected; too large and some are missed.
   * - ``param1``
     - 200
     - Higher of the two thresholds passed to the Canny edge detector. The lower is half
       of it.
   * - ``param2``
     - 100
     - Accumulator threshold for circle centers. The smaller it is, the more false
       circles are detected.
   * - ``min_radius``
     - 0
     - Minimum circle radius.
   * - ``max_radius``
     - 0
     - Maximum circle radius.

``detector1.cxx`` takes a configuration file as an optional second argument. It differs
from the program above in two places:

.. code-block:: cpp

   // Look for name of config file as second parameter
   kwiver::vital::config_block_sptr config;
   if ( argc > 2 )
   {
     config = kwiver::vital::read_config_file( argv[2] );
   }

.. code-block:: cpp

   // If there was a config structure, then pass it to the algorithm
   if ( config )
   {
     detector->set_configuration( config );
   }

A matching configuration file is below. Settings left out of the file keep their
defaults.

::

   min_dist = 120
   param1 = 100

Choosing the Detector at Run Time
=================================

``detector3.cxx`` goes a step further and reads which detector to use from the
configuration file, so the program no longer names one:

.. code-block:: cpp

   #include <vital/plugin_management/plugin_manager.h>
   #include <vital/config/config_block_io.h>
   #include <vital/algo/image_object_detector.h>
   #include <vital/algo/algorithm.txx>
   #include <arrows/ocv/image_container.h>
   #include <arrows/ocv/algo/image_io.h>

   #include <string>

   namespace kv = kwiver::vital;
   namespace kva = kwiver::vital::algo;

   int main( int argc, char* argv[] )
   {
     // Create logger to use for reporting errors and other diagnostics
     kv::logger_handle_t logger( kv::get_logger( "detector_test" ) );

     // Initialize and load all discoverable plugins
     kv::plugin_manager::instance().load_all_plugins();

     std::string filename = argv[1];
     kv::config_block_sptr config = kv::read_config_file( argv[2] );

     kva::image_io_sptr image_reader( new kwiver::arrows::ocv::image_io() );
     kv::image_container_sptr the_image = image_reader->load( filename );

     // Create the detector named in the configuration
     kva::image_object_detector_sptr detector;
     kv::set_nested_algo_configuration< kva::image_object_detector >(
       "detector", config, detector );

     if ( ! detector )
     {
       LOG_ERROR( logger, "Unable to create detector" );
       return 1;
     }

     // Report configuration problems before running
     if ( ! kv::check_nested_algo_configuration< kva::image_object_detector >(
              "detector", config ) )
     {
       LOG_ERROR( logger, "Configuration check failed." );
       return 1;
     }

     kv::detected_object_set_sptr detections = detector->detect( the_image );

     std::cout << "There were " << detections->size() << " detections in the image." << std::endl;

     return 0;
   }

The plugin manager loads every algorithm it can find, and the detector is then created
from the configuration. This file selects and configures the Hough circle detector:

::

   # select detector type
   detector:type = hough_circle

   # specify configuration for selected detector
   detector:hough_circle:dp = 1
   detector:hough_circle:min_dist = 100
   detector:hough_circle:param1 = 200
   detector:hough_circle:param2 = 100
   detector:hough_circle:min_radius = 0
   detector:hough_circle:max_radius = 0

The ``:`` character separates levels in a key. The first level, ``detector``, matches
the name passed to ``set_nested_algo_configuration`` in the program. ``detector:type``
names the algorithm to use, and the settings for that algorithm sit under its own name.
Selecting a different detector called ``foo`` would look like this:

::

   detector:type = foo
   detector:foo:param1 = 20
   detector:foo:param2 = 10

Because each algorithm's settings sit under its own name, settings for several
algorithms can share one file. This is how larger applications are configured.

******
Python
******

The same interfaces are available from python through the ``viame`` package, installed with
``pip install viame``. The steps below mirror the C++ ones above; see the `python
interface <https://viame.github.io/VIAME/sections/python_interface.html>`__ for the package as a whole.

Running a Single Detector
=========================

``viame.open`` loads an image from a file as an ``ImageContainer``. A detector is created by
name, and returns a ``DetectedObjectSet``:

.. code-block:: python

   import sys

   import viame
   from viame.algo import ImageObjectDetector

   # Read the image
   image = viame.open(sys.argv[1])

   # Create the detector
   detector = ImageObjectDetector.create("hough_circle")

   # Send image to detector and get detections
   detections = detector.detect(image)

   # See what was detected
   print("There were", len(detections), "detections in the image.")

   for detection in detections:
       box = detection.bounding_box
       print(box.min_x(), box.min_y(), box.max_x(), box.max_y(), detection.confidence)

Plugins are loaded the first time an algorithm is created, so no explicit call is needed. An
image already held in memory as a numpy array is passed as ``ImageContainer(Image(array))``,
with both types coming from ``viame.types``.

Logging
=======

Python code reports through the standard ``logging`` module:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger("detector_test")

   logger.info("There were %d detections in the image.", len(detections))

Messages from the algorithms themselves come from the C++ logger, whose level is set by the
``KWIVER_DEFAULT_LOG_LEVEL`` environment variable.

Configuration Support
=====================

``get_configuration()`` and ``set_configuration()`` work as they do in C++. The settings of an
algorithm can be listed, changed and applied in code:

.. code-block:: python

   config = detector.get_configuration()

   for key in config.available_values():
       print(key, "=", config.get_value(key))

   config.set_value("min_dist", "120")
   config.set_value("param1", "100")

   detector.set_configuration(config)

``detector1.py`` instead takes a configuration file as an optional second argument, in the
same format as the C++ program, and merges it over the defaults:

.. code-block:: python

   from viame.config import read_config_file

   config = detector.get_configuration()
   if len(sys.argv) > 2:
       config.merge_config(read_config_file(sys.argv[2]))
   detector.set_configuration(config)

Choosing the Detector at Run Time
=================================

``detector3.py`` reads which detector to use from the configuration file, which is the same
file ``detector3.cxx`` takes:

.. code-block:: python

   import sys

   import viame
   from viame.algo import ImageObjectDetector
   from viame.config import read_config_file

   image = viame.open(sys.argv[1])
   config = read_config_file(sys.argv[2])

   # Report configuration problems before running
   if not ImageObjectDetector.check_nested_algo_configuration("detector", config):
       sys.exit("Configuration check failed.")

   # Create the detector named in the configuration
   detector = ImageObjectDetector.set_nested_algo_configuration("detector", config)

   if detector is None:
       sys.exit("Unable to create detector")

   detections = detector.detect(image)

   print("There were", len(detections), "detections in the image.")

``ImageObjectDetector.registered_names()`` lists the detectors that can be named in the file.
Trackers, readers, writers and the other algorithm types in ``viame.algo`` are created and
configured the same way.

*****************************
Connecting Several Algorithms
*****************************

A real application usually does more than run one detector. Images may come from a video
or a camera, be filtered before detection, and have the results drawn or written to a
file afterwards. Writing that as a single program makes every change slow.

The pipeline framework in KWIVER, sprokit, handles this by connecting small processes in
a pipeline file. Each process runs one algorithm, configured with the same keys shown
above. See `new module creation
examples <https://viame.github.io/VIAME/sections/example_pipeline.html>`__ for
a pipeline that runs a detector, and `plugin
creation <https://viame.github.io/VIAME/sections/plugin_creation.html>`__ for
adding an algorithm of your own.


.. dive-crosslink

******************
DIVE Documentation
******************

* `DIVE pipeline import and export <https://viame.github.io/VIAME/sections/dive/Pipeline-Import-Export.html>`__ covers loading a custom pipeline into the interface, which is how an algorithm is run from DIVE rather than from code


********************
Code and Build Flags
********************

Flags to enable when building VIAME from source for this example:

* ``VIAME_ENABLE_KWIVER``
* ``VIAME_ENABLE_OPENCV``

Source code:

* examples/using_algorithms_in_code/detector1.cxx
* examples/using_algorithms_in_code/detector1.py
* examples/using_algorithms_in_code/detector3.cxx
* examples/using_algorithms_in_code/detector3.py
* examples/using_algorithms_in_code/detector4.cxx
* packages/kwiver/arrows/ocv/algo/hough_circle_detector.cxx
