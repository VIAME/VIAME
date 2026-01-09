================================================================================
                         VIAME Example Plugins
================================================================================

Overview
--------
This directory contains simple, well-documented example implementations of
VIAME/KWIVER plugins. These examples demonstrate the fundamental patterns for
creating custom image processing algorithms that integrate with the VIAME
pipeline system.

Two example plugin types are provided:
  1. Object Detector - Takes an image and produces a set of detections
  2. Image Filter    - Takes an image and produces a processed image

Each example is implemented in both C++ and Python to demonstrate the patterns
for each language. The implementations are intentionally simple (they only log
a configurable text message) so developers can focus on understanding the
framework integration rather than algorithm complexity.


Directory Contents
------------------
  README.txt                   - This documentation file

  C++ Implementation:
    hello_world_detector.h     - Object detector header (interface definition)
    hello_world_detector.cxx   - Object detector implementation
    hello_world_filter.h       - Image filter header (interface definition)
    hello_world_filter.cxx     - Image filter implementation
    register_algorithms.cxx    - Plugin registration for C++ algorithms

  Python Implementation:
    hello_world_detector.py    - Object detector implementation
    hello_world_filter.py      - Image filter implementation
    __init__.py                - Python module initialization and registration

  Build Configuration:
    CMakeLists.txt             - CMake build configuration


Plugin Descriptions
-------------------
Hello World Detector:
  - Input:  image (kwiver:image)
  - Output: detected_object_set (kwiver:detected_object_set)
  - Config: text (string, default="Hello World")

  This example detector receives an image and outputs an empty detection set.
  It logs the configured text message each time the detect() method is called.
  Use this as a template when creating new object detection algorithms.

Hello World Filter:
  - Input:  image (kwiver:image)
  - Output: out_image (kwiver:image)
  - Config: text (string, default="Hello World")

  This example filter receives an image and outputs the same image unchanged.
  It logs the configured text message each time the filter is applied.
  Use this as a template when creating new image processing filters.


Using in a Pipeline
-------------------
These examples can be used in KWIVER/VIAME pipelines. Example pipeline snippet:

  # Using the C++ detector
  process detector
    :: image_object_detector
    :detector:type                       hello_world
    :detector:hello_world:text           My Custom Message

  # Using the Python detector
  process py_detector
    :: hello_world_detector
    :text                                My Custom Message

  # Using the C++ filter
  process filter
    :: image_filter
    :filter:type                         hello_world_filter
    :filter:hello_world_filter:text      Filtering Image

  # Using the Python filter
  process py_filter
    :: hello_world_filter
    :text                                Filtering Image


Creating Your Own Plugin
------------------------
To create a new plugin based on these examples:

1. For C++ Plugins:
   a. Copy the .h and .cxx files for the appropriate type (detector/filter)
   b. Rename the class and files to match your algorithm name
   c. Update the namespace and include guards
   d. Implement your algorithm logic in the detect() or filter() method
   e. Add configuration parameters as needed in get_configuration()
   f. Register your algorithm in register_algorithms.cxx
   g. Update CMakeLists.txt to include your new files

2. For Python Plugins:
   a. Copy the .py file for the appropriate type (detector/filter)
   b. Rename the class to match your algorithm name
   c. Implement your algorithm logic in the _step() method
   d. Add configuration parameters using add_config_trait()
   e. Register your process in __init__.py

See also the 'templates' directory for additional code templates that provide
more extensive boilerplate for common plugin patterns.


Key Concepts Demonstrated
-------------------------
These examples demonstrate several important KWIVER/VIAME concepts:

1. Algorithm Interface Pattern:
   - C++ plugins inherit from kwiver::vital::algo base classes
   - Python plugins inherit from KwiverProcess

2. Configuration System:
   - Declaring configuration parameters with defaults and descriptions
   - Reading configuration values in set_configuration() / _configure()
   - Validating configuration in check_configuration()

3. Port Declaration:
   - Declaring input/output ports with type traits
   - Using required vs optional port flags

4. Data Flow:
   - Grabbing data from input ports
   - Processing data (your algorithm goes here)
   - Pushing results to output ports

5. Logging:
   - Using the KWIVER logging system for debug/info/error messages

6. Plugin Registration:
   - C++: Using kwiver::vital::algo::algorithm_factory_manager
   - Python: Using add_process() in __init__.py

7. Pimpl Pattern (C++ only):
   - Using private implementation class for binary compatibility
   - Storing member variables in the private class


Build Requirements
------------------
C++ plugins require:
  - KWIVER libraries (vital, vital_algo, vital_config, vital_logger)
  - OpenCV (for image handling)

Python plugins require:
  - VIAME_ENABLE_PYTHON=ON during CMake configuration
  - kwiver Python bindings installed


Related Resources
-----------------
  - templates/     - Additional code templates for common patterns
  - ../core/       - Production plugins for reference
  - ../opencv/     - OpenCV-based plugins for reference
  - KWIVER documentation: https://kwiver.readthedocs.io/

