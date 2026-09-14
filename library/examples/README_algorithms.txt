================================================================================
                         VIAME Example Algorithms
================================================================================

Overview
--------
The hello world algorithms and processes are the smallest complete VIAME
plugins: each one declares its configuration, logs a configurable message and
hands its input on. They show how a new algorithm or process fits into the
pipeline system without any algorithm of their own to read past.

Two kinds are provided:
  1. Object Detector - Takes an image and produces a set of detections
  2. Image Filter    - Takes an image and produces a processed image

Each is written as a C++ algorithm and as a Python process. README.rst beside
this file covers the sprokit process template, which is the starting point
for a new C++ process rather than a new algorithm.


Directory Contents
------------------
  README_algorithms.txt        - This file
  README.rst                   - The C++ process template

  C++ algorithms:
    hello_world_detector.h     - Object detector
    hello_world_detector.cxx
    hello_world_filter.h       - Image filter
    hello_world_filter.cxx
    register_algorithms.cxx    - Registers both

  Python processes:
    hello_world_detector.py    - Object detector process
    hello_world_filter.py      - Image filter process
    __init__.py                - Declares both processes

  templates/cxx, templates/python
                               - Detector templates with @template@
                                 placeholders, for starting a new algorithm

  template_process.*, template_algo_wrapper.*, template_type_traits.h,
  register_processes.cxx, processes/
                               - The sprokit process template and sprokit's
                                 own worked example processes

  CMakeLists.txt               - Builds all of the above except templates/


Descriptions
------------
Hello World Detector:
  - Input:  image (kwiver:image)
  - Output: detected_object_set (kwiver:detected_object_set)
  - Config: text (string, default="Hello World")

  Receives an image and outputs an empty detection set, logging the
  configured text each time detect() is called.

Hello World Filter:
  - Input:  image (kwiver:image)
  - Output: out_image (kwiver:image)
  - Config: text (string, default="Hello World")

  Receives an image and outputs the same image unchanged, logging the
  configured text each time the filter is applied.


Using in a Pipeline
-------------------
  # The C++ detector
  process detector
    :: image_object_detector
    :detector:type                       hello_world
    :detector:hello_world:text           My Custom Message

  # The Python detector
  process py_detector
    :: hello_world_detector
    :text                                My Custom Message

  # The C++ filter
  process filter
    :: image_filter
    :filter:type                         hello_world_filter
    :filter:hello_world_filter:text      Filtering Image

  # The Python filter
  process py_filter
    :: hello_world_filter
    :text                                Filtering Image


Creating Your Own
-----------------
1. A C++ algorithm:
   a. Copy the .h and .cxx for the appropriate kind, or start from
      templates/cxx, into a directory under library/ named for what it does
   b. Rename the class and files to match your algorithm
   c. Declare its configuration with PARAM_DEFAULT entries in PLUGGABLE_IMPL;
      each becomes a config key with that default and a get_<name>() accessor
   d. Implement detect() or filter()
   e. Register it in register_algorithms.cxx with register_algorithm<>()
      from <viame/algorithm_framework/plugin/register_algorithm.h>
   f. Add the files to that directory's CMakeLists.txt

2. A Python process:
   a. Copy the .py for the appropriate kind
   b. Rename the class to match your process
   c. Declare configuration with add_config_trait() and ports with the
      declare_*_port_using_trait() calls in __init__()
   d. Implement _step()
   e. Add a line for it to __sprokit_process_declarations__ in the package's
      __init__.py: ( name, description, "module:Class" )

   A Python algorithm rather than a process starts from templates/python and
   is declared in __vital_algorithm_declarations__ the same way.


Key Concepts Demonstrated
-------------------------
1. Interfaces:
   - C++ algorithms derive from the kwiver::vital::algo interfaces
   - Python processes derive from KwiverProcess

2. Configuration:
   - Parameters declared once, with defaults and descriptions
   - Validated in check_configuration()

3. Ports (Python processes):
   - Input and output ports declared with type traits
   - Required and optional port flags

4. Data flow:
   - Grabbing data from input ports, processing it, pushing results

5. Logging through the KWIVER logger

6. Registration:
   - C++: register_algorithm<>() in register_algorithms.cxx
   - Python: declarations in __init__.py, which cost nothing until a
     pipeline asks for the process


Build Requirements
------------------
C++ algorithms need only the vital libraries. Python processes need
VIAME_ENABLE_PYTHON=ON.


Related Resources
-----------------
  - templates/     - Detector templates for new algorithms
  - README.rst     - The C++ process template
  - ../            - The other libraries, for real implementations of each
                     interface
  - KWIVER documentation: https://kwiver.readthedocs.io/
