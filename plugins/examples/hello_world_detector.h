/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_HELLO_WORLD_DETECTOR_H
#define VIAME_HELLO_WORLD_DETECTOR_H

#include "viame_examples_export.h"

#include <vital/algo/image_object_detector.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

// =============================================================================
/**
 * @brief Example object detector demonstrating C++ plugin development.
 *
 * This example shows how to create an object detector in C++ that integrates
 * with the KWIVER/VIAME pipeline system. It receives images as input and
 * produces detection sets as output.
 *
 * The implementation is intentionally simple - it only logs a configurable
 * text message and returns an empty detection set. Use this as a template
 * when creating your own detection algorithms.
 *
 * @section config Configuration
 *   - text (string): Message to display when processing each image.
 *                    Default: "Hello World"
 *
 * @section usage Pipeline Usage Example
 * @code
 *   process detector
 *     :: image_object_detector
 *     :detector:type                       hello_world
 *     :detector:hello_world:text           My Custom Message
 * @endcode
 *
 * @section howto Creating Your Own Detector
 *   1. Copy this .h and .cxx file and rename appropriately
 *   2. Update the class name, namespace, and include guards
 *   3. Implement your detection logic in detect()
 *   4. Add configuration parameters via PARAM_DEFAULT macros
 *   5. Register your algorithm in register_algorithms.cxx
 *   6. Update CMakeLists.txt to include your new files
 *
 * @see hello_world_filter for an image filter example
 */
class VIAME_EXAMPLES_EXPORT hello_world_detector :
  public kwiver::vital::algo::image_object_detector
{
public:
#define VIAME_EXAMPLES_HWD_PARAMS \
    PARAM_DEFAULT( \
      text, std::string, \
      "Message to display when processing each image.", \
      "Hello World" )

  PLUGGABLE_VARIABLES( VIAME_EXAMPLES_HWD_PARAMS )
  PLUGGABLE_CONSTRUCTOR( hello_world_detector, VIAME_EXAMPLES_HWD_PARAMS )

  static std::string plugin_name() { return "hello_world"; }
  static std::string plugin_description() { return "Example hello world detector"; }

  PLUGGABLE_STATIC_FROM_CONFIG( hello_world_detector, VIAME_EXAMPLES_HWD_PARAMS )
  PLUGGABLE_STATIC_GET_DEFAULT( VIAME_EXAMPLES_HWD_PARAMS )
  PLUGGABLE_SET_CONFIGURATION( hello_world_detector, VIAME_EXAMPLES_HWD_PARAMS )

  virtual ~hello_world_detector();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;
};

} // end namespace viame

#endif /* VIAME_HELLO_WORLD_DETECTOR_H */
