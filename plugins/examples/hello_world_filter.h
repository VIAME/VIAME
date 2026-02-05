/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_HELLO_WORLD_FILTER_H
#define VIAME_HELLO_WORLD_FILTER_H

#include "viame_examples_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

// =============================================================================
/**
 * @brief Example image filter demonstrating C++ plugin development.
 *
 * This example shows how to create an image filter in C++ that integrates
 * with the KWIVER/VIAME pipeline system. It receives images as input and
 * produces processed images as output.
 *
 * The implementation is intentionally simple - it only logs a configurable
 * text message and returns the input image unchanged. Use this as a template
 * when creating your own image processing filters.
 *
 * @section config Configuration
 *   - text (string): Message to display when processing each image.
 *                    Default: "Hello World"
 *
 * @section usage Pipeline Usage Example
 * @code
 *   process filter
 *     :: image_filter
 *     :filter:type                         hello_world_filter
 *     :filter:hello_world_filter:text      Filtering image...
 * @endcode
 *
 * @section howto Creating Your Own Filter
 *   1. Copy this .h and .cxx file and rename appropriately
 *   2. Update the class name, namespace, and include guards
 *   3. Implement your image processing logic in filter()
 *   4. Add configuration parameters in PLUGGABLE_IMPL
 *   5. Register your algorithm in register_algorithms.cxx
 *   6. Update CMakeLists.txt to include your new files
 *
 * @see hello_world_detector for an object detector example
 */
class VIAME_EXAMPLES_EXPORT hello_world_filter :
  public kwiver::vital::algo::image_filter
{
public:
  PLUGGABLE_IMPL(
    hello_world_filter,
    "Example hello world image filter",
    PARAM_DEFAULT(
      text, std::string,
      "Message to display when processing each image.",
      "Hello World" )
  )

  virtual ~hello_world_filter();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );
};

} // end namespace viame

#endif /* VIAME_HELLO_WORLD_FILTER_H */
