/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_HELLO_WORLD_FILTER_H
#define VIAME_HELLO_WORLD_FILTER_H

#include "viame_examples_export.h"

#include <vital/algo/image_filter.h>

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
 *   4. Add configuration parameters in get_configuration()
 *   5. Register your algorithm in register_algorithms.cxx
 *   6. Update CMakeLists.txt to include your new files
 *
 * @see hello_world_detector for an object detector example
 */
class VIAME_EXAMPLES_EXPORT hello_world_filter :
  public kwiver::vital::algo::image_filter
{
public:
  /// Algorithm registration name used in pipeline configuration
  static constexpr char const* name = "hello_world_filter";

  /// Human-readable description shown in algorithm listings
  static constexpr char const* description = "Example hello world image filter";

  hello_world_filter();
  virtual ~hello_world_filter();

  /**
   * @brief Get the current configuration for this filter.
   *
   * Returns a config_block containing all configuration parameters
   * with their current values and descriptions. The base class
   * configuration is included automatically.
   *
   * @return Configuration block with current parameter values
   */
  virtual kwiver::vital::config_block_sptr get_configuration() const;

  /**
   * @brief Apply configuration values to this filter.
   *
   * Called by the pipeline system after parsing configuration files.
   * Extract your parameter values from the config block here.
   *
   * @param config Configuration block containing parameter values
   */
  virtual void set_configuration( kwiver::vital::config_block_sptr config );

  /**
   * @brief Validate the configuration.
   *
   * Called to verify that configuration values are valid before
   * processing begins. Return false and log an error if invalid.
   *
   * @param config Configuration block to validate
   * @return true if configuration is valid, false otherwise
   */
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  /**
   * @brief Main filtering method - implement your algorithm here.
   *
   * This is the core method where your image processing runs.
   * It receives an input image and should return the processed image.
   *
   * @param image_data Input image to process
   * @return Processed output image
   */
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );

private:
  /// Private implementation class (Pimpl pattern for binary compatibility)
  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace viame

#endif /* VIAME_HELLO_WORLD_FILTER_H */
