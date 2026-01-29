/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_NORMALIZE_IMAGE_PERCENTILE_H
#define VIAME_CORE_NORMALIZE_IMAGE_PERCENTILE_H

#include "viame_core_export.h"

#include <vital/algo/image_filter.h>

namespace viame {

/// @brief Percentile-based image normalization filter.
///
/// This filter normalizes image intensity values using percentile-based
/// min/max calculation. It supports arbitrary input types (8-bit, 16-bit,
/// float, etc.) and can output either 8-bit or native format.
///
/// The normalization formula is:
///   output = (input - p_low) / (p_high - p_low) * max_value
///
/// where p_low is the lower percentile value, p_high is the upper
/// percentile value, and max_value depends on the output format.
/// Values are clipped to the valid range for the output type.
///
/// @section config Configuration
///   - lower_percentile (double): Lower percentile for min value (default: 1.0)
///   - upper_percentile (double): Upper percentile for max value (default: 100.0)
///   - output_format (string): "8-bit" or "native" (default: "8-bit")
///     - 8-bit: Always output 8-bit unsigned (0-255)
///     - native: Output same type as input, stretched to full range
///
/// @section usage Pipeline Usage Example
/// @code
///   process filter
///     :: image_filter
///     :filter:type                                    percentile_norm
///     :filter:percentile_norm:lower_percentile        1.0
///     :filter:percentile_norm:upper_percentile        99.0
///     :filter:percentile_norm:output_format           native
/// @endcode
class VIAME_CORE_EXPORT normalize_image_percentile :
  public kwiver::vital::algo::image_filter
{
public:
  PLUGIN_INFO( "percentile_norm",
               "Percentile-based normalization with configurable output format" )

  normalize_image_percentile();
  virtual ~normalize_image_percentile();

  /// Get the current configuration for this filter.
  virtual kwiver::vital::config_block_sptr get_configuration() const;

  /// Apply configuration values to this filter.
  virtual void set_configuration( kwiver::vital::config_block_sptr config );

  /// Validate the configuration.
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  /// Main filtering method - performs percentile normalization.
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );

private:
  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace viame

#endif /* VIAME_CORE_NORMALIZE_IMAGE_PERCENTILE_H */
