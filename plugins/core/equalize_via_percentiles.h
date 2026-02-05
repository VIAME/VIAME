/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_EQUALIZE_VIA_PERCENTILES_H
#define VIAME_CORE_EQUALIZE_VIA_PERCENTILES_H

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
///     :filter:type                                    equalize_via_percentiles
///     :filter:equalize_via_percentiles:lower_percentile        1.0
///     :filter:equalize_via_percentiles:upper_percentile        99.0
///     :filter:equalize_via_percentiles:output_format           native
/// @endcode
class VIAME_CORE_EXPORT equalize_via_percentiles :
  public kwiver::vital::algo::image_filter
{
public:
  PLUGGABLE_IMPL(
    equalize_via_percentiles,
    "Percentile-based normalization with configurable output format",
    PARAM_DEFAULT(
      lower_percentile, double,
      "Lower percentile for minimum value calculation (0.0 to 100.0). "
      "Default is 1.0 to exclude outliers.",
      1.0 ),
    PARAM_DEFAULT(
      upper_percentile, double,
      "Upper percentile for maximum value calculation (0.0 to 100.0). "
      "Default is 100.0 (maximum value).",
      100.0 ),
    PARAM_DEFAULT(
      output_format, std::string,
      "Output format: '8-bit' for 8-bit unsigned output, 'native' for same "
      "type as input with values stretched to full range. Default is '8-bit'.",
      "8-bit" )
  )

  virtual ~equalize_via_percentiles();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );
};

} // end namespace viame

#endif /* VIAME_CORE_EQUALIZE_VIA_PERCENTILES_H */
