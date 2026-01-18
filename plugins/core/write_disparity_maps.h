/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Disparity map visualization and writing algorithm
 */

#ifndef VIAME_CORE_WRITE_DISPARITY_MAPS_H
#define VIAME_CORE_WRITE_DISPARITY_MAPS_H

#include "viame_core_export.h"

#include <vital/algo/image_io.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

/**
 * \brief Image writer for disparity map visualization
 *
 * This algorithm converts disparity maps to colorized images for visualization
 * and uses an internal image_io algorithm to write the result to disk.
 *
 * Disparity maps can be in various formats:
 * - uint16 scaled by 256 (as produced by foundation_stereo_process)
 * - float32 (raw disparity values)
 * - uint8/uint16 grayscale
 *
 * Visualization options include:
 * - Various colormaps (jet, inferno, viridis, grayscale)
 * - Automatic or manual min/max range for normalization
 * - Invalid disparity handling
 */
class VIAME_CORE_EXPORT write_disparity_maps :
  public kwiver::vital::algo::image_io
{
public:
  PLUGGABLE_IMPL(
    write_disparity_maps,
    "Visualize and write disparity maps using colormaps. "
    "Converts disparity data to colorized images for visualization, "
    "then uses an internal image_io algorithm to write the result.",
    PARAM_DEFAULT(
      colormap, std::string,
      "Colormap to use for visualization. Options: jet, inferno, viridis, grayscale",
      "jet" ),
    PARAM_DEFAULT(
      min_disparity, double,
      "Minimum disparity value for normalization. Used when auto_range is false.",
      0.0 ),
    PARAM_DEFAULT(
      max_disparity, double,
      "Maximum disparity value for normalization. Used when auto_range is false.",
      0.0 ),
    PARAM_DEFAULT(
      auto_range, bool,
      "Automatically compute min/max disparity from the image data.",
      true ),
    PARAM_DEFAULT(
      disparity_scale, double,
      "Scale factor for disparity values. Set to 256.0 for uint16 disparity maps "
      "from foundation_stereo_process, or 1.0 for raw float disparity.",
      256.0 ),
    PARAM_DEFAULT(
      invalid_color, std::string,
      "RGB color for invalid disparity values (comma-separated, e.g., '0,0,0' for black)",
      "0,0,0" ),
    PARAM(
      image_writer, kwiver::vital::algo::image_io_sptr,
      "Algorithm pointer to nested image writer" )
  )

  virtual ~write_disparity_maps() = default;

  /// Check configuration validity
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

private:
  void initialize() override;

  void set_configuration_internal(
    kwiver::vital::config_block_sptr config ) override;

  /// Load is not supported - this is a write-only algorithm
  virtual kwiver::vital::image_container_sptr load_(
    std::string const& filename ) const override;

  /// Save disparity map with visualization
  virtual void save_(
    std::string const& filename,
    kwiver::vital::image_container_sptr data ) const override;

  // Parsed invalid color components
  uint8_t m_invalid_color_r;
  uint8_t m_invalid_color_g;
  uint8_t m_invalid_color_b;
};

} // namespace viame

#endif // VIAME_CORE_WRITE_DISPARITY_MAPS_H
