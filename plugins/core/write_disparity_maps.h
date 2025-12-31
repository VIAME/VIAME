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

#include <memory>

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
  static constexpr char const* name = "write_disparity_maps";

  static constexpr char const* description =
    "Visualize and write disparity maps using colormaps. "
    "Converts disparity data to colorized images for visualization, "
    "then uses an internal image_io algorithm to write the result.";

  write_disparity_maps();
  virtual ~write_disparity_maps();

  /// Get the current configuration
  virtual kwiver::vital::config_block_sptr get_configuration() const override;

  /// Set configuration
  virtual void set_configuration( kwiver::vital::config_block_sptr config ) override;

  /// Check configuration validity
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

private:
  /// Load is not supported - this is a write-only algorithm
  virtual kwiver::vital::image_container_sptr load_(
    std::string const& filename ) const override;

  /// Save disparity map with visualization
  virtual void save_(
    std::string const& filename,
    kwiver::vital::image_container_sptr data ) const override;

  class priv;
  std::unique_ptr< priv > d;
};

} // namespace viame

#endif // VIAME_CORE_WRITE_DISPARITY_MAPS_H
