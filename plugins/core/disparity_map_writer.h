/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Disparity map visualization and writing algorithm
 */

#ifndef VIAME_DISPARITY_MAP_WRITER_H
#define VIAME_DISPARITY_MAP_WRITER_H

#include <plugins/core/viame_core_export.h>

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
class VIAME_CORE_EXPORT disparity_map_writer :
  public kwiver::vital::algo::image_io
{
public:
  static constexpr char const* name = "disparity_map";

  static constexpr char const* description =
    "Visualize and write disparity maps using colormaps. "
    "Converts disparity data to colorized images for visualization, "
    "then uses an internal image_io algorithm to write the result.";

  disparity_map_writer();
  virtual ~disparity_map_writer();

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

#endif // VIAME_DISPARITY_MAP_WRITER_H
