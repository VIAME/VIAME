/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_PROCESSING_AVERAGE_FRAMES_H
#define VIAME_IMAGE_PROCESSING_AVERAGE_FRAMES_H

#include "viame_image_processing_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

namespace kv = kwiver::vital;

/// @brief Running average of the frames seen so far.
///
/// The motion pipelines run two of these at different window sizes and
/// difference them. Config keys, defaults and per-pixel results match the
/// `vxl_average` implementation it replaces, including its windowed mode
/// not being a sliding mean; see `image_ops/temporal.h`.
class VIAME_IMAGE_PROCESSING_EXPORT average_frames
  : public kv::algo::image_filter
{
public:
#define VIAME_AVERAGE_FRAMES_PARAMS \
    PARAM_DEFAULT( \
      type, std::string, \
      "Operating mode of this filter, possible values: window, " \
      "cumulative, exponential", \
      "window" ), \
    PARAM_DEFAULT( \
      window_size, unsigned, \
      "The window size if computing a windowed moving average.", \
      10 ), \
    PARAM_DEFAULT( \
      exp_weight, double, \
      "Exponential averaging coefficient if computing an exp average.", \
      0.3 ), \
    PARAM_DEFAULT( \
      round, bool, \
      "Should we spend a little extra time rounding when the input type " \
      "is an integer type?", \
      false ), \
    PARAM_DEFAULT( \
      output_variance, bool, \
      "If set, will compute an estimated variance for each pixel which " \
      "will be outputted as either a double or a float image.", \
      false )

  // PLUGGABLE_IMPL_NAMED rather than the pieces spelled out: it is
  // the only spelling that also generates get_configuration, without
  // which a partial config from a pipe file throws on the first key
  // the file does not set
  PLUGGABLE_IMPL_NAMED(
    average_frames,
    "average_frames",
    "Compute a running average of the input frames",
    VIAME_AVERAGE_FRAMES_PARAMS )

  virtual ~average_frames();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::image_container_sptr filter( kv::image_container_sptr image_data ) override;

  void set_configuration_internal( kv::config_block_sptr config ) override;

private:
  void initialize() override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif // VIAME_IMAGE_PROCESSING_AVERAGE_FRAMES_H
