/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_VIDEO_IO_CORE_IMAGE_IO_H
#define VIAME_VIDEO_IO_CORE_IMAGE_IO_H

#include "viame_video_io_export.h"

#include <vital/algo/image_io.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

namespace kv = kwiver::vital;

/// @brief Read and write images, with control over depth and range.
///
/// Decoding is delegated to another image_io; what this adds is the range
/// handling the pipelines rely on. Most of them read 16 bit sensor frames
/// with `force_byte` set, and expect 8 bit images out.
///
/// Config keys, defaults and results match the `vxl` image_io it replaces.
/// The plain `ocv` reader is not a substitute for it: that one has no
/// configuration at all, so `force_byte` and the stretch options would be
/// silently dropped.
class VIAME_VIDEO_IO_EXPORT core_image_io
  : public kv::algo::image_io
{
public:
#define VIAME_CORE_IMAGE_IO_PARAMS \
    PARAM_DEFAULT( \
      force_byte, bool, \
      "When loading, convert the image to 8 bit regardless of what the " \
      "file holds", \
      false ), \
    PARAM_DEFAULT( \
      auto_stretch, bool, \
      "Stretch the image's own intensity range across the output range", \
      false ), \
    PARAM_DEFAULT( \
      manual_stretch, bool, \
      "Stretch the range given by intensity_range across the output range", \
      false ), \
    PARAM_DEFAULT( \
      intensity_range, std::string, \
      "The intensity range to stretch, as 'low high', when manual_stretch " \
      "is set", \
      "0 255" ), \
    PARAM_DEFAULT( \
      split_channels, bool, \
      "Read and write each channel as its own file, named after the first " \
      "with _1, _2 and so on appended", \
      false )

  // PLUGGABLE_IMPL_NAMED rather than the pieces spelled out: it is
  // the only spelling that also generates get_configuration, without
  // which a partial config from a pipe file throws on the first key
  // the file does not set
  PLUGGABLE_IMPL_NAMED(
    core_image_io,
    "core",
    "Read and write images with depth and range control",
    VIAME_CORE_IMAGE_IO_PARAMS )

  virtual ~core_image_io();

  bool check_configuration( kv::config_block_sptr config ) const override;

  void set_configuration_internal( kv::config_block_sptr config ) override;

private:
  void initialize() override;

  kv::image_container_sptr load_( std::string const& filename ) const override;

  void save_( std::string const& filename,
              kv::image_container_sptr data ) const override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif // VIAME_VIDEO_IO_CORE_IMAGE_IO_H
