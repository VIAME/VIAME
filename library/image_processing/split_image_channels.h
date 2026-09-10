// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for OCV split_image_channels algorithm

#ifndef KWIVER_ARROWS_OCV_SPLIT_IMAGE_CHANNELS_H_
#define KWIVER_ARROWS_OCV_SPLIT_IMAGE_CHANNELS_H_

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/split_image.h>

namespace kwiver {

namespace arrows {

namespace ocv {

/// Split an image into its channel planes.
///
/// Distinct from \ref split_image, which halves an image spatially. This one
/// emits one single-channel image per channel of the input, which is what a
/// pipeline stacking intensity/hue/motion planes consumes.
class VIAME_IMAGE_PROCESSING_EXPORT split_image_channels
  : public vital::algo::split_image
{
public:
  // Registered as "ocv_channels" in register_algorithms.cxx
  PLUGGABLE_IMPL(
    split_image_channels,
    "Split an image into multiple channel images (also known as planes)" )

  /// Destructor
  virtual ~split_image_channels();

  bool
  check_configuration(
    [[maybe_unused]] kwiver::vital::config_block_sptr config ) const override
  {
    return true;
  }

  /// Split image into its channel planes
  std::vector< kwiver::vital::image_container_sptr >
  split( kwiver::vital::image_container_sptr img ) const override;
};

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
