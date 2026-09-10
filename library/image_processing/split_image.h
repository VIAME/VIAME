// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for OCV split_image algorithm

#ifndef KWIVER_ARROWS_OCV_SPLIT_IMAGE_H_
#define KWIVER_ARROWS_OCV_SPLIT_IMAGE_H_

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/split_image.h>

namespace kwiver {

namespace arrows {

namespace ocv {

/// A class for writing out image chips around detections, useful as a debugging
/// process
/// for ensuring that the refine detections process is running on desired ROIs.
class VIAME_IMAGE_PROCESSING_EXPORT split_image
  : public vital::algo::split_image
{
public:
  PLUGGABLE_IMPL(
    split_image,
    "Split an image  into multiple smaller images using opencv functions" )

  /// Destructor
  virtual ~split_image();

  bool
  check_configuration(
    [[maybe_unused]] kwiver::vital::config_block_sptr config ) const override
  {
    return true;
  }

  /// Split image
  std::vector< kwiver::vital::image_container_sptr >
  split( kwiver::vital::image_container_sptr img ) const override;
};

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
