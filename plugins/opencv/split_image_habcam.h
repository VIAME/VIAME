// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/master/LICENSE for details.

/// \file
/// \brief Header for HabCam auto split_image algorithm

#ifndef VIAME_OPENCV_SPLIT_IMAGE_HABCAM_H
#define VIAME_OPENCV_SPLIT_IMAGE_HABCAM_H

#include "viame_opencv_export.h"

#include <vital/algo/split_image.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

/// A class for splitting an image in class horizontally, only when needed.
class VIAME_OPENCV_EXPORT split_image_habcam
  : public kwiver::vital::algo::split_image
{
public:
  PLUGGABLE_IMPL(
    split_image_habcam,
    "Split an image into multiple smaller images",
    PARAM_DEFAULT( require_stereo, bool,
      "Fail if the input is not a conjoined stereo image pair", false ),
    PARAM_DEFAULT( required_width_factor, double,
      "If the width is this time as many heights, it is a stereo pair.", 2.0 )
  )

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const
  {
    return true;
  }

  /// Split image
  virtual std::vector< kwiver::vital::image_container_sptr >
  split( kwiver::vital::image_container_sptr img ) const;
};

} // end namespace viame

#endif // VIAME_OPENCV_SPLIT_IMAGE_HABCAM_H
