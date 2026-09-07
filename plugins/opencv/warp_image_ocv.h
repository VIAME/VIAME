/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Header for OCV warp_image algorithm
 */

#ifndef VIAME_OPENCV_WARP_IMAGE_OCV_H
#define VIAME_OPENCV_WARP_IMAGE_OCV_H

#include "viame_opencv_export.h"

#include <vital/algo/warp_image.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

/// A class for warping an image onto another with a homography.
class VIAME_OPENCV_EXPORT warp_image_ocv
  : public kwiver::vital::algo::warp_image
{
public:
  PLUGGABLE_IMPL_NAMED(
    warp_image_ocv, "ocv",
    "Warp an image onto another with a homography using opencv functions" )

  virtual ~warp_image_ocv() = default;

  virtual bool check_configuration(
    kwiver::vital::config_block_sptr config ) const { return true; }

  /// Warp src_image onto dst_image
  virtual kwiver::vital::image_container_sptr warp(
    kwiver::vital::image_container_sptr src_image,
    kwiver::vital::image_container_sptr dst_image,
    kwiver::vital::homography_sptr homography,
    kwiver::vital::image_container_sptr alpha_mask = nullptr ) const;
};

} // end namespace viame

#endif
