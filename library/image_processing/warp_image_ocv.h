/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Header for OCV warp_image algorithm
 */

#ifndef VIAME_IMAGE_PROCESSING_WARP_IMAGE_OCV_H
#define VIAME_IMAGE_PROCESSING_WARP_IMAGE_OCV_H

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/warp_image.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

namespace viame {

/// A class for warping an image onto another with a homography.
class VIAME_IMAGE_PROCESSING_EXPORT warp_image_ocv
  : public viame::algo::warp_image
{
public:
  PLUGGABLE_IMPL_NAMED(
    warp_image_ocv, "ocv",
    "Warp an image onto another with a homography using opencv functions" )

  virtual ~warp_image_ocv() = default;

  virtual bool check_configuration(
    viame::config_block_sptr config ) const { return true; }

  /// Warp src_image onto dst_image
  virtual viame::image_container_sptr warp(
    viame::image_container_sptr src_image,
    viame::image_container_sptr dst_image,
    viame::homography_sptr homography,
    viame::image_container_sptr alpha_mask = nullptr ) const;
};

} // end namespace viame

#endif
