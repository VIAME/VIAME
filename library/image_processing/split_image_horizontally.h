// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for OCV split_image algorithm

#ifndef VIAME_IMAGE_PROCESSING_SPLIT_IMAGE_HORIZONTALLY_H
#define VIAME_IMAGE_PROCESSING_SPLIT_IMAGE_HORIZONTALLY_H

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/split_image.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

namespace viame {

/// A class for splitting an image in class horizontally.
class VIAME_IMAGE_PROCESSING_EXPORT split_image_horizontally
  : public viame::algo::split_image
{
public:
  PLUGGABLE_IMPL_NAMED(
    split_image_horizontally, "ocv_horizontally",
                  "Split an image  into multiple smaller images using opencv functions" )

  virtual ~split_image_horizontally() = default;

  virtual bool check_configuration( viame::config_block_sptr config ) const { return true; }

  /// Split image
  virtual std::vector< viame::image_container_sptr >
  split( viame::image_container_sptr img ) const;
};

} // end namespace viame

#endif
