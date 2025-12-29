// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for OCV split_image algorithm

#ifndef VIAME_OPENCV_SPLIT_IMAGE_HORIZONTALLY_H
#define VIAME_OPENCV_SPLIT_IMAGE_HORIZONTALLY_H

#include <plugins/opencv/viame_opencv_export.h>

#include <vital/algo/split_image.h>

namespace viame {

/// A class for splitting an image in class horizontally.
class VIAME_OPENCV_EXPORT split_image_horizontally
  : public kwiver::vital::algo::split_image
{
public:
  PLUGIN_INFO( "ocv_horizontally",
               "Split an image  into multiple smaller images using opencv functions" )

  /// Constructor
  split_image_horizontally();

  /// Destructor
  virtual ~split_image_horizontally();

  virtual void set_configuration( kwiver::vital::config_block_sptr ) {}
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const { return true; }

  /// Split image
  virtual std::vector< kwiver::vital::image_container_sptr >
  split( kwiver::vital::image_container_sptr img ) const;
};

} // end namespace viame

#endif
