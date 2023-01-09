// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/master/LICENSE for details.

/// \file
/// \brief Header for HabCam auto split_image algorithm

#ifndef VIAME_OPENCV_SPLIT_IMAGE_HABCAM_H
#define VIAME_OPENCV_SPLIT_IMAGE_HABCAM_H

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/split_image.h>

namespace viame {

/// A class for splitting an image in class horizontally, only when needed.
class KWIVER_ALGO_OCV_EXPORT split_image_habcam
  : public kwiver::vital::algo::split_image
{
public:
  PLUGIN_INFO( "habcam",
               "Split an image into multiple smaller images" )

  /// Constructor
  split_image_habcam();

  /// Destructor
  virtual ~split_image_habcam();

  virtual void set_configuration( kwiver::vital::config_block_sptr ) {}
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const { return true; }

  /// Split image
  virtual std::vector< kwiver::vital::image_container_sptr >
  split( kwiver::vital::image_container_sptr img ) const;
};

} // end namespace viame

#endif // VIAME_OPENCV_SPLIT_IMAGE_HABCAM_H
