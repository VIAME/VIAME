// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for OCV split_image algorithm
 */

#ifndef KWIVER_ARROWS_OCV_SPLIT_IMAGE_H_
#define KWIVER_ARROWS_OCV_SPLIT_IMAGE_H_

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/split_image.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// A class for writing out image chips around detections, useful as a debugging process
/// for ensuring that the refine detections process is running on desired ROIs.
class KWIVER_ALGO_OCV_EXPORT split_image
  : public vital::algo::split_image
{
public:
  PLUGIN_INFO( "ocv",
               "Split an image  into multiple smaller images using opencv functions" )

  /// Constructor
  split_image();

  /// Destructor
  virtual ~split_image();

  virtual void set_configuration( kwiver::vital::config_block_sptr ) { }
  virtual bool check_configuration( kwiver::vital::config_block_sptr config) const { return true; }

  /// Split image
  virtual std::vector< kwiver::vital::image_container_sptr >
  split(kwiver::vital::image_container_sptr img) const;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
