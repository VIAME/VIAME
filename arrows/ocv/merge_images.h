// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for OCV merge_images algorithm
 */

#pragma once

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/merge_images.h>

namespace kwiver {
namespace arrows {
namespace ocv {

// Implementation of merge image channels.
class KWIVER_ALGO_OCV_EXPORT merge_images
  : public vital::algo::merge_images
{
public:
  PLUGIN_INFO( "ocv",
               "Merge two images into one using opencv functions.\n\n"
               "The channels from the first image are added to the "
               "output image first, followed by the channels from the "
               "second image. This implementation takes no configuration "
               "parameters."
    )

  /// Constructor
  merge_images();

  /// Destructor
  virtual ~merge_images() = default;

  void set_configuration( kwiver::vital::config_block_sptr ) override { }
  bool check_configuration( kwiver::vital::config_block_sptr config ) const override
  { return true; }

  /// Merge images
  kwiver::vital::image_container_sptr
    merge(kwiver::vital::image_container_sptr image1,
          kwiver::vital::image_container_sptr image2) const override;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
