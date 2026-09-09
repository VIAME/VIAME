// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#pragma once

#include <viame/algorithm_framework/vital_config.h>

#include <string>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/image_container.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for converting base image type
class VITAL_ALGO_EXPORT merge_images
  : public kwiver::vital::algorithm
{
public:
  merge_images();
  PLUGGABLE_INTERFACE( merge_images );
  /// Merge images
  virtual kwiver::vital::image_container_sptr
  merge(
    kwiver::vital::image_container_sptr image1,
    kwiver::vital::image_container_sptr image2 ) const = 0;
};

typedef std::shared_ptr< merge_images > merge_images_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver
