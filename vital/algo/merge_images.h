// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#pragma once

#include <vital/vital_config.h>

#include <string>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for converting base image type
class VITAL_ALGO_EXPORT merge_images
  : public kwiver::vital::algorithm_def<merge_images>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "merge_images"; }

  /// Merge images
  virtual kwiver::vital::image_container_sptr
    merge(kwiver::vital::image_container_sptr image1,
          kwiver::vital::image_container_sptr image2) const = 0;

protected:
  merge_images();

};

typedef std::shared_ptr<merge_images> merge_images_sptr;

} } } // end namespace

