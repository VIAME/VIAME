// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_SPLIT_IMAGE_H_
#define VITAL_ALGO_SPLIT_IMAGE_H_

#include <vital/vital_config.h>

#include <string>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for converting base image type
class VITAL_ALGO_EXPORT split_image
  : public kwiver::vital::algorithm_def<split_image>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "split_image"; }

  /// Split image
  virtual std::vector< kwiver::vital::image_container_sptr >
    split(kwiver::vital::image_container_sptr img) const = 0;

protected:
  split_image();

};

typedef std::shared_ptr<split_image> split_image_sptr;

} } } // end namespace

#endif // VITAL_ALGO_SPLIT_IMAGE_H_
