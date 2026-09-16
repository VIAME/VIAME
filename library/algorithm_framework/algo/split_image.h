// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_SPLIT_IMAGE_H_
#define VITAL_ALGO_SPLIT_IMAGE_H_

#include <viame/algorithm_framework/viame_compiler_config.h>

#include <string>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/image_container.h>

namespace viame {

namespace algo {

/// An abstract base class for converting base image type
class VIAME_ALGO_EXPORT split_image
  : public viame::algorithm
{
public:
  split_image();
  PLUGGABLE_INTERFACE( split_image );
  /// Split image
  virtual std::vector< viame::image_container_sptr >
  split( viame::image_container_sptr img ) const = 0;
};

typedef std::shared_ptr< split_image > split_image_sptr;

} // namespace algo

} // namespace viame

#endif // VITAL_ALGO_SPLIT_IMAGE_H_
