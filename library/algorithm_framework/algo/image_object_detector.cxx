// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief image_object_detector algorithm instantiation

#include <viame/algorithm_framework/algo/image_object_detector.h>

namespace kwiver {

namespace vital {

namespace algo {

image_object_detector
::image_object_detector()
{
  attach_logger( "algo.image_object_detector" ); // specify a logger
}

} // namespace algo

} // namespace vital

} // namespace kwiver
