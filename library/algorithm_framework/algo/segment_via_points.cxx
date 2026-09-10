// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of segment_via_points algorithm

#include <viame/algorithm_framework/algo/segment_via_points.h>

namespace kwiver {

namespace vital {

namespace algo {

segment_via_points
::segment_via_points()
{
  attach_logger( "algo.segment_via_points" );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
