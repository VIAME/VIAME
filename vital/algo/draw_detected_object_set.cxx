// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/algorithm.txx>
#include <vital/algo/draw_detected_object_set.h>

namespace kwiver {
namespace vital {
namespace algo {

draw_detected_object_set
::draw_detected_object_set()
{
  attach_logger( "algo.draw_detected_object_set" );
}

} } }

INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::draw_detected_object_set);
