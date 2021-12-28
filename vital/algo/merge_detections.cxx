// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/algorithm.txx>

#include "merge_detections.h"

INSTANTIATE_ALGORITHM_DEF( kwiver::vital::algo::merge_detections );

namespace kwiver {
namespace vital {
namespace algo {

merge_detections
::merge_detections()
{
  attach_logger( "merge_detections" );
}

} } } // end namespace
