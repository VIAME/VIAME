// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief activity_detector algorithm instantiation
 */

#include <vital/algo/algorithm.txx>
#include <vital/algo/activity_detector.h>

namespace kwiver {
namespace vital {
namespace algo {

activity_detector
::activity_detector()
{
  attach_logger( "algo.activity_detector" ); // specify a logger
}

} } }

INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::activity_detector);
