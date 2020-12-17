// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief refine_detections algorithm instantiation
 */

#include <vital/algo/algorithm.txx>
#include <vital/algo/refine_detections.h>

namespace kwiver {
namespace vital {
namespace algo {

refine_detections
::refine_detections()
{
  attach_logger( "algo.refine_detections" ); // specify a logger
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::refine_detections);
/// \endcond
