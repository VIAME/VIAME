// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief detected_object_filter algorithm instantiation
 */

#include <vital/algo/algorithm.txx>
#include <vital/algo/detected_object_filter.h>

namespace kwiver {
namespace vital {
namespace algo {

detected_object_filter
::detected_object_filter()
{
  attach_logger( "algo.detected_object_filter" );
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::detected_object_filter);
/// \endcond
