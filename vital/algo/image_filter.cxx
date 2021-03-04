// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief image_filter algorithm instantiation
 */

#include <vital/algo/algorithm.txx>
#include <vital/algo/image_filter.h>

namespace kwiver {
namespace vital {
namespace algo {

image_filter
::image_filter()
{
  attach_logger( "algo.image_filter" ); // specify a logger
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::image_filter);
/// \endcond
