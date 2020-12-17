// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of canonical similarity transform
 *        estimation algorithm definition.
 */

#include <vital/algo/algorithm.txx>
#include <vital/algo/estimate_canonical_transform.h>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::estimate_canonical_transform);
/// \endcond

namespace kwiver {
namespace vital {
namespace algo {

estimate_canonical_transform
::estimate_canonical_transform()
{
  attach_logger( "algo.estimate_canonical_transform" );
}

} } } // end namespace
