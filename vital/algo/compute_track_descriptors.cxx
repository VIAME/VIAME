// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief compute_track_descriptors algorithm instantiation
 */

#include <vital/algo/compute_track_descriptors.h>
#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

compute_track_descriptors
::compute_track_descriptors()
{
  attach_logger( "algo.compute_track_descriptors" );
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF( kwiver::vital::algo::compute_track_descriptors );
/// \endcond
