// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/algorithm.txx>
#include <vital/algo/compute_ref_homography.h>

namespace kwiver {
namespace vital {
namespace algo {

compute_ref_homography
::compute_ref_homography()
{
  attach_logger( "algo.compute_ref_homography" );
}

} } }

INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::compute_ref_homography);
