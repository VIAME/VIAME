// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief estimate_homography algorithm definition instantiation +
/// implementation

#include <viame/algorithm_framework/algo/estimate_homography.h>

namespace kwiver {

namespace vital {

namespace algo {

estimate_homography
::estimate_homography()
{
  attach_logger( "algo.estimate_homography" );
}

/// Estimate a homography matrix from corresponding features
homography_sptr
estimate_homography
::estimate(
  feature_set_sptr feat1,
  feature_set_sptr feat2,
  match_set_sptr matches,
  std::vector< bool >& inliers,
  double inlier_scale ) const
{
  if( !feat1 || !feat2 || !matches )
  {
    return homography_sptr();
  }

  std::vector< feature_sptr > vf1 = feat1->features();
  std::vector< feature_sptr > vf2 = feat2->features();
  std::vector< match > mset = matches->matches();
  std::vector< vector_2d > vv1, vv2;

  for( match m : mset )
  {
    vv1.push_back( vf1[ m.first ]->loc() );
    vv2.push_back( vf2[ m.second ]->loc() );
  }
  return this->estimate( vv1, vv2, inliers, inlier_scale );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
