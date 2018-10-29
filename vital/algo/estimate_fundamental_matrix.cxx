/*ckwg +29
 * Copyright 2014-2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief estimate fundamental matrix instantiation
 */

#include <vital/algo/estimate_fundamental_matrix.h>
#include <vital/algo/algorithm.txx>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::estimate_fundamental_matrix);
/// \endcond


namespace kwiver {
namespace vital {
namespace algo {


estimate_fundamental_matrix
::estimate_fundamental_matrix()
{
  attach_logger( "algo.estimate_fundamental_matrix" );
}


/// Estimate an fundamental matrix from corresponding features
fundamental_matrix_sptr
estimate_fundamental_matrix
::estimate(feature_set_sptr feat1,
           feature_set_sptr feat2,
           match_set_sptr matches,
           std::vector<bool>& inliers,
           double inlier_scale) const
{
  if( !feat1 || !feat2 || !matches )
  {
    return fundamental_matrix_sptr();
  }
  std::vector<feature_sptr> vf1 = feat1->features();
  std::vector<feature_sptr> vf2 = feat2->features();
  std::vector<match> mset = matches->matches();
  std::vector<vector_2d> vv1, vv2;

  for( match m : mset)
  {
    vv1.push_back(vf1[m.first]->loc());
    vv2.push_back(vf2[m.second]->loc());
  }
  return this->estimate(vv1, vv2, inliers, inlier_scale);
}

} } } // end namespace
