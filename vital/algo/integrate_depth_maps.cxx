/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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
 * \brief Instantiation of \link
 *        kwiver::vital::algo::algorithm_def algorithm_def<T> \endlink for \link
 *        kwiver::vital::algo::integrate_depth_maps integrate_depth_maps \endlink
 */

#include <vital/algo/integrate_depth_maps.h>
#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

integrate_depth_maps
::integrate_depth_maps()
{
  attach_logger( "algo.integrate_depth_maps" );
}

/// Integrate multiple depth maps into a common volume
/**
*
* \param [in]     minpt_bound the min point of the bounding region
* \param [in]     maxpt_bound the max point of the bounding region
* \param [in]     depth_maps  the set of floating point depth map images
* \param [in]     cameras     the set of cameras, one for each depth map
* \param [in,out] volume      the fused volumetric data
* \param [out]    spacing     the spacing between voxels in each dimension
*/
void
integrate_depth_maps
::integrate(vector_3d const& minpt_bound,
            vector_3d const& maxpt_bound,
            std::vector<image_container_sptr> const& depth_maps,
            std::vector<camera_perspective_sptr> const& cameras,
            image_container_sptr& volume,
            vector_3d &spacing) const
{
  // call the weighted version, but leave the weights empty
  this->integrate(minpt_bound, maxpt_bound, depth_maps, {},
                  cameras, volume, spacing);
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::integrate_depth_maps);
/// \endcond
