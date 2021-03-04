// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
