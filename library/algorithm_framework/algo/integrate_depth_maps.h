// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining abstract \link
///        kwiver::vital::algo::integrate_depth_maps
///        integrate_depth_maps \endlink algorithm

#ifndef VITAL_ALGO_INTEGRATE_DEPTH_MAPS_H_
#define VITAL_ALGO_INTEGRATE_DEPTH_MAPS_H_

#include <viame/algorithm_framework/vital_config.h>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/camera_perspective.h>
#include <viame/core_types/image_container.h>

#include <viame/core_types/vector.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for integration of depth maps into a volume
///
///  This algorithm takes a set of depth map images and a corresponding
///  set of cameras and integrates the depth maps into a 3D voxel grid such
///  that a level set (zero crossing) of the volumetric data is represents
///  the fused 3D model surface.
///
///  A common implementation of this algorithm is to integrate a truncated
///  signed distance function (TSDF) along a ray for each pixel of each
///  depth map.  However, this API is not restricted to TSDF.
class VITAL_ALGO_EXPORT integrate_depth_maps
  : public kwiver::vital::algorithm
{
public:
  integrate_depth_maps();
  PLUGGABLE_INTERFACE( integrate_depth_maps );
  /// Integrate multiple depth maps into a common volume
  ///
  /// \param [in]     minpt_bound the min point of the bounding region
  /// \param [in]     maxpt_bound the max point of the bounding region
  /// \param [in]     depth_maps  the set of floating point depth map images
  /// \param [in]     cameras     the set of cameras, one for each depth map
  /// \param [in,out] volume      the fused volumetric data
  /// \param [out]    spacing     the spacing between voxels in each dimension
  virtual void
  integrate(
    kwiver::vital::vector_3d const& minpt_bound,
    kwiver::vital::vector_3d const& maxpt_bound,
    std::vector< kwiver::vital::image_container_sptr > const& depth_maps,
    std::vector< kwiver::vital::camera_perspective_sptr > const& cameras,
    kwiver::vital::image_container_sptr& volume,
    kwiver::vital::vector_3d& spacing ) const;

  /// Integrate multiple depth maps with per-pixel weights into a common volume
  ///
  /// The weight maps in this variant encode how much weight to give each depth
  /// pixel in the integration sum.  If the vector of weight_maps is empty then
  /// all depths are given full weight.
  ///
  /// \param [in]     minpt_bound the min point of the bounding region
  /// \param [in]     maxpt_bound the max point of the bounding region
  /// \param [in]     depth_maps  the set of floating point depth map images
  /// \param [in]     weight_maps the set of floating point [0,1] weight maps
  /// \param [in]     cameras     the set of cameras, one for each depth map
  /// \param [in,out] volume      the fused volumetric data
  /// \param [out]    spacing     the spacing between voxels in each dimension
  virtual void
  integrate(
    kwiver::vital::vector_3d const& minpt_bound,
    kwiver::vital::vector_3d const& maxpt_bound,
    std::vector< kwiver::vital::image_container_sptr > const& depth_maps,
    std::vector< kwiver::vital::image_container_sptr > const& weight_maps,
    std::vector< kwiver::vital::camera_perspective_sptr > const& cameras,
    kwiver::vital::image_container_sptr& volume,
    kwiver::vital::vector_3d& spacing ) const = 0;
};

/// type definition for shared pointer to a bundle adjust algorithm
typedef std::shared_ptr< integrate_depth_maps > integrate_depth_maps_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_INTEGRATE_DEPTH_MAPS_H_
