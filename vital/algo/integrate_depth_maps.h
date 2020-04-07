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
 * \brief Header defining abstract \link kwiver::vital::algo::integrate_depth_maps
 *        integrate_depth_maps \endlink algorithm
 */

#ifndef VITAL_ALGO_INTEGRATE_DEPTH_MAPS_H_
#define VITAL_ALGO_INTEGRATE_DEPTH_MAPS_H_

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/image_container.h>

#include <vital/types/vector.h>


namespace kwiver {
namespace vital {
namespace algo {


/// An abstract base class for integration of depth maps into a volume
/**
 *  This algorithm takes a set of depth map images and a corresponding
 *  set of cameras and integrates the depth maps into a 3D voxel grid such
 *  that a level set (zero crossing) of the volumetric data is represents
 *  the fused 3D model surface.
 *
 *  A common implementation of this algorithm is to integrate a truncated
 *  signed distance function (TSDF) along a ray for each pixel of each
 *  depth map.  However, this API is not restricted to TSDF.
 */
class VITAL_ALGO_EXPORT integrate_depth_maps
  : public kwiver::vital::algorithm_def<integrate_depth_maps>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "integrate_depth_maps"; }

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
  virtual void
    integrate(kwiver::vital::vector_3d const& minpt_bound,
              kwiver::vital::vector_3d const& maxpt_bound,
              std::vector<kwiver::vital::image_container_sptr> const& depth_maps,
              std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
              kwiver::vital::image_container_sptr& volume,
              kwiver::vital::vector_3d &spacing) const;

  /// Integrate multiple depth maps with per-pixel weights into a common volume
  /**
  * The weight maps in this variant encode how much weight to give each depth
  * pixel in the integration sum.  If the vector of weight_maps is empty then
  * all depths are given full weight.
  *
  * \param [in]     minpt_bound the min point of the bounding region
  * \param [in]     maxpt_bound the max point of the bounding region
  * \param [in]     depth_maps  the set of floating point depth map images
  * \param [in]     weight_maps the set of floating point [0,1] weight maps
  * \param [in]     cameras     the set of cameras, one for each depth map
  * \param [in,out] volume      the fused volumetric data
  * \param [out]    spacing     the spacing between voxels in each dimension
  */
  virtual void
    integrate(kwiver::vital::vector_3d const& minpt_bound,
      kwiver::vital::vector_3d const& maxpt_bound,
      std::vector<kwiver::vital::image_container_sptr> const& depth_maps,
      std::vector<kwiver::vital::image_container_sptr> const& weight_maps,
      std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
      kwiver::vital::image_container_sptr& volume,
      kwiver::vital::vector_3d &spacing) const = 0;

protected:
  integrate_depth_maps();
};


/// type definition for shared pointer to a bundle adjust algorithm
typedef std::shared_ptr<integrate_depth_maps> integrate_depth_maps_sptr;

} } } // end namespace

#endif // VITAL_ALGO_INTEGRATE_DEPTH_MAPS_H_
