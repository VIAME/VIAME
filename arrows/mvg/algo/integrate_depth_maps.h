// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Header file for integration of depth maps
*/

#ifndef KWIVER_ARROWS_MVG_INTEGRATE_DEPTH_MAPS_H_
#define KWIVER_ARROWS_MVG_INTEGRATE_DEPTH_MAPS_H_

#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vital/algo/integrate_depth_maps.h>
#include <vital/vital_config.h>
#include <vital/types/vector.h>

namespace kwiver {
namespace arrows {
namespace mvg {

class KWIVER_ALGO_MVG_EXPORT integrate_depth_maps
  : public vital::algo::integrate_depth_maps
{
public:
  PLUGIN_INFO( "mvg",
               "CPU depth map fusion" )

  /// Constructor
  integrate_depth_maps();

  /// Destructor
  virtual ~integrate_depth_maps();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

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
  ///
  /// \note the volume data is stored as a 3D image.  Metadata fields on the
  /// image specify the origin and scale of the volume in world coordinates.
  virtual void
    integrate(kwiver::vital::vector_3d const& minpt_bound,
              kwiver::vital::vector_3d const& maxpt_bound,
              std::vector<kwiver::vital::image_container_sptr> const& depth_maps,
              std::vector<kwiver::vital::image_container_sptr> const& weight_maps,
              std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
              kwiver::vital::image_container_sptr& volume,
              kwiver::vital::vector_3d &spacing) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

}  // end namespace mvg
}  // end namespace arrows
}  // end namespace kwiver

#endif
