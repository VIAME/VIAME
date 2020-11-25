// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Header file for compute depth
*/

#ifndef KWIVER_ARROWS_SUPER3D_COMPUTE_DEPTH_H_
#define KWIVER_ARROWS_SUPER3D_COMPUTE_DEPTH_H_

#include <arrows/super3d/kwiver_algo_super3d_export.h>

#include <vital/algo/compute_depth.h>
#include <vital/vital_config.h>
#include <vital/types/vector.h>

namespace kwiver {
namespace arrows {
namespace super3d {

/// A class for bundle adjustment of feature tracks using VXL
class KWIVER_ALGO_SUPER3D_EXPORT compute_depth
  : public vital::algo::compute_depth
{
public:
  /// Constructor
  compute_depth();

  /// Destructor
  virtual ~compute_depth();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Compute a depth map from an image sequence and return uncertainty by ref
  /**
  * Implementations of this function should not modify the underlying objects
  * contained in the input structures. Output references should either be new
  * instances or the same as input.
  *
  * \param [in] frames image sequence to compute depth with
  * \param [in] cameras corresponding to the image sequence
  * \param [in] depth_min minimum depth expected
  * \param [in] depth_max maximum depth expected
  * \param [in] reference_frame index into image sequence denoting the frame that depth is computed on
  * \param [in] roi region of interest within reference image (can be entire image)
  * \param [out] depth_uncertainty reference which will contain depth uncertainty
  * \param [in] masks optional masks corresponding to the image sequence
  */
  virtual kwiver::vital::image_container_sptr
  compute(std::vector<kwiver::vital::image_container_sptr> const& frames,
          std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
          double depth_min, double depth_max,
          unsigned int reference_frame,
          vital::bounding_box<int> const& roi,
          kwiver::vital::image_container_sptr& depth_uncertainty,
          std::vector<kwiver::vital::image_container_sptr> const& masks =
          std::vector<kwiver::vital::image_container_sptr>()) const;

  /// Set callback for receiving incremental updates
  virtual void set_callback(callback_t cb);

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

}  // end namespace super3d
}  // end namespace arrows
}  // end namespace kwiver

#endif
