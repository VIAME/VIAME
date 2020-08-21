/*ckwg +29
* Copyright 2017-2020 by Kitware, Inc.
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

  /// Compute a depth map from an image sequence
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
  * \param [in] masks optional masks corresponding to the image sequence
  */
  virtual kwiver::vital::image_container_sptr
  compute(std::vector<kwiver::vital::image_container_sptr> const& frames,
          std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
          double depth_min, double depth_max,
          unsigned int reference_frame,
          vital::bounding_box<int> const& roi,
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
