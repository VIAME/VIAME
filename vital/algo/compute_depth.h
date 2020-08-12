/*ckwg +29
* Copyright 2017-2018 by Kitware, Inc.
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
* \brief Header defining abstract \link kwiver::vital::algo::compute_depth
*        compute depth \endlink algorithm
*/

#ifndef VITAL_ALGO_COMPUTE_DEPTH_H_
#define VITAL_ALGO_COMPUTE_DEPTH_H_

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/image_container.h>
#include <vital/types/landmark.h>
#include <vital/types/bounding_box.h>

#include <vector>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for depth map estimation
class VITAL_ALGO_EXPORT compute_depth
  : public kwiver::vital::algorithm_def<compute_depth>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "compute_depth"; }

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
            std::vector<kwiver::vital::image_container_sptr>()) const = 0;

  /// Typedef for the callback function signature
  typedef std::function<bool (kwiver::vital::image_container_sptr,
                              std::string const&,
                              unsigned int)> callback_t;

  /// Set a callback function to report intermediate progress
  virtual void set_callback(callback_t cb);

protected:
  compute_depth();

  /// The callback function
  callback_t m_callback;
};


/// type definition for shared pointer to a compute depth algorithm
typedef std::shared_ptr<compute_depth> compute_depth_sptr;

} // end namespace algo
} // end namespace vital
} // end namespace kwiver

#endif // VITAL_ALGO_COMPUTE_DEPTH_H_
