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

#include <vital/algo/compute_depth.h>
#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

compute_depth
::compute_depth()
{
  attach_logger("algo.compute_depth");
}

/// Set a callback function to report intermediate progress
void
compute_depth
::set_callback(callback_t cb)
{
  this->m_callback = cb;
}

/// Helper for computing without depth uncertainty pointer
kwiver::vital::image_container_sptr
compute_depth
::compute(std::vector<kwiver::vital::image_container_sptr> const& frames,
          std::vector<kwiver::vital::camera_perspective_sptr> const& cameras,
          double depth_min, double depth_max,
          unsigned int reference_frame,
          vital::bounding_box<int> const& roi,
          std::vector<kwiver::vital::image_container_sptr> const& masks) const
{
        kwiver::vital::image_container_sptr depth_uncertainty = nullptr;
        return compute(frames, cameras, depth_min, depth_max,
                       reference_frame, roi, depth_uncertainty, masks);
}

}  // end namespace algo
}  // end namespace vital
}  // end namespace kwiver

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::compute_depth);
/// \endcond
