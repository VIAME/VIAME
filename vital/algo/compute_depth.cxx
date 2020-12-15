// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
