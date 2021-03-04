// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Instantiation of \link kwiver::vital::algo::algorithm_def algorithm_def<T>
 *        \endlink for \link kwiver::vital::algo::bundle_adjust bundle_adjust \endlink
 */

#include <vital/algo/bundle_adjust.h>
#include <vital/algo/algorithm.txx>
#include <vital/logger/logger.h>
#include <vital/vital_config.h>

namespace kwiver {
namespace vital {
namespace algo {

bundle_adjust
::bundle_adjust()
{
  attach_logger( "algo.bundle_adjust" );
}

/// Set a callback function to report intermediate progress
void
bundle_adjust
::set_callback(callback_t cb)
{
  this->m_callback = cb;
}

void
bundle_adjust
::optimize(
  kwiver::vital::simple_camera_perspective_map &cameras,
  kwiver::vital::landmark_map::map_landmark_t &landmarks,
  vital::feature_track_set_sptr tracks,
  VITAL_UNUSED const std::set<vital::frame_id_t>& fixed_cameras,
  VITAL_UNUSED const std::set<vital::landmark_id_t>& fixed_landmarks,
  kwiver::vital::sfm_constraints_sptr constraints) const
{
  auto cam_map = std::static_pointer_cast<vital::camera_map>(
                   std::make_shared<vital::simple_camera_map>(cameras.cameras()));

  auto lm_map = std::static_pointer_cast<vital::landmark_map>(
                  std::make_shared<vital::simple_landmark_map>(landmarks));
  this->optimize(cam_map, lm_map, tracks, constraints);

  cameras.set_from_base_camera_map(cam_map->cameras());
  landmarks = lm_map->landmarks();
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::bundle_adjust);
/// \endcond
