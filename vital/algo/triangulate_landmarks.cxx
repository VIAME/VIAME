// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Instantiation of \link kwiver::vital::algo::algorithm_def algorithm_def<T>
 *        \endlink for \link kwiver::vital::algo::triangulate_landmarks
 *        triangulate_landmarks \endlink
 */

#include <vital/algo/triangulate_landmarks.h>
#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

triangulate_landmarks
::triangulate_landmarks()
{
  attach_logger( "algo.triangulate_landmarks" );
}

void
triangulate_landmarks
::triangulate(vital::camera_map_sptr cameras,
              vital::track_map_t tracks,
              vital::landmark_map_sptr& landmarks) const
{
  std::vector<track_sptr> track_vec(tracks.size());
  size_t i = 0;
  for (auto const& t : tracks)
  {
    track_vec[i++] = t.second;
  }
  auto track_ptr = std::make_shared<vital::feature_track_set>(track_vec);
  triangulate(cameras, track_ptr, landmarks);
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::triangulate_landmarks);
/// \endcond
