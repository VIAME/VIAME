// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining abstract \link kwiver::vital::algo::triangulate_landmarks
 *        triangulate landmarks \endlink algorithm
 */

#ifndef VITAL_ALGO_TRIANGULATE_LANDMARKS_H_
#define VITAL_ALGO_TRIANGULATE_LANDMARKS_H_

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/camera_map.h>
#include <vital/types/landmark_map.h>

namespace kwiver {
namespace vital {

typedef std::map<vital::track_id_t, vital::track_sptr> track_map_t;

namespace algo {

/// An abstract base class for triangulating landmarks
class VITAL_ALGO_EXPORT triangulate_landmarks
: public kwiver::vital::algorithm_def<triangulate_landmarks>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "triangulate_landmarks"; }

  /// Triangulate the landmark locations given sets of cameras and feature tracks
  /**
   * \param [in] cameras the cameras viewing the landmarks
   * \param [in] tracks the feature tracks to use as constraints
   * \param [in,out] landmarks the landmarks to triangulate
   *
   * This function only triangulates the landmarks with indices in the
   * landmark map and which have support in the feature tracks and cameras
   */
  virtual void
  triangulate(kwiver::vital::camera_map_sptr cameras,
              kwiver::vital::feature_track_set_sptr tracks,
              kwiver::vital::landmark_map_sptr& landmarks) const = 0;

  /// Triangulate the landmark locations given sets of cameras and feature tracks
  /**
  * \param [in] cameras the cameras viewing the landmarks
  * \param [in] tracks the feature tracks to use as constraints stored in a track map
  * \param [in,out] landmarks the landmarks to triangulate
  *
  * This function only triangulates the landmarks with indices in the
  * landmark map and which have support in the feature tracks and cameras.
  */
  virtual void
  triangulate(vital::camera_map_sptr cameras,
              vital::track_map_t tracks,
              vital::landmark_map_sptr& landmarks) const;

protected:
  triangulate_landmarks();

};

/// type definition for shared pointer to a triangulate landmarks algorithm
typedef std::shared_ptr<triangulate_landmarks> triangulate_landmarks_sptr;

} // end namespace algo
} // end namespace vital
} // end namespace kwiver

#endif // VITAL_ALGO_TRIANGULATE_LANDMARKS_H_
