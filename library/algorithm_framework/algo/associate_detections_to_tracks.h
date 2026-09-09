// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief associate_detections_to_tracks algorithm definition

#ifndef VITAL_ALGO_ASSOCIATE_DETECTIONS_TO_TRACKS_H_
#define VITAL_ALGO_ASSOCIATE_DETECTIONS_TO_TRACKS_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/vital_config.h>

#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/matrix.h>
#include <viame/core_types/object_track_set.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for using cost matrices to assign detections to
/// tracks
class VITAL_ALGO_EXPORT associate_detections_to_tracks
  : public kwiver::vital::algorithm
{
public:
  associate_detections_to_tracks();
  PLUGGABLE_INTERFACE( associate_detections_to_tracks );

  /// Use cost matrices to assign detections to existing tracks
  ///
  /// \param ts frame ID
  /// \param image contains the input image for the current frame
  /// \param tracks active track set from the last frame
  /// \param detections detected object sets from the current frame
  /// \param matrix matrix containing detection to track association scores
  /// \param output the output updated detection set
  /// \param unused output detection set for any detections not associated
  /// \returns whether or not any tracks were updated
  virtual bool
  associate(
    kwiver::vital::timestamp ts,
    kwiver::vital::image_container_sptr image,
    kwiver::vital::object_track_set_sptr tracks,
    kwiver::vital::detected_object_set_sptr detections,
    kwiver::vital::matrix_d matrix,
    kwiver::vital::object_track_set_sptr& output,
    kwiver::vital::detected_object_set_sptr& unused ) const = 0;
};

/// Shared pointer for associate_detections_to_tracks algorithm definition
/// class
typedef std::shared_ptr< associate_detections_to_tracks >
  associate_detections_to_tracks_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_ASSOCIATE_DETECTIONS_TO_TRACKS_H_
