// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief initialize_object_tracks algorithm definition

#ifndef VITAL_ALGO_INITIALIZE_OBJECT_TRACKS_MATRIX_H_
#define VITAL_ALGO_INITIALIZE_OBJECT_TRACKS_MATRIX_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/viame_compiler_config.h>

#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/object_track_set.h>
#include <viame/core_types/timestamp.h>

namespace viame {

namespace algo {

/// An abstract base class for computing association cost matrices for tracking
class VIAME_ALGO_EXPORT initialize_object_tracks
  : public viame::algorithm
{
public:
  initialize_object_tracks();
  PLUGGABLE_INTERFACE( initialize_object_tracks );
  /// Initialize new object tracks given detections.
  ///
  /// \param ts frame ID
  /// \param image contains the input image for the current frame
  /// \param detections detected object sets from the current frame
  /// \returns newly initialized tracks
  virtual viame::object_track_set_sptr
  initialize(
    viame::timestamp ts,
    viame::image_container_sptr image,
    viame::detected_object_set_sptr detections ) const = 0;
};

/// Shared pointer for initialize_object_tracks algorithm definition class
typedef std::shared_ptr< initialize_object_tracks >
  initialize_object_tracks_sptr;

} // namespace algo

} // namespace viame

#endif // VITAL_ALGO_INITIALIZE_OBJECT_TRACKS_H_
