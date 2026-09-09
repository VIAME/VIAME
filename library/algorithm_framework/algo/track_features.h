// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining abstract \link kwiver::vital::algo::track_features
/// feature
///        tracking \endlink algorithm

#ifndef VITAL_ALGO_TRACK_FEATURES_H_
#define VITAL_ALGO_TRACK_FEATURES_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/feature_track_set.h>
#include <viame/core_types/image_container.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for tracking feature points
class VITAL_ALGO_EXPORT track_features
  : public kwiver::vital::algorithm
{
public:
  track_features();
  PLUGGABLE_INTERFACE( track_features );
  /// Extend a previous set of feature tracks using the current frame
  ///
  /// \throws image_size_mismatch_exception
  ///    When the given non-zero mask image does not match the size of the
  ///    dimensions of the given image data.
  ///
  /// \param [in] prev_tracks the feature tracks from previous tracking steps
  /// \param [in] frame_number the frame number of the current frame
  /// \param [in] image_data the image pixels for the current frame
  /// \param [in] mask Optional mask image that uses positive values to denote
  ///                  regions of the input image to consider for feature
  ///                  tracking. An empty sptr indicates no mask (default
  ///                  value).
  /// \returns an updated set of feature tracks including the current frame
  virtual feature_track_set_sptr
  track(
    feature_track_set_sptr prev_tracks,
    frame_id_t frame_number,
    image_container_sptr image_data,
    image_container_sptr mask = {} ) const = 0;
};

/// Shared pointer for generic track_features definition type.
typedef std::shared_ptr< track_features > track_features_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_TRACK_FEATURES_H_
