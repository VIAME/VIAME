// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining abstract \link kwiver::vital::algo::track_features feature
 *        tracking \endlink algorithm
 */

#ifndef VITAL_ALGO_TRACK_FEATURES_H_
#define VITAL_ALGO_TRACK_FEATURES_H_

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_track_set.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for tracking feature points
class VITAL_ALGO_EXPORT track_features
  : public algorithm_def<track_features>
{
public:

  /// Return the name of this algorithm
  static std::string static_type_name() { return "track_features"; }

  /// Extend a previous set of feature tracks using the current frame
  /**
   * \throws image_size_mismatch_exception
   *    When the given non-zero mask image does not match the size of the
   *    dimensions of the given image data.
   *
   * \param [in] prev_tracks the feature tracks from previous tracking steps
   * \param [in] frame_number the frame number of the current frame
   * \param [in] image_data the image pixels for the current frame
   * \param [in] mask Optional mask image that uses positive values to denote
   *                  regions of the input image to consider for feature
   *                  tracking. An empty sptr indicates no mask (default
   *                  value).
   * \returns an updated set of feature tracks including the current frame
   */
  virtual feature_track_set_sptr
  track(feature_track_set_sptr prev_tracks,
        frame_id_t frame_number,
        image_container_sptr image_data,
        image_container_sptr mask = {}) const = 0;

protected:
    track_features();

};

/// Shared pointer for generic track_features definition type.
typedef std::shared_ptr<track_features> track_features_sptr;

} } } // end namespace

#endif // VITAL_ALGO_TRACK_FEATURES_H_
