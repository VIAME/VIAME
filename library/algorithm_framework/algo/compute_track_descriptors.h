// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief compute_track_descriptors algorithm definition

#ifndef VITAL_ALGO_COMPUTE_TRACK_DESCRIPTORS_H_
#define VITAL_ALGO_COMPUTE_TRACK_DESCRIPTORS_H_

#include <viame/algorithm_framework/vital_config.h>

#include <viame/algorithm_framework/algo/algorithm.h>

#include <viame/core_types/image_container.h>
#include <viame/core_types/object_track_set.h>
#include <viame/core_types/timestamp.h>
#include <viame/core_types/track_descriptor_set.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for computing track descriptors
class VITAL_ALGO_EXPORT compute_track_descriptors
  : public kwiver::vital::algorithm
{
public:
  compute_track_descriptors();
  PLUGGABLE_INTERFACE( compute_track_descriptors );
  /// Compute track descriptors given an image and tracks
  ///
  /// \param ts timestamp for the current frame
  /// \param image_data contains the image data to process
  /// \param tracks the tracks to extract descriptors around
  ///
  /// \returns a set of track descriptors
  virtual kwiver::vital::track_descriptor_set_sptr
  compute(
    kwiver::vital::timestamp ts,
    kwiver::vital::image_container_sptr image_data,
    kwiver::vital::object_track_set_sptr tracks ) = 0;

  /// Flush any remaining in-progress descriptors
  ///
  /// This is typically called at the end of a video, in case
  /// any temporal descriptors and currently in progress and
  /// still need to be output.
  ///
  /// \returns a set of track descriptors
  virtual kwiver::vital::track_descriptor_set_sptr flush() = 0;
};

/// Shared pointer for base compute_track_descriptors algorithm definition
/// class
typedef std::shared_ptr< compute_track_descriptors >
  compute_track_descriptors_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_COMPUTE_TRACK_DESCRIPTORS_H_
