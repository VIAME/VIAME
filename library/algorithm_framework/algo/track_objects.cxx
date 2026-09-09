// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief track_objects algorithm implementation

#include <viame/algorithm_framework/algo/track_objects.h>

#include <viame/algorithm_framework/vital_config.h>

namespace kwiver {

namespace vital {

namespace algo {

track_objects
::track_objects()
{
  attach_logger( "algo.track_objects" );
}

// ----------------------------------------------------------------------------
kwiver::vital::object_track_set_sptr
track_objects
::track(
  kwiver::vital::timestamp ts,
  kwiver::vital::image_container_sptr image,
  kwiver::vital::detected_object_set_sptr detections,
  [[maybe_unused]] kwiver::vital::f2f_homography_sptr src_to_ref ) const
{
  // Default implementation ignores homography and calls base track method.
  // Implementations that support homography should override this.
  return this->track( ts, image, detections );
}

// ----------------------------------------------------------------------------
kwiver::vital::object_track_set_sptr
track_objects
::track(
  kwiver::vital::timestamp ts,
  kwiver::vital::image_container_sptr image,
  kwiver::vital::detected_object_set_sptr detections,
  [[maybe_unused]] kwiver::vital::object_track_set_sptr existing_tracks ) const
{
  // Default implementation ignores existing tracks and calls base track method.
  // Implementations that support track continuation should override this.
  return this->track( ts, image, detections );
}

// ----------------------------------------------------------------------------
kwiver::vital::object_track_set_sptr
track_objects
::initialize(
  [[maybe_unused]] kwiver::vital::timestamp ts,
  [[maybe_unused]] kwiver::vital::image_container_sptr image,
  [[maybe_unused]] kwiver::vital::detected_object_set_sptr seed_detections )
const
{
  // Default implementation does nothing.
  // Not all trackers initialize from seeds, most do internal initialization.
  return std::make_shared< kwiver::vital::object_track_set >();
}

// ----------------------------------------------------------------------------
kwiver::vital::object_track_set_sptr
track_objects
::finalize() const
{
  // Default implementation returns empty track set.
  // Implementations should override to return accumulated tracks.
  return std::make_shared< kwiver::vital::object_track_set >();
}

// ----------------------------------------------------------------------------
void
track_objects
::reset() const
{
  // Default implementation does nothing.
  // Implementations should override to clear internal state.
}

} // namespace algo

} // namespace vital

} // namespace kwiver
