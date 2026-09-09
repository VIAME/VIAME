// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief track_objects algorithm definition

#ifndef VITAL_ALGO_TRACK_OBJECTS_H_
#define VITAL_ALGO_TRACK_OBJECTS_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/vital_config.h>

#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/homography_f2f.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/object_track_set.h>
#include <viame/core_types/timestamp.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for single camera object tracking algorithms.
///
/// This algorithm provides a unified interface for various object tracking
/// approaches including:
/// - Detection-based multi-object trackers (ByteTrack, OC-SORT, DeepSORT, etc.)
/// - Template-based single object trackers (SiamMask, MDNet, etc.)
/// - IOU-based trackers with optional homography support
///
/// The algorithm maintains internal state across frames to track objects
/// over time. Implementations may use different strategies such as:
/// - Kalman filtering for motion prediction
/// - Deep appearance features for re-identification
/// - Homography transforms for camera motion compensation
///
/// All tracker implementations should provide the track() method for
/// frame-by-frame processing. Optional methods support initialization,
/// finalization, and state management.
class VITAL_ALGO_EXPORT track_objects
  : public kwiver::vital::algorithm
{
public:
  track_objects();
  PLUGGABLE_INTERFACE( track_objects );

  /// Track objects in a new frame.
  ///
  /// This is the primary tracking method that processes a single frame.
  /// It takes the current frame's detections and optional additional
  /// inputs, updates internal track state, and returns the current
  /// set of active tracks.
  ///
  /// \param ts Timestamp for the current frame
  /// \param image The input image for the current frame (may be null
  ///              for trackers that don't require image data)
  /// \param detections Detected objects from the current frame
  /// \returns Updated object track set containing all active tracks
  virtual kwiver::vital::object_track_set_sptr
  track(
    kwiver::vital::timestamp ts,
    kwiver::vital::image_container_sptr image,
    kwiver::vital::detected_object_set_sptr detections ) const = 0;

  /// Track objects with homography support.
  ///
  /// This overload supports trackers that use frame-to-frame or
  /// frame-to-reference homographies for camera motion compensation.
  /// This is useful for aerial or moving camera scenarios.
  ///
  /// \param ts Timestamp for the current frame
  /// \param image The input image for the current frame
  /// \param detections Detected objects from the current frame
  /// \param src_to_ref Homography from source (current frame) to
  ///                   reference coordinates
  /// \returns Updated object track set containing all active tracks
  virtual kwiver::vital::object_track_set_sptr
  track(
    kwiver::vital::timestamp ts,
    kwiver::vital::image_container_sptr image,
    kwiver::vital::detected_object_set_sptr detections,
    kwiver::vital::f2f_homography_sptr src_to_ref ) const;

  /// Track objects with existing tracks provided.
  ///
  /// This overload allows passing in existing tracks for continuation
  /// or re-initialization scenarios. Useful for multi-stage tracking
  /// pipelines or when tracks need to be initialized from external sources.
  ///
  /// \param ts Timestamp for the current frame
  /// \param image The input image for the current frame
  /// \param detections Detected objects from the current frame
  /// \param existing_tracks Previously computed tracks to continue
  /// \returns Updated object track set with both existing and new tracks
  virtual kwiver::vital::object_track_set_sptr
  track(
    kwiver::vital::timestamp ts,
    kwiver::vital::image_container_sptr image,
    kwiver::vital::detected_object_set_sptr detections,
    kwiver::vital::object_track_set_sptr existing_tracks ) const;

  /// Initialize the tracker for a new sequence.
  ///
  /// Called at the start of a new video sequence to reset internal
  /// state and prepare for tracking. Some trackers may require
  /// initialization with seed detections or bounding boxes.
  ///
  /// \param ts Initial timestamp
  /// \param image Initial frame image
  /// \param seed_detections Optional initial detections to seed tracks
  /// \returns Initial track set (may be empty if no seeds provided)
  virtual kwiver::vital::object_track_set_sptr
  initialize(
    kwiver::vital::timestamp ts,
    kwiver::vital::image_container_sptr image,
    kwiver::vital::detected_object_set_sptr seed_detections ) const;

  /// Finalize tracking and return all tracks.
  ///
  /// Called at the end of a sequence to perform any final processing
  /// and return the complete set of tracks. This may include tracks
  /// that were previously lost but should still be returned.
  ///
  /// \returns Final object track set with all tracks from the sequence
  virtual kwiver::vital::object_track_set_sptr
  finalize() const;

  /// Reset the tracker state.
  ///
  /// Clears all internal state, active tracks, and cached data.
  /// After reset, the tracker is ready to begin a new sequence.
  virtual void reset() const;
};

/// Shared pointer for track_objects algorithm definition class
typedef std::shared_ptr< track_objects > track_objects_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_TRACK_OBJECTS_H_
