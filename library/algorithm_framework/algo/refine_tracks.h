// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining abstract track refiner

#ifndef VITAL_ALGO_REFINE_TRACKS_H_
#define VITAL_ALGO_REFINE_TRACKS_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/object_track_set.h>
#include <viame/core_types/timestamp.h>

namespace kwiver {

namespace vital {

namespace algo {

// ----------------------------------------------------------------------------

/// @brief Base class for refining object track sets.
///
/// This algorithm refines tracks on a per-frame basis. Unlike track_objects
/// which maintains internal state across frames, refine_tracks operates
/// on the current frame's track state only.
///
/// Use cases include:
/// - Re-segmenting track masks using SAM for improved polygon quality
/// - Filtering low-quality tracks based on mask/detection confidence
/// - Adjusting bounding boxes based on refined masks
/// - Adding new objects that don't overlap with existing tracks
/// - Removing tracks that no longer match query criteria
///
class VITAL_ALGO_EXPORT refine_tracks
  : public kwiver::vital::algorithm
{
public:
  refine_tracks();
  PLUGGABLE_INTERFACE( refine_tracks );

  /// Refine all object tracks for the current frame.
  ///
  /// This method analyzes the supplied image and tracks, returning
  /// a refined set of tracks for the current frame.
  ///
  /// \param ts Timestamp for the current frame
  /// \param image_data The image pixels for the current frame
  /// \param tracks Object tracks to refine (containing states for current
  /// frame)
  /// \returns Refined object track set
  virtual object_track_set_sptr
  refine(
    timestamp ts,
    image_container_sptr image_data,
    object_track_set_sptr tracks ) const = 0;

  /// Finalize the refiner after all frames have been processed.
  ///
  /// Called when the pipeline signals completion.  Implementations may
  /// override this to run deferred processing (e.g. video propagation
  /// over the full accumulated buffer).
  ///
  /// \returns Final refined object track set, or nullptr if no final
  ///          output is needed.
  virtual object_track_set_sptr
  finalize() const { return nullptr; }
};

/// Shared pointer for generic refine_tracks definition type.
typedef std::shared_ptr< refine_tracks > refine_tracks_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_REFINE_TRACKS_H_
