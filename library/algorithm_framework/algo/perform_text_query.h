// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining abstract perform_text_query algorithm

#ifndef VITAL_ALGO_PERFORM_TEXT_QUERY_H_
#define VITAL_ALGO_PERFORM_TEXT_QUERY_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/object_track_set.h>
#include <viame/core_types/timestamp.h>

#include <string>
#include <vector>

namespace kwiver {

namespace vital {

namespace algo {

// ----------------------------------------------------------------------------

/// @brief Abstract base class for text-based object detection and track
/// refinement.
///
/// This algorithm uses natural language text queries to detect and segment
/// objects in one or more images. It can operate in several modes:
///
/// 1. Single image detection: One image, no tracks
///    - Returns detections matching the text query
///
/// 2. Multi-image detection: Multiple images (different cameras or frames)
///    - Returns detections for each image
///
/// 3. Track refinement: Images + existing tracks
///    - Refines existing tracks using text-guided segmentation
///    - Associates new detections with existing tracks
///
/// 4. Video processing: Images + timestamps
///    - Processes frames with temporal context
///    - Can maintain state across frames
///
class VITAL_ALGO_EXPORT perform_text_query
  : public kwiver::vital::algorithm
{
public:
  perform_text_query();
  PLUGGABLE_INTERFACE( perform_text_query );

  /// Perform text-based detection/segmentation on images.
  ///
  /// \param text_query Natural language description of objects to detect.
  ///        Examples: "fish", "red car", "person wearing hat"
  ///        Can be comma-separated for multiple classes: "fish, crab, starfish"
  ///
  /// \param images Vector of images to process. Interpretation depends on
  /// context:
  ///        - Multiple cameras at same time
  ///        - Multiple frames from video
  ///        - Mixed (e.g., stereo pairs over time)
  ///
  /// \param timestamps Optional timestamps corresponding to each image.
  ///        If provided, must match length of images.
  ///        Enables temporal reasoning and proper track state assignment.
  ///        If empty, images are treated as independent.
  ///
  /// \param input_tracks Optional existing tracks to refine, one per image.
  ///        If provided, must match length of images.
  ///        Detections are associated with existing tracks via IoU matching.
  ///        If empty, new track sets are created from detections.
  ///
  /// \returns Vector of track sets, one per input image.
  ///          Each contains object_track_state entries with bounding box,
  ///          confidence score, classification, and optional polygon mask.
  ///
  virtual std::vector< object_track_set_sptr >
  perform_query(
    std::string const& text_query,
    std::vector< image_container_sptr > const& images,
    std::vector< timestamp > const& timestamps = {},
    std::vector< object_track_set_sptr > const& input_tracks = {} ) const = 0;
};

/// Shared pointer for perform_text_query algorithm definition type.
typedef std::shared_ptr< perform_text_query > perform_text_query_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_PERFORM_TEXT_QUERY_H_
