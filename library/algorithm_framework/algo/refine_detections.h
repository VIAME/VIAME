// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining abstract image object detector

#ifndef VITAL_ALGO_REFINE_DETECTIONS_H_
#define VITAL_ALGO_REFINE_DETECTIONS_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/image_container.h>

#include <vector>

namespace kwiver {

namespace vital {

namespace algo {

// ----------------------------------------------------------------------------

/// @brief Case class for refining detected object sets.
///
class VITAL_ALGO_EXPORT refine_detections
  : public kwiver::vital::algorithm
{
public:
  refine_detections();
  PLUGGABLE_INTERFACE( refine_detections );
  /// Refine all object detections on the provided image
  ///
  /// This method analyzes the supplied image and and detections on it,
  /// returning a refined set of detections.
  ///
  /// \param image_data the image pixels
  /// \param detections detected objects
  /// \returns vector of image objects refined
  virtual detected_object_set_sptr
  refine(
    image_container_sptr image_data,
    detected_object_set_sptr detections ) const = 0;
};

/// Shared pointer for generic refine_detections definition type.
typedef std::shared_ptr< refine_detections > refine_detections_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_REFINE_DETECTIONS_H_
