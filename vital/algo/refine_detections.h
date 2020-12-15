// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining abstract image object detector
 */

#ifndef VITAL_ALGO_REFINE_DETECTIONS_H_
#define VITAL_ALGO_REFINE_DETECTIONS_H_

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>

#include <vector>

namespace kwiver {
namespace vital {
namespace algo {

// ----------------------------------------------------------------
/**
 * @brief Case class for refining detected object sets.
 *
 */
class VITAL_ALGO_EXPORT refine_detections
: public algorithm_def<refine_detections>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "refine_detections"; }

  /// Refine all object detections on the provided image
  /**
   * This method analyzes the supplied image and and detections on it,
   * returning a refined set of detections.
   *
   * \param image_data the image pixels
   * \param detections detected objects
   * \returns vector of image objects refined
   */
  virtual detected_object_set_sptr
  refine( image_container_sptr image_data,
          detected_object_set_sptr detections ) const = 0;

protected:
  refine_detections();
};

/// Shared pointer for generic refine_detections definition type.
typedef std::shared_ptr<refine_detections> refine_detections_sptr;

} } } // end namespace

#endif //VITAL_ALGO_REFINE_DETECTIONS_H_
