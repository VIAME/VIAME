// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 */

#ifndef VITAL_ALGO_DETECTED_OBJECT_FILTER_H_
#define VITAL_ALGO_DETECTED_OBJECT_FILTER_H_

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>

#include <vector>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for filtering sets of detected objects
// ----------------------------------------------------------------
/**
 * A detected object filter accepts a set of detections and produces
 * another set of detections. The output set may be different from the
 * input set. It all depends on the actual implementation. In any
 * case, the input detection set shall be unmodified.
 */
class VITAL_ALGO_EXPORT detected_object_filter
  : public algorithm_def<detected_object_filter>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "detected_object_filter"; }

  /// Filter set of detected objects.
  /**
   * This method applies a filter to the input set to create an output
   * set. The input set of detections is unmodified.
   *
   * \param input_set Set of detections to be filtered.
   * \returns Filtered set of detections.
   */
  virtual detected_object_set_sptr
      filter( const detected_object_set_sptr input_set) const = 0;

protected:
  detected_object_filter();
};

/// Shared pointer for generic detected_object_filter definition type.
typedef std::shared_ptr<detected_object_filter> detected_object_filter_sptr;

} } } // end namespace

#endif
