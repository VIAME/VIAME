// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file

#ifndef VITAL_ALGO_DETECTED_OBJECT_FILTER_H_
#define VITAL_ALGO_DETECTED_OBJECT_FILTER_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/image_container.h>
#include <viame/algorithm_framework/vital_config.h>

#include <vector>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for filtering sets of detected objects
// ----------------------------------------------------------------------------

/// A detected object filter accepts a set of detections and produces
/// another set of detections. The output set may be different from the
/// input set. It all depends on the actual implementation. In any
/// case, the input detection set shall be unmodified.
class VITAL_ALGO_EXPORT detected_object_filter
  : public kwiver::vital::algorithm
{
public:
  detected_object_filter();
  PLUGGABLE_INTERFACE( detected_object_filter );
  /// Filter set of detected objects.
  ///
  /// This method applies a filter to the input set to create an output
  /// set. The input set of detections is unmodified.
  ///
  /// \param input_set Set of detections to be filtered.
  /// \returns Filtered set of detections.
  virtual detected_object_set_sptr
  filter( const detected_object_set_sptr input_set ) const = 0;
};

/// Shared pointer for generic detected_object_filter definition type.
typedef std::shared_ptr< detected_object_filter > detected_object_filter_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif
