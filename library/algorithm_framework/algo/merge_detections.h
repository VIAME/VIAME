// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#pragma once

#include <viame/algorithm_framework/vital_config.h>

#include <string>
#include <vector>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/detected_object_set.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for merging detection sets
class VITAL_ALGO_EXPORT merge_detections
  : public kwiver::vital::algorithm
{
public:
  merge_detections();
  PLUGGABLE_INTERFACE( merge_detections );

  /// Merge several detection sets into one
  ///
  /// Combines the supplied detection sets into a single set. Implementations
  /// decide how overlapping detections are resolved, for example by
  /// non-maximum suppression or by fusing their type scores.
  ///
  /// \param sets Detection sets to merge. All sets are expected to describe
  ///             the same frame.
  ///
  /// \returns The merged detection set. Empty if \p sets is empty or contains
  ///          no detections.
  virtual kwiver::vital::detected_object_set_sptr
  merge(
    std::vector< kwiver::vital::detected_object_set_sptr > const& sets ) const =
  0;
};

typedef std::shared_ptr< merge_detections > merge_detections_sptr;

} // namespace algo

} // namespace vital

}     // end namespace
