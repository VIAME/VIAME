// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining abstract activity detector
 */

#ifndef VITAL_ALGO_ACTIVITY_DETECTOR_H_
#define VITAL_ALGO_ACTIVITY_DETECTOR_H_

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/activity.h>

#include <vector>

namespace kwiver {
namespace vital {
namespace algo {

// ----------------------------------------------------------------
/**
 * @brief activity detector base class/
 *
 */
class VITAL_ALGO_EXPORT activity_detector
: public algorithm_def<activity_detector>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "activity_detector"; }

  virtual std::vector<kwiver::vital::activity>
      detect( image_container_sptr image) const = 0;

protected:
  activity_detector();
};

/// Shared pointer for generic activity_detector definition type.
typedef std::shared_ptr<activity_detector> activity_detector_sptr;

} } } // end namespace

#endif
