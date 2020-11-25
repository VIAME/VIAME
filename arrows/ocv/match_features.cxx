// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV match_features algorithm implementation
 */

#include "match_features.h"

#include <vector>

#include <vital/vital_config.h>

#include <arrows/ocv/descriptor_set.h>
#include <arrows/ocv/match_set.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

/// Match one set of features and corresponding descriptors to another
vital::match_set_sptr
match_features
::match( VITAL_UNUSED vital::feature_set_sptr feat1,
         vital::descriptor_set_sptr desc1,
         VITAL_UNUSED vital::feature_set_sptr feat2,
         vital::descriptor_set_sptr desc2) const
{
  // Return empty match set pointer if either of the input sets were empty
  // pointers
  if( !desc1 || !desc2 )
  {
    return vital::match_set_sptr();
  }
  // Only perform matching if both pointers are valid and if both descriptor
  // sets contain non-zero elements
  if( !desc1->size() || !desc2->size() )
  {
    return vital::match_set_sptr();
  }

  cv::Mat d1 = descriptors_to_ocv_matrix(*desc1);
  cv::Mat d2 = descriptors_to_ocv_matrix(*desc2);
  if( d1.empty() || d2.empty())
  {
    LOG_DEBUG( logger(), "Unable to convert descriptors to OpenCV format");
    return vital::match_set_sptr();
  }

  std::vector<cv::DMatch> matches;
  ocv_match(d1, d2, matches);
  return vital::match_set_sptr(new ocv::match_set(matches));
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
