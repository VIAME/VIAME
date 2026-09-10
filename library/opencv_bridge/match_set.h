// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV match_set interface

#ifndef KWIVER_ARROWS_OCV_MATCH_SET_H_
#define KWIVER_ARROWS_OCV_MATCH_SET_H_

#include "viame_opencv_bridge_export.h"
#include <viame/algorithm_framework/vital_config.h>

#include <opencv2/features2d/features2d.hpp>

#include <viame/core_types/match_set.h>

namespace kwiver {

namespace arrows {

namespace ocv {

/// A concrete match set that wraps OpenCV cv::DMatch objects
class VIAME_OPENCV_BRIDGE_EXPORT match_set
  : public vital::match_set
{
public:
  /// Default constructor
  match_set() {}

  /// Constructor from a vector of cv::DMatch
  explicit match_set( const std::vector< cv::DMatch >& matches )
    : data_( matches ) {}

  /// Return the number of matches in the set
  virtual size_t
  size() const { return data_.size(); }

  /// Return a vector of matching indices
  virtual std::vector< vital::match > matches() const;

  /// Return the underlying OpenCV match data structures
  const std::vector< cv::DMatch >&
  ocv_matches() const { return data_; }

private:
  // The vector of OpenCV match structures
  std::vector< cv::DMatch > data_;
};

/// Convert any match set to a vector of OpenCV cv::DMatch
VIAME_OPENCV_BRIDGE_EXPORT std::vector< cv::DMatch >
matches_to_ocv_dmatch( const vital::match_set& match_set );

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
