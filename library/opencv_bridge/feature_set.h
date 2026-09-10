// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV feature_set interface

#ifndef KWIVER_ARROWS_OCV_FEATURE_SET_H_
#define KWIVER_ARROWS_OCV_FEATURE_SET_H_

#include "viame_opencv_bridge_export.h"
#include <viame/algorithm_framework/vital_config.h>

#include <opencv2/features2d/features2d.hpp>

#include <viame/core_types/feature_set.h>

namespace kwiver {

namespace arrows {

namespace ocv {

/// A concrete feature set that wraps OpenCV KeyPoints
class VIAME_OPENCV_BRIDGE_EXPORT feature_set
  : public vital::feature_set
{
public:
  /// Default Constructor
  feature_set() {}

  /// Constructor from a vector of cv::KeyPoints
  explicit feature_set( const std::vector< cv::KeyPoint >& features )
    : data_( features ) {}

  /// Return the number of feature in the set
  virtual size_t
  size() const { return data_.size(); }

  /// Return a vector of feature shared pointers
  virtual std::vector< vital::feature_sptr > features() const;

  /// Return the underlying OpenCV vector of cv::KeyPoints
  const std::vector< cv::KeyPoint >&
  ocv_keypoints() const { return data_; }

protected:
  /// The vector of KeyPoints
  std::vector< cv::KeyPoint > data_;
};

/// Convert any feature set to a vector of OpenCV cv::KeyPoints
VIAME_OPENCV_BRIDGE_EXPORT std::vector< cv::KeyPoint >
features_to_ocv_keypoints( const vital::feature_set& features );

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
