// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV feature_set interface
 */

#ifndef KWIVER_ARROWS_OCV_FEATURE_SET_H_
#define KWIVER_ARROWS_OCV_FEATURE_SET_H_

#include <vital/vital_config.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <opencv2/features2d/features2d.hpp>

#include <vital/types/feature_set.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// A concrete feature set that wraps OpenCV KeyPoints
class KWIVER_ALGO_OCV_EXPORT feature_set
  : public vital::feature_set
{
public:
  /// Default Constructor
  feature_set() {}

  /// Constructor from a vector of cv::KeyPoints
  explicit feature_set(const std::vector<cv::KeyPoint>& features)
  : data_(features) {}

  /// Return the number of feature in the set
  virtual size_t size() const { return data_.size(); }

  /// Return a vector of feature shared pointers
  virtual std::vector<vital::feature_sptr> features() const;

  /// Return the underlying OpenCV vector of cv::KeyPoints
  const std::vector<cv::KeyPoint>& ocv_keypoints() const { return data_; }

protected:

  /// The vector of KeyPoints
  std::vector<cv::KeyPoint> data_;
};

/// Convert any feature set to a vector of OpenCV cv::KeyPoints
KWIVER_ALGO_OCV_EXPORT std::vector<cv::KeyPoint>
features_to_ocv_keypoints(const vital::feature_set& features);

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
