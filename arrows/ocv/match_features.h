// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV match_features algorithm impl interface
 */

#ifndef KWIVER_ARROWS_OCV_MATCH_FEATURES_H_
#define KWIVER_ARROWS_OCV_MATCH_FEATURES_H_

#include <vital/algo/match_features.h>

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <opencv2/features2d/features2d.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

/// OCV specific definition for algorithms that match feature point descriptors
/**
 * This extended algorithm_def provides a common implementation for the match
 * method.
 */
class KWIVER_ALGO_OCV_EXPORT match_features
  : public vital::algo::match_features
{
public:
  /// Match one set of features and corresponding descriptors to another
  /**
   * \param feat1 the first set of features to match
   * \param desc1 the descriptors corresponding to \a feat1
   * \param feat2 the second set fof features to match
   * \param desc2 the descriptors corresponding to \a feat2
   * \returns a set of matching indices from \a feat1 to \a feat2
   */
  virtual vital::match_set_sptr
  match(vital::feature_set_sptr feat1, vital::descriptor_set_sptr desc1,
        vital::feature_set_sptr feat2, vital::descriptor_set_sptr desc2) const;

protected:
  /// Perform matching based on the underlying OpenCV implementation
  /**
   * Implementations of this sub-definition must implement this method based on
   * the OpeCV implementation being wrapped.
   *
   * \param [in] descriptors1 First set of descriptors to match.
   * \param [in] descriptors2 Second set of descriptors to match.
   * \param [out] matches Vector of result matches.
   */
  virtual void ocv_match(const cv::Mat& descriptors1,
                         const cv::Mat& descriptors2,
                         std::vector<cv::DMatch>& matches) const = 0;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
