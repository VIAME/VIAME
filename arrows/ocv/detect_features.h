// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV detect_features algorithm impl interface
 */

#ifndef KWIVER_ARROWS_OCV_DETECT_FEATURES_H_
#define KWIVER_ARROWS_OCV_DETECT_FEATURES_H_

#include <vital/algo/detect_features.h>

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <opencv2/features2d/features2d.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

/// OCV Specific base definition for algorithms that detect feature points
/**
 * This extended algorithm_def provides a common implementation for the detect
 * method.
 */
class KWIVER_ALGO_OCV_EXPORT detect_features
  : public kwiver::vital::algo::detect_features
{
public:
  /// Extract a set of image features from the provided image
  /**
   * A given mask image should be one-channel (mask->depth() == 1). If the
   * given mask image has more than one channel, only the first will be
   * considered.
   *
   * \param image_data contains the image data to process
   * \param mask Mask image where regions of positive values (boolean true)
   *             indicate regions to consider. Only the first channel will be
   *             considered.
   * \returns a set of image features
   */
  virtual vital::feature_set_sptr
  detect(vital::image_container_sptr image_data,
         vital::image_container_sptr mask = vital::image_container_sptr()) const;

protected:
  /// the feature detector algorithm
  cv::Ptr<cv::FeatureDetector> detector;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
