// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV flann-based feature matcher wrapper
 */

#ifndef KWIVER_ARROWS_MATCH_FEATURES_FLANNBASED_H
#define KWIVER_ARROWS_MATCH_FEATURES_FLANNBASED_H

#include <memory>
#include <vector>

#include <arrows/ocv/kwiver_algo_ocv_export.h>
#include <arrows/ocv/match_features.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// Feature matcher implementation using OpenCV's FLANN-based feature matcher
class KWIVER_ALGO_OCV_EXPORT match_features_flannbased
  : public ocv::match_features
{
public:
  PLUGIN_INFO( "ocv_flann_based",
               "OpenCV feature matcher using FLANN (Approximate Nearest Neighbors).")

  /// Constructor
  match_features_flannbased();

  /// Destructor
  virtual ~match_features_flannbased();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration vital::config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

protected:
  /// Perform matching based on the underlying OpenCV implementation
  virtual void ocv_match(const cv::Mat& descriptors1,
                         const cv::Mat& descriptors2,
                         std::vector<cv::DMatch>& matches) const;

private:
  class priv;
  std::unique_ptr<priv> const p_;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //KWIVER_ARROWS_MATCH_FEATURES_FLANNBASED_H
