// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV MSD feature detector wrapper
 */

#ifndef KWIVER_ARROWS_DETECT_FEATURES_MSD_H_
#define KWIVER_ARROWS_DETECT_FEATURES_MSD_H_

// Only available in OpenCV 3.x xfeatures2d
#include <opencv2/opencv_modules.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D

#include <arrows/ocv/detect_features.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace ocv {

class KWIVER_ALGO_OCV_EXPORT detect_features_MSD
  : public ocv::detect_features
{
public:
  PLUGIN_INFO( "ocv_MSD",
               "OpenCV feature detection via the MSD algorithm" )

  /// Constructor
  detect_features_MSD();

  /// Destructor
  virtual ~detect_features_MSD();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration vital::config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

private:
  class priv;
  std::unique_ptr<priv> const p_;
};

#define KWIVER_OCV_HAS_MSD

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //HAVE_OPENCV_XFEATURES2D

#endif //KWIVER_ARROWS_DETECT_FEATURES_MSD_H_
