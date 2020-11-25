// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV Star feature detector wrapper
 */

#ifndef KWIVER_ARROWS_DETECT_FEATURES_STAR_H_
#define KWIVER_ARROWS_DETECT_FEATURES_STAR_H_

#include <opencv2/opencv_modules.hpp>
#if KWIVER_OPENCV_VERSION_MAJOR < 3 || defined(HAVE_OPENCV_XFEATURES2D)

#include <memory>
#include <string>

#include <arrows/ocv/detect_features.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

namespace kwiver {
namespace arrows {
namespace ocv {

class KWIVER_ALGO_OCV_EXPORT detect_features_STAR
  : public ocv::detect_features
{
public:
  PLUGIN_INFO( "ocv_STAR",
               "OpenCV feature detection via the STAR algorithm" )

  /// Constructor
  detect_features_STAR();

  /// Destructor
  virtual ~detect_features_STAR();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

private:
  class priv;
  std::unique_ptr<priv> p_;
};

#define KWIVER_OCV_HAS_STAR

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //has OCV support

#endif //KWIVER_ARROWS_DETECT_FEATURES_STAR_H_
