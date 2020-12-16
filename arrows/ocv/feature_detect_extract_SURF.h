// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV SURF feature detector and extractor wrapper
 */

#ifndef KWIVER_ARROWS_FEATURE_DETECT_EXTRACT_SURF_H_
#define KWIVER_ARROWS_FEATURE_DETECT_EXTRACT_SURF_H_

#include <opencv2/opencv_modules.hpp>
#if defined(HAVE_OPENCV_NONFREE) || defined(HAVE_OPENCV_XFEATURES2D)

#include <arrows/ocv/detect_features.h>
#include <arrows/ocv/extract_descriptors.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace ocv {

class KWIVER_ALGO_OCV_EXPORT detect_features_SURF
  : public ocv::detect_features
{
public:
  PLUGIN_INFO( "ocv_SURF",
               "OpenCV feature detection via the SURF algorithm" )

  /// Constructor
  detect_features_SURF();

  /// Destructor
  virtual ~detect_features_SURF();

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

class KWIVER_ALGO_OCV_EXPORT extract_descriptors_SURF
  : public ocv::extract_descriptors
{
public:
  PLUGIN_INFO( "ocv_SURF",
               "OpenCV feature-point descriptor extraction via the SURF algorithm" )

  /// Constructor
  extract_descriptors_SURF();

  /// Destructor
  virtual ~extract_descriptors_SURF();

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

#define KWIVER_OCV_HAS_SURF

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //defined(HAVE_OPENCV_NONFREE) || defined(HAVE_OPENCV_XFEATURES2D)

#endif
