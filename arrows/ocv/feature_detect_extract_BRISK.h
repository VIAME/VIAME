// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV BRISK feature detector and extractor wrapper
 */

#ifndef KWIVER_ARROWS_FEATURE_DETECT_EXTRACT_BRISK_H_
#define KWIVER_ARROWS_FEATURE_DETECT_EXTRACT_BRISK_H_

#include <arrows/ocv/detect_features.h>
#include <arrows/ocv/extract_descriptors.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace ocv {

class KWIVER_ALGO_OCV_EXPORT detect_features_BRISK
  : public ocv::detect_features
{
public:
  PLUGIN_INFO( "ocv_BRISK",
               "OpenCV feature detection via the BRISK algorithm" )

  /// Constructor
  detect_features_BRISK();

  /// Destructor
  virtual ~detect_features_BRISK();

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

class KWIVER_ALGO_OCV_EXPORT extract_descriptors_BRISK
  : public ocv::extract_descriptors
{
public:
  PLUGIN_INFO( "ocv_BRISK",
               "OpenCV feature-point descriptor extraction via the BRISK algorithm" )

  /// Constructor
  extract_descriptors_BRISK();

  /// Destructor
  virtual ~extract_descriptors_BRISK();

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

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
