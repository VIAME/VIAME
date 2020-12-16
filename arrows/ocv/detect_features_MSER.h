// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV MSER feature detector wrapper
 */

#ifndef KWIVER_ARROWS_DETECT_FEATURES_MSER_H_
#define KWIVER_ARROWS_DETECT_FEATURES_MSER_H_

#include <arrows/ocv/detect_features.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace ocv{

class KWIVER_ALGO_OCV_EXPORT detect_features_MSER
  : public detect_features
{
public:
  PLUGIN_INFO("ocv_MSER",
              "OpenCV feature detection via the MSER algorithm" )

  /// Constructor
  detect_features_MSER();

  /// Destructor
  virtual ~detect_features_MSER();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration vital::config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

private:
  class priv;
  std::unique_ptr<priv> p_;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //KWIVER_ARROWS_DETECT_FEATURES_MSER_H_
