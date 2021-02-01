// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV simple blob feature detector wrapper
 */

#ifndef KWIVER_ARROWS_DETECT_FEATURES_SIMPLE_BLOB_H_
#define KWIVER_ARROWS_DETECT_FEATURES_SIMPLE_BLOB_H_

#include <arrows/ocv/detect_features.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace ocv {

class KWIVER_ALGO_OCV_EXPORT detect_features_simple_blob
  : public ocv::detect_features
{
public:
  PLUGIN_INFO( "ocv_simple_blob",
               "OpenCV feature detection via the simple_blob algorithm." )

  /// Constructor
  detect_features_simple_blob();

  /// Destructor
  virtual ~detect_features_simple_blob();

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

#endif //KWIVER_ARROWS_DETECT_FEATURES_SIMPLE_BLOB_H_
