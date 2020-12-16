// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VISCL_DETECT_FEATURES_H_
#define KWIVER_ARROWS_VISCL_DETECT_FEATURES_H_

#include <arrows/viscl/kwiver_algo_viscl_export.h>

#include <vital/algo/detect_features.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// An algorithm class for detecting feature points using VisCL
class KWIVER_ALGO_VISCL_EXPORT detect_features
: public vital::algo::detect_features
{
public:
  /// Constructor
  detect_features();

  /// Destructor
  virtual ~detect_features();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);

  /// Check that the algorithm's configuration vital::config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Extract a set of image features from the provided image
  /**
    * \param image_data contains the image data to process
    * \returns a set of image features
    */
  virtual vital::feature_set_sptr
  detect(vital::image_container_sptr image_data,
         vital::image_container_sptr mask = vital::image_container_sptr()) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver

#endif
