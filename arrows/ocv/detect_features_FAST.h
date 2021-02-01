// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV FAST feature detector wrapper
 */

#ifndef KWIVER_ARROWS_DETECT_FEATURES_FAST_H_
#define KWIVER_ARROWS_DETECT_FEATURES_FAST_H_

#include <arrows/ocv/detect_features.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace ocv{

class KWIVER_ALGO_OCV_EXPORT detect_features_FAST
  : public ocv::detect_features
{
public:
  PLUGIN_INFO("ocv_FAST",
              "OpenCV feature detection via the FAST algorithm" )

  /// Constructor
  detect_features_FAST();

  /// Destructor
  virtual ~detect_features_FAST();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration vital::config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Extract a set of image features from the provided image
  /**
  * A given mask image should be one-channel (mask->depth() == 1). If the
  * given mask image has more than one channel, only the first will be
  * considered.
  * This method overrides the base detect method and adds dynamic threshold
  * adaptation.  It adjusts the detector's feature strength threshold to try
  * and extract a target number of features in each frame. Because scene
  * content varies between images, different feature strength thresholds may
  * be necessary to get the same number of feautres in different images.
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

private:
  class priv;
  std::unique_ptr<priv> const p_;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //KWIVER_ARROWS_DETECT_FEATURES_FAST_H_
