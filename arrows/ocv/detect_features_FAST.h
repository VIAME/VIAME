/*ckwg +29
 * Copyright 2016, 2020 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
