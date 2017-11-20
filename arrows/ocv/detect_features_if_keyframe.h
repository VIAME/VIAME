/*ckwg +29
* Copyright 2013-2016 by Kitware, Inc.
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
* \brief OCV detect_features algorithm impl interface
*/

#ifndef KWIVER_ARROWS_OCV_DETECT_FEATURES_IF_KEYFRAME_H_
#define KWIVER_ARROWS_OCV_DETECT_FEATURES_IF_KEYFRAME_H_


#include <vital/vital_config.h>
#include <vital/algo/detect_features_if_keyframe.h>

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/types/feature_track_set.h>

namespace kwiver {
namespace arrows {
namespace ocv {

  /// OCV Specific base definition for algorithms that detect feature points
  /**
  * This extended algorithm_def provides a common implementation for the detect
  * method.
  */
  class KWIVER_ALGO_OCV_EXPORT detect_features_if_keyframe
    : public vital::algo::detect_features_if_keyframe
  {
  public:

    /// Default constructor
    detect_features_if_keyframe();

    /// Destructor
    virtual ~detect_features_if_keyframe() noexcept;

    /// Get this algorithm's \link vital::config_block configuration block \endlink
    /**
    * This base virtual function implementation returns an empty configuration
    * block whose name is set to \c this->type_name.
    *
    * \returns \c config_block containing the configuration for this algorithm
    *          and any nested components.
    */
    virtual vital::config_block_sptr get_configuration() const;

    /// Set this algorithm's properties via a config block
    /**
    * \throws no_such_configuration_value_exception
    *    Thrown if an expected configuration value is not present.
    * \throws algorithm_configuration_exception
    *    Thrown when the algorithm is given an invalid \c config_block or is'
    *    otherwise unable to configure itself.
    *
    * \param config  The \c config_block instance containing the configuration
    *                parameters for this algorithm
    */
    virtual void set_configuration(vital::config_block_sptr config);

    /// Check that the algorithm's currently configuration is valid
    /**
    * This checks solely within the provided \c config_block and not against
    * the current state of the instance. This isn't static for inheritence
    * reasons.
    *
    * \param config  The config block to check configuration of.
    *
    * \returns true if the configuration check passed and false if it didn't.
    */
    virtual bool check_configuration(vital::config_block_sptr config) const;


    /// Extract a set of image features from the provided image.  If a track_set
    /// has been provided it will only extract features if the image is a keyframe.
    /**
    * A given mask image should be one-channel (mask->depth() == 1). If the
    * given mask image has more than one channel, only the first will be
    * considered.
    *
    * \param image_data contains the image data to process
    * \param mask Mask image where regions of positive values (boolean true)
    *             indicate regions to consider. Only the first channel will be
    *             considered.
    * \returns a set of image features
    */

    virtual kwiver::vital::feature_track_set_sptr
    detect(kwiver::vital::image_container_sptr image_data,
           unsigned int frame_number,
           kwiver::vital::feature_track_set_sptr feature_tracks,
           kwiver::vital::image_container_sptr mask = kwiver::vital::image_container_sptr()) const;

  protected:

    /// the feature detector algorithm
    class priv;
    std::shared_ptr<priv> d_;
  };

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif // KWIVER_ARROWS_OCV_DETECT_FEATURES_IF_KEYFRAME_H_
