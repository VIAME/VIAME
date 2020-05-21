/*ckwg +29
 * Copyright 2016, 2019-2020 by Kitware, Inc.
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
 * \brief OCV ORB feature detector and extractor wrapper
 */

#ifndef KWIVER_ARROWS_FEATURE_DETECT_EXTRACT_ORB_H_
#define KWIVER_ARROWS_FEATURE_DETECT_EXTRACT_ORB_H_

#include <arrows/ocv/detect_features.h>
#include <arrows/ocv/extract_descriptors.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace ocv {


class KWIVER_ALGO_OCV_EXPORT detect_features_ORB
  : public ocv::detect_features
{
public:
  PLUGIN_INFO( "ocv_ORB",
               "OpenCV feature detection via the ORB algorithm" )

  /// Constructor
  detect_features_ORB();

  /// Destructor
  virtual ~detect_features_ORB();

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


class KWIVER_ALGO_OCV_EXPORT extract_descriptors_ORB
  : public ocv::extract_descriptors
{
public:
  PLUGIN_INFO( "ocv_ORB",
               "OpenCV feature-point descriptor extraction via the ORB algorithm" )

  /// Constructor
  extract_descriptors_ORB();

  /// Destructor
  virtual ~extract_descriptors_ORB();

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
