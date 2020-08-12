/*ckwg +29
 * Copyright 2018, 2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#ifndef KWIVER_ARROWS_BURNOUT_IMAGE_ENHANCER
#define KWIVER_ARROWS_BURNOUT_IMAGE_ENHANCER

#include <arrows/burnout/kwiver_algo_burnout_export.h>

#include <vital/algo/image_filter.h>

namespace kwiver {
namespace arrows {
namespace burnout {

/**
 * @brief Burnout Image Filtering
 *
 * This method contains basic methods for image filtering on top of input
 * images via automatic white balancing and smoothing.
 */
class KWIVER_ALGO_BURNOUT_EXPORT burnout_image_enhancer
  : public vital::algo::image_filter
{
public:
  burnout_image_enhancer();
  virtual ~burnout_image_enhancer();

  PLUGIN_INFO( "burnout_enhancer",
               "Image filtering using burnout" )

  vital::config_block_sptr get_configuration() const override;

  void set_configuration( vital::config_block_sptr config ) override;
  bool check_configuration( vital::config_block_sptr config ) const override;

  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data ) override;

private:

  class priv;
  const std::unique_ptr<priv> d;
};

} } }

#endif /* KWIVER_ARROWS_BURNOUT_IMAGE_ENHANCER */
