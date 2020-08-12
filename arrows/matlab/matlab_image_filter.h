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

/**
 * \file
 * \brief Header defining matlab image_filter
 */

#ifndef VITAL_BINDINGS_MATLAB_IMAGE_FILTER_H
#define VITAL_BINDINGS_MATLAB_IMAGE_FILTER_H

#include <vital/algo/image_filter.h>
#include <arrows/matlab/kwiver_algo_matlab_export.h>

namespace kwiver {
namespace arrows {
namespace matlab {

class KWIVER_ALGO_MATLAB_EXPORT matlab_image_filter
  : public vital::algo::image_filter
{
public:
  matlab_image_filter();
  virtual ~matlab_image_filter();

  PLUGIN_INFO( "matlab",
               "Bridge to matlab image filter implementation." );

  vital::config_block_sptr get_configuration() const override;
  void set_configuration(vital::config_block_sptr config) override;
  bool check_configuration(vital::config_block_sptr config) const override;

  // Main detection method
  vital::image_container_sptr filter( vital::image_container_sptr image_data) override;

private:
  class priv;
  const std::unique_ptr<priv> d;
};

} } } // end namespace

#endif // VITAL_BINDINGS_MATLAB_IMAGE_FILTER_H
