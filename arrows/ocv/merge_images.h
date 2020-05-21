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
 * \brief Header for OCV merge_images algorithm
 */

#pragma once

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/merge_images.h>

namespace kwiver {
namespace arrows {
namespace ocv {

// Implementation of merge image channels.
class KWIVER_ALGO_OCV_EXPORT merge_images
  : public vital::algo::merge_images
{
public:
  PLUGIN_INFO( "ocv",
               "Merge two images into one using opencv functions.\n\n"
               "The channels from the first image are added to the "
               "output image first, followed by the channels from the "
               "second image. This implementation takes no configuration "
               "parameters."
    )

  /// Constructor
  merge_images();

  /// Destructor
  virtual ~merge_images() = default;

  void set_configuration( kwiver::vital::config_block_sptr ) override { }
  bool check_configuration( kwiver::vital::config_block_sptr config ) const override
  { return true; }

  /// Merge images
  kwiver::vital::image_container_sptr
    merge(kwiver::vital::image_container_sptr image1,
          kwiver::vital::image_container_sptr image2) const override;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
