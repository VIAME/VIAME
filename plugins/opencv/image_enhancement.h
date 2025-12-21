/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#ifndef VIAME_OPENCV_IMAGE_ENHANCEMENT_H
#define VIAME_OPENCV_IMAGE_ENHANCEMENT_H

#include <plugins/opencv/viame_opencv_export.h>

#include <vital/algo/image_filter.h>

namespace viame {

class VIAME_OPENCV_EXPORT image_enhancement
  : public kwiver::vital::algo::image_filter
{
public:
  PLUGIN_INFO( "ocv_enhancer",
               "Simple illumination normalization using Lab space and CLAHE" )

  image_enhancement();
  virtual ~image_enhancement();

  // Get the current configuration (parameters) for this filter
  virtual kwiver::vital::config_block_sptr get_configuration() const;

  // Set configurations automatically parsed from input pipeline and config files
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main filtering method
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );

private:
  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace

#endif /* VIAME_OPENCV_IMAGE_ENHANCEMENT_H */
