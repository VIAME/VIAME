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

#pragma once

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class merge_images_process
 *
 * \brief Merges two images
 *
 * \iports
 * \iport{image}
 *
 * \oports
 * \oport{image1}
 * \oport{image2}
 *
 */
class KWIVER_PROCESSES_NO_EXPORT merge_images_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "merge_images",
               "Merge multiple images into one.\n\n"
               "This process merges all the channels in two input "
               "images into a single output image based on the "
               "implementation "
               "of the 'merge_images' algorithm selected.\n"
               "This process has two input ports accepting "
               "the images. The first two connect commands will be "
               "accepted as the input ports. The actual input port "
               "names do not matter.\n"
               "The channels from the image from the first port "
               "connected will be added to the output image first."
    )

  merge_images_process( kwiver::vital::config_block_sptr const& config );
  virtual ~merge_images_process();

protected:
  void _configure() override;
  void _step() override;
  void input_port_undefined(port_t const& port) override;

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class merge_images_process


} // end namespace
