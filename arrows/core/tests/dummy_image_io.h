/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#ifndef ARROWS_CORE_TEST_DUMMY_IMAGE_IO_H
#define ARROWS_CORE_TEST_DUMMY_IMAGE_IO_H

#include <vital/algo/image_io.h>
#include <vital/algo/algorithm_factory.h>


/// A dummy image_io algorithm that only checks for valid paths
/**
 * This dummy implementation is used as a stand-in for image_io in core unit
 * tests because core does not provide any image_io implementations.
 */
class image_io_dummy
  : public kwiver::vital::algorithm_impl<image_io_dummy, kwiver::vital::algo::image_io>
{
public:
  /// Constructor
  image_io_dummy() {}

  /// Destructor
  virtual ~image_io_dummy() {}

  /// Set this algorithm's properties via a config block
  virtual void set_configuration(kwiver::vital::config_block_sptr config) {}
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(kwiver::vital::config_block_sptr config) const { return true; }


private:
  /// Implementation specific load functionality.
  virtual kwiver::vital::image_container_sptr
  load_(const std::string& filename) const
  {
    LOG_DEBUG( logger(), "image_io_dummy::load_() got file: " << filename );
    return kwiver::vital::image_container_sptr();
  }

  /// Implementation specific save functionality.
  virtual void
  save_(const std::string& filename,
        kwiver::vital::image_container_sptr data) const
  {
    LOG_DEBUG( logger(), "image_io_dummy::save_() got file: " << filename );
  }
};


/// Register the dummy_image_io implementation with the plugin manager
void register_dummy_image_io()
{
  // register the dummy image_io defined above
  auto& vpm = kwiver::vital::plugin_manager::instance();
  auto fact = vpm.ADD_ALGORITHM( "dummy", image_io_dummy );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "A dummy implementation of an image_io algorithm for testing" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, "test" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
}

#endif /* ARROWS_CORE_TEST_DUMMY_IMAGE_IO_H */
