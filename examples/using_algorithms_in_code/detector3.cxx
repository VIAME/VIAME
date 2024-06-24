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


#include <vital/plugin_loader/plugin_manager.h>
#include <vital/config/config_block_io.h>
#include <vital/algo/image_object_detector.h>
#include <arrows/ocv/image_container.h>
#include <arrows/ocv/image_io.h>

#include <string>

int main( int argc, char* argv[] )
{
  // (1) Create logger to use for reporting errors and other diagnostics.
  kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger( "detector_test" ));

  // (2) Initialize and load all discoverable plugins
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  // (3) get file name for input image
  std::string filename = argv[1];

  // (4) Look for name of config file as second parameter
  kwiver::vital::config_block_sptr config;
  config = kwiver::vital::read_config_file( argv[2] );

  // (5) create image reader
  kwiver::vital::algo::image_io_sptr image_reader( new kwiver::arrows::ocv::image_io() );

  // (6) Read the image
  kwiver::vital::image_container_sptr the_image = image_reader->load( filename );

  // (7) Create the detector
  kwiver::vital::algo::image_object_detector_sptr detector;
  kwiver::vital::algo::image_object_detector::set_nested_algo_configuration( "detector", config, detector );

  if ( ! detector )
  {
    LOG_ERROR( logger, "Unable to create detector" );
    return 1;
  }

  kwiver::vital::algo::image_object_detector::get_nested_algo_configuration( "detector", config, detector );

  // Check config so it will give run-time diagnostic of config problems
  if ( ! kwiver::vital::algo::image_object_detector::check_nested_algo_configuration( "detector", config ) )
  {
    LOG_ERROR( logger, "Configuration check failed." );
    return 1;
  }

  // (5) Send image to detector and get detections.
  kwiver::vital::detected_object_set_sptr detections = detector->detect( the_image );

  // (6) See what was detected
  std::cout << "There were " << detections->size() << " detections in the image." << std::endl;

  return 0;
}
