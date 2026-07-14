/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */


#include <vital/plugin_management/plugin_manager.h>
#include <vital/config/config_block_io.h>
#include <vital/algo/image_object_detector.h>
#include <vital/algo/algorithm.txx>
#include <arrows/ocv/image_container.h>
#include <arrows/ocv/algo/image_io.h>

#include <string>

namespace kv = kwiver::vital;
namespace kva = kwiver::vital::algo;

int main( int argc, char* argv[] )
{
  // (1) Create logger to use for reporting errors and other diagnostics.
  kv::logger_handle_t logger( kv::get_logger( "detector_test" ));

  // (2) Initialize and load all discoverable plugins
  kv::plugin_manager::instance().load_all_plugins();

  // (3) get file name for input image
  std::string filename = argv[1];

  // (4) Look for name of config file as second parameter
  kv::config_block_sptr config;
  config = kv::read_config_file( argv[2] );

  // (5) create image reader
  kva::image_io_sptr image_reader( new kwiver::arrows::ocv::image_io() );

  // (6) Read the image
  kv::image_container_sptr the_image = image_reader->load( filename );

  // (7) Create the detector
  kva::image_object_detector_sptr detector;
  kv::set_nested_algo_configuration< kva::image_object_detector >(
    "detector", config, detector );

  if ( ! detector )
  {
    LOG_ERROR( logger, "Unable to create detector" );
    return 1;
  }

  kv::get_nested_algo_configuration< kva::image_object_detector >(
    "detector", config, detector );

  // Check config so it will give run-time diagnostic of config problems
  if ( ! kv::check_nested_algo_configuration< kva::image_object_detector >(
           "detector", config ) )
  {
    LOG_ERROR( logger, "Configuration check failed." );
    return 1;
  }

  // (5) Send image to detector and get detections.
  kv::detected_object_set_sptr detections = detector->detect( the_image );

  // (6) See what was detected
  std::cout << "There were " << detections->size() << " detections in the image." << std::endl;

  return 0;
}
