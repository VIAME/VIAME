

#include <viame/algorithm_framework/algorithm_plugin_manager.h>
#include <viame/algorithm_framework/config/config_block_io.h>
#include <viame/algorithm_framework/algo/image_object_detector.h>
#include <viame/image_io/core_image_io.h>

#include <string>

int main( int argc, char* argv[] )
{
  // (1) Create logger to use for reporting errors and other diagnostics.
  viame::logger_handle_t logger( viame::get_logger( "detector_test" ));

  // (2) Initialize and load all discoverable plugins
  viame::algorithm_plugin_manager::load_plugins_once();

  // (3) get file name for input image
  std::string filename = argv[1];

  // (4) Look for name of config file as second parameter
  viame::config_block_sptr config;
  config = viame::read_config_file( argv[2] );

  // (5) create image reader
  viame::algo::image_io_sptr image_reader( new viame::core_image_io() );

  // (6) Read the image
  viame::image_container_sptr the_image = image_reader->load( filename );

  // (7) Create the detector
  viame::algo::image_object_detector_sptr detector;
  viame::algo::image_object_detector::set_nested_algo_configuration( "detector", config, detector );

  if ( ! detector )
  {
    LOG_ERROR( logger, "Unable to create detector" );
    return 1;
  }

  viame::algo::image_object_detector::get_nested_algo_configuration( "detector", config, detector );

  // Check config so it will give run-time diagnostic of config problems
  if ( ! viame::algo::image_object_detector::check_nested_algo_configuration( "detector", config ) )
  {
    LOG_ERROR( logger, "Configuration check failed." );
    return 1;
  }

  // (5) Send image to detector and get detections.
  viame::detected_object_set_sptr detections = detector->detect( the_image );

  // (6) See what was detected
  std::cout << "There were " << detections->size() << " detections in the image." << std::endl;


  return 0;
}
