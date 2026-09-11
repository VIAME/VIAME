

#include <viame/algorithm_framework/config/config_block_io.h>
#include <viame/video_io/core_image_io.h>
#include <viame/object_detectors/detect_heat_map.h>

#include <string>

int main( int argc, char* argv[] )
{
  // (1) get file name for input image
  std::string filename = argv[1];

  // (1.1) Look for name of config file as second parameter
  kwiver::vital::config_block_sptr config;
  if ( argc > 2 )
  {
    config = kwiver::vital::read_config_file( argv[2] );
  }

  // (2) create image reader
  kwiver::vital::algo::image_io_sptr image_reader( new viame::core_image_io() );

  // (3) Read the image
  kwiver::vital::image_container_sptr the_image = image_reader->load( filename );

  // (4) Create the detector
  //
  // Constructed directly rather than through the plugin manager, which is
  // what this example is for; detector3 shows the other way. It was
  // `hough_circle` until P7-T04 moved that implementation to python, and a
  // python implementation has no class to construct here.
  kwiver::vital::algo::image_object_detector_sptr detector( new kwiver::arrows::ocv::detect_heat_map() );

  // (4.1) If there was a config structure, then pass it to the algorithm.
  if (config)
  {
    detector->set_configuration( config );
  }

  // (5) Send image to detector and get detections.
  kwiver::vital::detected_object_set_sptr detections = detector->detect( the_image );

  // (6) See what was detected
  std::cout << "There were " << detections->size() << " detections in the image." << std::endl;

  return 0;
}
