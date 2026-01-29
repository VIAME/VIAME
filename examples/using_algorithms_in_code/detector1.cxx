

#include <vital/config/config_block_io.h>
#include <arrows/ocv/image_container.h>
#include <arrows/ocv/algo/image_io.h>
#include <arrows/ocv/algo/hough_circle_detector.h>

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
  kwiver::vital::algo::image_io_sptr image_reader( new kwiver::arrows::ocv::image_io() );

  // (3) Read the image
  kwiver::vital::image_container_sptr the_image = image_reader->load( filename );

  // (4) Create the detector
  kwiver::vital::algo::image_object_detector_sptr detector( new kwiver::arrows::ocv::hough_circle_detector() );

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
