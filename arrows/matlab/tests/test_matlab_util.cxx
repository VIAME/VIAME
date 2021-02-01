// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test detected_object class
 */

#include <test_common.h>

#include <vital/types/detected_object.h>
#include <arrows/matlab/matlab_util.h>
#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define DEBUG 0

#define TEST_ARGS ( kwiver::vital::path_t const &data_dir )

DECLARE_TEST_MAP();

int
main(int argc, char** argv)
{
  CHECK_ARGS(2);

  testname_t const testname = argv[1];
  kwiver::vital::path_t data_dir( argv[2] );

  RUN_TEST(testname, data_dir);
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(image_conversion)
{
  kwiver::vital::path_t test_read_file = data_dir + "/test_kitware_logo.jpg";

  cv::Mat ocv_image;
  ocv_image = cv::imread(test_read_file, CV_LOAD_IMAGE_COLOR);   // Read the file
  if(! ocv_image.data )                              // Check for invalid input
  {
    TEST_ERROR( "Could not open or find the image" );
    return;
  }

  auto ic_sptr = std::make_shared< kwiver::arrows::ocv::image_container >(
    ocv_image,
    kwiver::arrows::ocv::image_container::ColorMode::BGR_COLOR );

#if DEBUG
  cv::namedWindow( "input OCV image", cv::WINDOW_AUTOSIZE ); // Create a window for display.
  cv::imshow( "input OCV image", ocv_image ); // Show our image inside it.
  cv::waitKey( 000 ); // pause for keystroke
#endif

  kwiver::arrows::matlab::MxArraySptr mx_image = kwiver::arrows::matlab::convert_mx_image( ic_sptr );

  auto ocv_ic = kwiver::arrows::matlab::convert_mx_image( mx_image );
  cv::Mat ocv_ic_mat = kwiver::arrows::ocv::image_container::vital_to_ocv(
    ocv_ic->get_image(),
    kwiver::arrows::ocv::image_container::ColorMode::BGR_COLOR );

#if DEBUG
  cv::namedWindow( "output OCV image", cv::WINDOW_AUTOSIZE ); // Create a window for display.
  cv::imshow( "output OCV image", ocv_ic_mat ); // Show our image inside it.
  cv::waitKey( 000 ); // pause for keystroke
#endif

  // Test to see if the images are the same
  bool isEqual = (cv::sum(ocv_image != ocv_ic_mat) == cv::Scalar(0,0,0,0));
  if ( ! isEqual )
  {
    TEST_ERROR( "Images fail comparison." );
  }
}
