/*ckwg +29
 * Copyright 2013-2016 by Kitware, Inc.
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
 * \brief test OCV image class
 */

#include <test_common.h>

#include <arrows/ocv/register_algorithms.h>
#include <arrows/ocv/image_container.h>
#include <arrows/ocv/image_io.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

using namespace kwiver::vital;

IMPLEMENT_TEST(factory)
{
  using namespace kwiver::arrows;
  ocv::register_algorithms();
  algo::image_io_sptr img_io = kwiver::vital::algo::image_io::create("ocv");
  if (!img_io)
  {
    TEST_ERROR("Unable to create image_io algorithm of type ocv");
  }
  algo::image_io* img_io_ptr = img_io.get();
  if (typeid(*img_io_ptr) != typeid(ocv::image_io))
  {
    TEST_ERROR("Factory method did not construct the correct type");
  }
}


namespace {

// helper function to populate the image with a pattern
// the dynamic range is stretched between minv and maxv
template <typename T>
void
populate_ocv_image(cv::Mat& img, T minv, T maxv)
{
  const double range = static_cast<double>(maxv) - static_cast<double>(minv);
  const double offset = - minv;
  const unsigned num_c = img.channels();
  for( unsigned int p=0; p<num_c; ++p )
  {
    for( unsigned int j=0; j<static_cast<unsigned int>(img.rows); ++j )
    {
      for( unsigned int i=0; i<static_cast<unsigned int>(img.cols); ++i )
      {
        const double pi = 3.14159265358979323846;
        double val = ((std::sin(pi*double(i)*(p+1)/10) * std::sin(pi*double(j)*(p+1)/10))+1) / 2;
        img.template ptr<T>(j)[num_c * i + p] = static_cast<T>(val * range + offset);
      }
    }
  }
}


// helper function to populate the image with a pattern
template <typename T>
void
populate_ocv_image(cv::Mat& img)
{
  const T minv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::min() : T(0);
  const T maxv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::max() : T(1);
  populate_ocv_image(img, minv, maxv);
}


// helper function to populate the image with a pattern
// the dynamic range is stretched between minv and maxv
template <typename T>
void
populate_vital_image(kwiver::vital::image& img, T minv, T maxv)
{
  const double range = static_cast<double>(maxv) - static_cast<double>(minv);
  const double offset = - minv;
  for( unsigned int p=0; p<img.depth(); ++p )
  {
    for( unsigned int j=0; j<img.height(); ++j )
    {
      for( unsigned int i=0; i<img.width(); ++i )
      {
        const double pi = 3.14159265358979323846;
        double val = ((std::sin(pi*double(i)*(p+1)/10) * std::sin(pi*double(j)*(p+1)/10))+1) / 2;
        img.at<T>(i,j,p) = static_cast<T>(val * range + offset);
      }
    }
  }
}


// helper function to populate the image with a pattern
template <typename T>
void
populate_vital_image(kwiver::vital::image& img)
{
  const T minv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::min() : T(0);
  const T maxv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::max() : T(1);
  populate_vital_image<T>(img, minv, maxv);
}


template <typename T>
void
run_ocv_conversion_tests(const cv::Mat& img, const std::string& type_str)
{
  using namespace kwiver::arrows;
  // convert to a vital image and verify that the properties are correct
  kwiver::vital::image vimg =  ocv::image_container::ocv_to_vital(img);
  TEST_EQUAL("OpenCV image conversion of type "+type_str+" has the correct bit depth",
             vimg.pixel_traits().num_bytes, sizeof(T));
  TEST_EQUAL("OpenCV image conversion of type "+type_str+" has the correct pixel type",
             vimg.pixel_traits().type, image_pixel_traits_of<T>::static_type);
  TEST_EQUAL("OpenCV image conversion of type "+type_str+" has the correct number of planes",
             vimg.depth(), static_cast< size_t >(img.channels()));
  TEST_EQUAL("OpenCV image conversion of type "+type_str+" has the correct width",
             vimg.height(), static_cast< size_t >(img.rows));
  TEST_EQUAL("OpenCV image conversion of type "+type_str+" has the correct height",
             vimg.width(), static_cast< size_t >(img.cols));
  TEST_EQUAL("OpenCV image conversion of type "+type_str+" has the same memory",
             vimg.first_pixel() == img.data, true);
  bool equal_data = true;
  const unsigned num_c = img.channels();
  for( unsigned int d=0; equal_data && d<vimg.depth(); ++d )
  {
    for( unsigned int j=0; equal_data && j<vimg.height(); ++j )
    {
      for( unsigned int i=0; equal_data && i<vimg.width(); ++i )
      {
        if( img.ptr<T>(j)[num_c * i + d] != vimg.at<T>(i,j,d) )
        {
          equal_data = false;
        }
      }
    }
  }
  TEST_EQUAL("OpenCV image conversion of type "+type_str+" has the same values",
             equal_data, true);

  // convert back to cv::Mat and test again
  cv::Mat img2 = ocv::image_container::vital_to_ocv(vimg);
  if( !img2.data)
  {
    TEST_ERROR("OpenCV image re-conversion of type "+type_str+" did not produce a valid cv::Mat");
    return;
  }

  TEST_EQUAL("OpenCV image re-conversion of type "+type_str+" has the correct pixel format",
             img.type(), img2.type());
  std::vector<cv::Mat> channels1(img.channels()), channels2(img2.channels());
  cv::split(img, channels1);
  cv::split(img2, channels2);
  unsigned int num_diff = 0;
  for (unsigned d=0; d<channels1.size(); ++d)
  {
    num_diff += cv::countNonZero( channels1[d] != channels2[d]);
  }
  TEST_EQUAL("OpenCV image re-conversion of type "+type_str+" is identical",
             num_diff, 0);
  TEST_EQUAL("OpenCV image re-conversion of type "+type_str+" has the same memory",
             img.data == img2.data, true);
}


template <typename T>
void
run_vital_conversion_tests(const kwiver::vital::image_of<T>& img,
                           const std::string& type_str,
                           bool requires_copy = false)
{
  using namespace kwiver::arrows;
  // convert to a cv::Mat and verify that the properties are correct
  cv::Mat ocv_img =  ocv::image_container::vital_to_ocv(img);
  if( !ocv_img.data )
  {
    TEST_ERROR("Vital image conversion of type "+type_str+" did not produce a valid cv::Mat");
    return;
  }
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the correct pixel format",
             ocv_img.type()%8, cv::Mat_<T>(1,1).type()%8);
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the correct number of planes",
             static_cast< size_t >(ocv_img.channels()), img.depth());
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the correct width",
             static_cast< size_t >(ocv_img.rows), img.height());
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the correct height",
             static_cast< size_t >(ocv_img.cols), img.width());
  if( !requires_copy )
  {
    TEST_EQUAL("Vital image conversion of type "+type_str+" has the same memory",
               reinterpret_cast<T *>(ocv_img.data) == img.first_pixel(), true);
  }
  bool equal_data = true;
  const unsigned num_c = ocv_img.channels();
  for( unsigned int d=0; equal_data && d<img.depth(); ++d )
  {
    for( unsigned int j=0; equal_data && j<img.height(); ++j )
    {
      for( unsigned int i=0; equal_data && i<img.width(); ++i )
      {
        if( img(i,j,d) != ocv_img.ptr<T>(j)[num_c * i + d] )
        {
          std::cout << "Pixel "<<i<<", "<<j<<", "<<d<<" has values "
                    <<int(img(i,j,d))<<" != "<< int(ocv_img.ptr<T>(j)[num_c * i + d]) <<std::endl;
          equal_data = false;
        }
      }
    }
  }
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the same values",
             equal_data, true);

  // convert back to vital::image and test again
  kwiver::vital::image img2 = ocv::image_container::ocv_to_vital(ocv_img);
  TEST_EQUAL("Vital image re-conversion of type "+type_str+" has the correct bit depth",
             img2.pixel_traits().num_bytes, sizeof(T));
  TEST_EQUAL("Vital image re-conversion of type "+type_str+" has the correct pixel type",
             img2.pixel_traits().type, image_pixel_traits_of<T>::static_type);
  TEST_EQUAL("Vital image re-conversion of type "+type_str+" is identical",
             kwiver::vital::equal_content(img, img2), true);
  TEST_EQUAL("Vital image re-conversion of type "+type_str+" has the same memory",
             reinterpret_cast<T *>(ocv_img.data) == img2.first_pixel(), true);
}


template <typename T>
void
test_conversion(const std::string& type_str)
{
  // create cv::Mat and convert to an from vital images
  {
    std::cout << "Testing single channel cv::Mat of type " << type_str << std::endl;
    cv::Mat_<T> img(100,200);
    populate_ocv_image<T>(img);
    run_ocv_conversion_tests<T>(img, type_str);
  }

  {
    std::cout << "Testing three channel cv::Mat of type " << type_str << std::endl;
    cv::Mat_<cv::Vec<T,3> > img(100,200);
    populate_ocv_image<T>(img);
    run_ocv_conversion_tests<T>(img, type_str);
  }

  {
    std::cout << "Testing cropped cv::Mat of type " << type_str << std::endl;
    cv::Mat_<T> img(200,300);
    populate_ocv_image<T>(img);
    cv::Rect window( cv::Point(40,50), cv::Point(240, 150) );
    cv::Mat_<T> img_crop(img, window);
    run_ocv_conversion_tests<T>(img_crop, type_str+" (cropped)");
  }

  // create vital images and convert to an from cv::Mat
  // Note: different code paths are taken depending on whether the image
  // is natively created as OpenCV or vital, so we need to test both ways.
  {
    std::cout << "Testing single channel vital::image of type " << type_str << std::endl;
    kwiver::vital::image_of<T> img(200, 300, 1);
    populate_vital_image<T>(img);
    run_vital_conversion_tests(img, type_str);
  }

  {
    std::cout << "Testing three channel vital::image of type " << type_str << std::endl;
    kwiver::vital::image_of<T> img(200, 300, 3);
    populate_vital_image<T>(img);
    run_vital_conversion_tests(img, type_str, true);
  }

  {
    std::cout << "Testing interleaved vital::image of type " << type_str << std::endl;
    kwiver::vital::image_of<T> img(200, 300, 3, true);
    populate_vital_image<T>(img);
    run_vital_conversion_tests(img, type_str+" (interleaved)");
  }
}

} // end anonymous namespace


IMPLEMENT_TEST(image_convert)
{
  using namespace kwiver;
  using namespace kwiver::arrows;
  test_conversion<uint8_t>("uint8");
  test_conversion<int8_t>("int8");
  test_conversion<uint16_t>("uint16");
  test_conversion<int16_t>("int16");
  test_conversion<int32_t>("int32");
  test_conversion<float>("float");
  test_conversion<double>("double");

  // some types not supported by OpenCV and should throw an exception
  std::cout << "Test conversion of types not supported by OpenCV" << std::endl;
  EXPECT_EXCEPTION(vital::image_type_mismatch_exception,
                   ocv::image_container::vital_to_ocv(vital::image_of<uint32_t>(200, 300)),
                   "converting uint32_t image to cv::Mat");
  EXPECT_EXCEPTION(vital::image_type_mismatch_exception,
                   ocv::image_container::vital_to_ocv(vital::image_of<int64_t>(200, 300)),
                   "converting int64_t image to cv::Mat");
  EXPECT_EXCEPTION(vital::image_type_mismatch_exception,
                   ocv::image_container::vital_to_ocv(vital::image_of<uint64_t>(200, 300)),
                   "converting uint64_t image to cv::Mat");
}


namespace {

template <typename T>
void
run_image_io_tests(kwiver::vital::image_of<T> const& img, std::string const& type_str)
{
  using namespace kwiver::arrows;
  const std::string image_path = "test_"+type_str+".png";
  image_container_sptr c(new simple_image_container(img));
  ocv::image_io io;
  io.save(image_path, c);
  image_container_sptr c2 = io.load(image_path);
  kwiver::vital::image img2 = c2->get_image();
  TEST_EQUAL("Image of type "+type_str+" has same type after saving and loading",
             img2.pixel_traits(), img.pixel_traits());
  TEST_EQUAL("Image of type "+type_str+" has same number of channels after saving and loading",
             img2.depth(), img.depth());
  TEST_EQUAL("Image of type "+type_str+" has same content after saving and loading",
             equal_content(img, img2), true);

  if( std::remove(image_path.c_str()) != 0 )
  {
    TEST_ERROR("Unable to delete temporary image file.");
  }
}

} // end anonymous namespace


IMPLEMENT_TEST(image_io_types)
{
  {
    kwiver::vital::image_of<uint8_t> img(200,300,1);
    populate_vital_image<uint8_t>(img);
    run_image_io_tests(img, "uint8_C1");
  }
  {
    kwiver::vital::image_of<uint8_t> img(200,300,3);
    populate_vital_image<uint8_t>(img);
    run_image_io_tests(img, "uint8_C3");
  }
  {
    kwiver::vital::image_of<uint8_t> img(200,300,4);
    populate_vital_image<uint8_t>(img);
    run_image_io_tests(img, "uint8_C4");
  }
  {
    kwiver::vital::image_of<uint16_t> img(200,300,1);
    populate_vital_image<uint16_t>(img);
    run_image_io_tests(img, "uint16_C1");
  }
  {
    kwiver::vital::image_of<uint16_t> img(200,300,3);
    populate_vital_image<uint16_t>(img);
    run_image_io_tests(img, "uint16_C3");
  }
}
