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
 * \brief test VXL image class functionality
 */

#include <test_common.h>

#include <arrows/vxl/register_algorithms.h>
#include <arrows/vxl/image_container.h>
#include <arrows/vxl/image_io.h>

#include <vil/vil_crop.h>

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
  vxl::register_algorithms();

  kwiver::vital::algo::image_io_sptr img_io = kwiver::vital::algo::image_io::create("vxl");
  if (!img_io)
  {
    TEST_ERROR("Unable to create image_io algorithm of type vxl");
  }
  algo::image_io* img_io_ptr = img_io.get();
  if (typeid(*img_io_ptr) != typeid(vxl::image_io))
  {
    TEST_ERROR("Factory method did not construct the correct type");
  }
}


// helper function to populate the image with a pattern
template <typename T>
void
populate_vil_image(vil_image_view<T>& img)
{
  const T minv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::min() : T(0);
  const T maxv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::max() : T(1);

  const double range = static_cast<double>(maxv) - static_cast<double>(minv);
  const double offset = - minv;
  for( unsigned int p=0; p<img.nplanes(); ++p )
  {
    for( unsigned int j=0; j<img.nj(); ++j )
    {
      for( unsigned int i=0; i<img.ni(); ++i )
      {
        double val = ((i/(5*(p+1)))%2) + ((j/(5*(p+1)))%2);
        img(i,j,p) = static_cast<T>(val / 2 * range + offset);
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

  const double range = static_cast<double>(maxv) - static_cast<double>(minv);
  const double offset = - minv;
  for( unsigned int p=0; p<img.depth(); ++p )
  {
    for( unsigned int j=0; j<img.height(); ++j )
    {
      for( unsigned int i=0; i<img.width(); ++i )
      {
        double val = ((i/(5*(p+1)))%2) + ((j/(5*(p+1)))%2);
        img.at<T>(i,j,p) = static_cast<T>(val / 2 * range + offset);
      }
    }
  }
}


IMPLEMENT_TEST(image_io)
{
  using namespace kwiver::arrows;
  kwiver::vital::image_of<kwiver::vital::byte> img(200,300,3);
  populate_vital_image<kwiver::vital::byte>(img);
  image_container_sptr c(new simple_image_container(img));
  vxl::image_io io;
  io.save("test.png", c);
  image_container_sptr c2 = io.load("test.png");
  kwiver::vital::image img2 = c2->get_image();
  if( ! equal_content(img, img2) )
  {
    TEST_ERROR("Saved image is not identical to loaded image");
  }
  if( std::remove("test.png") != 0 )
  {
    TEST_ERROR("Unable to delete temporary image file.");
  }
}


template <typename T>
void
run_vil_conversion_tests(const vil_image_view<T>& img, const std::string& type_str)
{
  using namespace kwiver::arrows;
  // convert to a vital image and verify that the properties are correct
  kwiver::vital::image vimg =  vxl::image_container::vxl_to_vital(img);
  TEST_EQUAL("VXL image conversion of type "+type_str+" has the correct bit depth",
             vimg.pixel_traits().num_bytes, sizeof(T));
  TEST_EQUAL("VXL image conversion of type "+type_str+" has the correct pixel type",
             vimg.pixel_traits().type, image_pixel_traits_of<T>::static_type);
  TEST_EQUAL("VXL image conversion of type "+type_str+" has the correct number of planes",
             vimg.depth(), img.nplanes());
  TEST_EQUAL("VXL image conversion of type "+type_str+" has the correct width",
             vimg.height(), img.nj());
  TEST_EQUAL("VXL image conversion of type "+type_str+" has the correct height",
             vimg.width(), img.ni());
  TEST_EQUAL("VXL image conversion of type "+type_str+" has the same memory",
             vimg.first_pixel(), img.top_left_ptr());
  bool equal_data = true;
  for( unsigned int p=0; equal_data && p<img.nplanes(); ++p )
  {
    for( unsigned int j=0; equal_data && j<img.nj(); ++j )
    {
      for( unsigned int i=0; equal_data && i<img.ni(); ++i )
      {
        if( img(i,j,p) != vimg.at<T>(i,j,p) )
        {
          equal_data = false;
        }
      }
    }
  }
  TEST_EQUAL("VXL image conversion of type "+type_str+" has the same values",
             equal_data, true);

  // convert back to VXL and test again
  vil_image_view<T> img2 = vxl::image_container::vital_to_vxl(vimg);
  if( !img2 )
  {
    TEST_ERROR("VXL image re-conversion of type "+type_str+" did not produce a valid vil_image_view");
    return;
  }
 
  TEST_EQUAL("VXL image re-conversion of type "+type_str+" has the correct pixel format",
             img.pixel_format(), img2.pixel_format());
  TEST_EQUAL("VXL image re-conversion of type "+type_str+" is identical",
             vil_image_view_deep_equality(img, img2), true);
  TEST_EQUAL("VXL image re-conversion of type "+type_str+" has the same memory",
             img.top_left_ptr(), img2.top_left_ptr());
}


template <typename T>
void
run_vital_conversion_tests(const kwiver::vital::image_of<T>& img, const std::string& type_str)
{
  using namespace kwiver::arrows;
  // convert to a vil image and verify that the properties are correct
  vil_image_view<T> vimg =  vxl::image_container::vital_to_vxl(img);
  if( !vimg )
  {
    TEST_ERROR("Vital image conversion of type "+type_str+" did not produce a valid vil_image_view");
    return;
  }
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the correct pixel format",
             vimg.pixel_format(), vil_pixel_format_of(T()));
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the correct number of planes",
             vimg.nplanes(), img.depth());
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the correct width",
             vimg.nj(), img.height());
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the correct height",
             vimg.ni(), img.width());
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the same memory",
             vimg.top_left_ptr(), img.first_pixel());
  bool equal_data = true;
  for( unsigned int p=0; equal_data && p<vimg.nplanes(); ++p )
  {
    for( unsigned int j=0; equal_data && j<vimg.nj(); ++j )
    {
      for( unsigned int i=0; equal_data && i<vimg.ni(); ++i )
      {
        if( img(i,j,p) != vimg(i,j,p) )
        {
          std::cout << "Pixel "<<i<<", "<<j<<", "<<p<<" has values "
                    <<img(i,j,p)<<" != "<<vimg(i,j,p)<<std::endl;
          equal_data = false;
        }
      }
    }
  }
  TEST_EQUAL("Vital image conversion of type "+type_str+" has the same values",
             equal_data, true);

  // convert back to VXL and test again
  kwiver::vital::image img2 = vxl::image_container::vxl_to_vital(vimg);
  TEST_EQUAL("Vital image re-conversion of type "+type_str+" has the correct bit depth",
             img2.pixel_traits().num_bytes, sizeof(T));
  TEST_EQUAL("Vital image re-conversion of type "+type_str+" has the correct pixel type",
             img2.pixel_traits().type, image_pixel_traits_of<T>::static_type);
  TEST_EQUAL("Vital image re-conversion of type "+type_str+" is identical",
             kwiver::vital::equal_content(img, img2), true);
  TEST_EQUAL("Vital image re-conversion of type "+type_str+" has the same memory",
             img.first_pixel(), img2.first_pixel());
}


template <typename T>
void
test_conversion(const std::string& type_str)
{
  // create vil_image_view and convert to an from vital images
  {
    vil_image_view<T> img(100,200,3);
    populate_vil_image(img);
    run_vil_conversion_tests(img, type_str);
  }

  {
    vil_image_view<T> img(100,200,3,true);
    populate_vil_image(img);
    run_vil_conversion_tests(img, type_str+" (interleaved)");
  }

  {
    vil_image_view<T> img(200,300,3);
    populate_vil_image(img);
    vil_image_view<T> img_crop = vil_crop(img, 50, 100, 40, 200);
    run_vil_conversion_tests(img_crop, type_str+" (cropped)");
  }

  // create vital images and convert to an from vil_image_view
  // Note: different code paths are taken depending on whether the image
  // is natively created as vil or vital, so we need to test both ways.
  {
    kwiver::vital::image_of<T> img(200, 300, 3);
    populate_vital_image<T>(img);
    run_vital_conversion_tests(img, type_str);
  }

  {
    kwiver::vital::image_of<T> img(200, 300, 3, true);
    populate_vital_image<T>(img);
    run_vital_conversion_tests(img, type_str+" (interleaved)");
  }
}


IMPLEMENT_TEST(image_convert)
{
  using namespace kwiver::arrows;
  test_conversion<vxl_byte>("byte");
  test_conversion<vxl_sbyte>("signed byte");
  test_conversion<vxl_uint_16>("uint_16");
  test_conversion<vxl_int_16>("int_16");
  test_conversion<vxl_uint_32>("uint_32");
  test_conversion<vxl_int_32>("int_32");
  test_conversion<vxl_uint_64>("uint_64");
  test_conversion<vxl_int_64>("int_64");
  test_conversion<float>("float");
  test_conversion<double>("double");
  test_conversion<bool>("bool");
}
