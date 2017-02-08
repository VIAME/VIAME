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
#include <vital/plugin_loader/plugin_manager.h>

#include <arrows/vxl/image_container.h>
#include <arrows/vxl/image_io.h>
#include <vital/util/transform_image.h>

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
  kwiver::vital::plugin_manager::instance().load_all_plugins();

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

namespace {

// helper functor for use in transform_image
template <typename T>
class scale_offset {
private:
  double scale;
  double offset;

public:
  scale_offset(double s, double o) : scale(s), offset(o) { }

  T operator () (T const& val) const
  {
    return static_cast<T>(scale * val + offset);
  }
};


// helper functor for use in transform_image
// This funcion mimics the vil_convert_stretch_range_limited operation
template <typename T>
class range_to_byte {
private:
  double scale;
  T lo;
  T hi;

public:
  range_to_byte(double minv, double maxv) : scale(255.0 / (maxv - minv)), lo(minv), hi(maxv) { }

  uint8_t operator () (T const& val) const
  {
    return val <= lo ? 0 : static_cast<uint8_t>( val >= hi ? 255 : (scale * (val - lo) + 0.5) );
  }
};



// helper function to populate the image with a pattern
// the dynamic range is stretched between minv and maxv
template <typename T>
void
populate_vil_image(vil_image_view<T>& img, T minv, T maxv)
{
  const double range = static_cast<double>(maxv) - static_cast<double>(minv);
  const double offset = - static_cast<double>(minv);
  for( unsigned int p=0; p<img.nplanes(); ++p )
  {
    for( unsigned int j=0; j<img.nj(); ++j )
    {
      for( unsigned int i=0; i<img.ni(); ++i )
      {
        const double pi = 3.14159265358979323846;
        double val = ((std::sin(pi*double(i)*(p+1)/10) * std::sin(pi*double(j)*(p+1)/10))+1) / 2;
        img(i,j,p) = static_cast<T>(val * range + offset);
      }
    }
  }
}


// helper function to populate the image with a pattern
template <typename T>
void
populate_vil_image(vil_image_view<T>& img)
{
  const T minv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::min() : T(0);
  const T maxv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::max() : T(1);
  populate_vil_image(img, minv, maxv);
}


// helper function to populate the image with a pattern
// the dynamic range is stretched between minv and maxv
template <typename T>
void
populate_vital_image(kwiver::vital::image& img, T minv, T maxv)
{
  const double range = static_cast<double>(maxv) - static_cast<double>(minv);
  const double offset = - static_cast<double>(minv);
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
run_image_io_tests(kwiver::vital::image_of<T> const& img, std::string const& type_str)
{
  using namespace kwiver::arrows;
  const std::string image_path = "test_"+type_str+".tiff";
  image_container_sptr c(new simple_image_container(img));
  vxl::image_io io;
  io.save(image_path, c);
  image_container_sptr c2 = io.load(image_path);
  kwiver::vital::image img2 = c2->get_image();
  TEST_EQUAL("Image of type "+type_str+" has same type after saving and loading",
             img.pixel_traits(), img2.pixel_traits());
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
    kwiver::vital::image_of<kwiver::vital::byte> img(200,300,3);
    populate_vital_image<kwiver::vital::byte>(img);
    run_image_io_tests(img, "uint8");
  }
  {
    kwiver::vital::image_of<float> img(200,300,3);
    populate_vital_image<float>(img);
    run_image_io_tests(img, "float");
  }
  {
    // currently VXL support only single channel double TIFFs
    kwiver::vital::image_of<double> img(200,300,1);
    populate_vital_image<double>(img);
    run_image_io_tests(img, "double");
  }
  {
    kwiver::vital::image_of<uint16_t> img(200,300,3);
    populate_vital_image<uint16_t>(img);
    run_image_io_tests(img, "uint16_t");
  }
  {
    // currently VXL support only single channel boolean TIFFs
    kwiver::vital::image_of<bool> img(200,300,1);
    populate_vital_image<bool>(img);
    run_image_io_tests(img, "bool");
  }
}


IMPLEMENT_TEST(image_io_stretch)
{
  using namespace kwiver;

  // an image with 12-bit data in a 16-bit image
  vital::image_of<uint16_t> img12(200,300,3);
  populate_vital_image<uint16_t>(img12, 0, 4095);
  image_container_sptr c(new simple_image_container(img12));

  vital::image_of<uint8_t> img8(200,300,3);
  vital::image_of<uint16_t> img16(200,300,3);
  img16.copy_from(img12);
  double scale = 255.0 / 4095.0;
  vital::transform_image(img16, scale_offset<uint16_t>(scale, 0));
  cast_image(img16, img8);
  img16.copy_from(img12);
  scale = (65536.0 - 1e-6) / 4095.0;
  vital::transform_image(img16, scale_offset<uint16_t>(scale, 0));

  // save a 12-bit image
  arrows::vxl::image_io io;
  io.save("test12.tiff", c);
  std::cout << "wrote 12-bit test image" <<std::endl;

  vital::config_block_sptr config = vital::config_block::empty_config();
  config->set_value("auto_stretch", true);
  io.set_configuration(config);
  c = io.load("test12.tiff");
  vital::image img_loaded = c->get_image();
  TEST_EQUAL("12-bit image is represented as 16-bits after saving and loading",
             img16.pixel_traits(), img_loaded.pixel_traits());
  TEST_EQUAL("12-bit image is automatically stretched to 16-bit range",
             equal_content(img_loaded, img16), true);

  // load as an 8-bit image
  config->set_value("force_byte", true);
  io.set_configuration(config);
  c = io.load("test12.tiff");
  img_loaded = c->get_image();
  TEST_EQUAL("12-bit image is compressed to 8-bits when loading with force_byte",
             img8.pixel_traits(), img_loaded.pixel_traits());
  TEST_EQUAL("12-bit image is automatically stretched to 8-bit range with force_byte",
             equal_content(img_loaded, img8), true);

  // load as an 8-bit image without stretching
  vital::image_of<uint8_t> img8t(200,300,3);
  vital::cast_image(img12, img8t);
  config->set_value("auto_stretch", false);
  io.set_configuration(config);
  c = io.load("test12.tiff");
  img_loaded = c->get_image();
  TEST_EQUAL("12-bit image is truncated to 8-bits when loading with force_byte",
             img8t.pixel_traits(), img_loaded.pixel_traits());
  TEST_EQUAL("12-bit image is truncated to 8-bit range with force_byte",
             equal_content(img_loaded, img8t), true);

  // load as an 8-bit image custom stretching
  vital::image_of<uint8_t> img8m(200,300,3);
  vital::transform_image(img12, img8m, range_to_byte<uint16_t>(100, 4000));
  config->set_value("manual_stretch", true);
  config->set_value("intensity_range", "100 4000");
  io.set_configuration(config);
  c = io.load("test12.tiff");
  img_loaded = c->get_image();
  TEST_EQUAL("12-bit image is compressed to 8-bits when loading with force_byte",
             img8m.pixel_traits(), img_loaded.pixel_traits());
  TEST_EQUAL("12-bit image is manually stretched to 8-bit range with force_byte",
             equal_content(img_loaded, img8m), true);

  if( std::remove("test12.tiff") != 0 )
  {
    TEST_ERROR("Unable to delete temporary image file.");
  }

  // test the range stretching at save time
  c = std::make_shared<simple_image_container>(img12);
  config->set_value("auto_stretch", true);
  config->set_value("manual_stretch", false);
  config->set_value("force_byte", true);
  io.set_configuration(config);
  io.save("test8.tiff", c);
  config->set_value("auto_stretch", false);
  config->set_value("force_byte", false);
  io.set_configuration(config);
  c = io.load("test8.tiff");
  img_loaded = c->get_image();
  TEST_EQUAL("12-bit image is compressed to 8-bits when saving with force_byte",
             img8.pixel_traits(), img_loaded.pixel_traits());
  TEST_EQUAL("8-bit image has correct content (stretched automatically when saved)",
             equal_content(img_loaded, img8), true);

  if( std::remove("test8.tiff") != 0 )
  {
    TEST_ERROR("Unable to delete temporary image file.");
  }
}


namespace {

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

} // end anonymous namespace


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
