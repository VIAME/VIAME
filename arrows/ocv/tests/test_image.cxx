/*ckwg +29
 * Copyright 2013-2019 by Kitware, Inc.
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

#include <test_tmpfn.h>

#include <arrows/tests/test_image.h>

#include <arrows/ocv/image_container.h>
#include <arrows/ocv/image_io.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;
using namespace kwiver::arrows;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(image, create)
{
  plugin_manager::instance().load_all_plugins();

  std::shared_ptr<algo::image_io> img_io;
  ASSERT_NE(nullptr, img_io = algo::image_io::create("ocv"));

  algo::image_io* img_io_ptr = img_io.get();
  EXPECT_EQ(typeid(ocv::image_io), typeid(*img_io_ptr))
    << "Factory method did not construct the correct type";
}

namespace {

// ----------------------------------------------------------------------------
// Helper function to populate the image with a pattern; the dynamic range is
// stretched between minv and maxv
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
        auto const val = static_cast<T>(value_at(i, j, p) * range + offset);
        img.template ptr<T>(j)[num_c * i + p] = val;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Helper function to populate the image with a pattern
template <typename T>
void
populate_ocv_image(cv::Mat& img)
{
  const T minv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::min() : T(0);
  const T maxv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::max() : T(1);
  populate_ocv_image(img, minv, maxv);
}

} // end anonymous namespace

// ----------------------------------------------------------------------------
template <typename T, int Depth>
struct image_type
{
  using pixel_type = T;
  static constexpr int depth = Depth;
};

// ----------------------------------------------------------------------------
template <typename T>
class image_io : public ::testing::Test
{
};

using io_types = ::testing::Types<
  image_type<byte, 1>,
  image_type<byte, 3>,
  image_type<byte, 4>,
  image_type<uint16_t, 1>,
  image_type<uint16_t, 3>,
  image_type<uint16_t, 4>
  >;

TYPED_TEST_CASE(image_io, io_types);

// ----------------------------------------------------------------------------
TYPED_TEST(image_io, type)
{
  using pix_t = typename TypeParam::pixel_type;
  kwiver::vital::image_of<pix_t> img( 200, 300, TypeParam::depth );
  populate_vital_image<pix_t>( img );

  auto const image_path = kwiver::testing::temp_file_name( "test-", ".png" );

  auto c = std::make_shared<simple_image_container>( img );
  ocv::image_io io;
  io.save( image_path, c );
  image_container_sptr c2 = io.load( image_path );
  kwiver::vital::image img2 = c2->get_image();
  EXPECT_EQ( img.pixel_traits(), img2.pixel_traits() );
  EXPECT_EQ( img.depth(), img2.depth() );
  EXPECT_TRUE( equal_content( img, img2 ) );
  EXPECT_EQ( 0, std::remove( image_path.c_str() ) )
    << "Failed to delete temporary image file.";
}

namespace {

// ----------------------------------------------------------------------------
template <typename T>
void
run_ocv_conversion_tests( cv::Mat const& img )
{
  // Convert to a vital image and verify that the properties are correct
  image const& vimg = ocv::image_container::ocv_to_vital( img, ocv::image_container::RGB_COLOR );
  EXPECT_EQ( sizeof(T), vimg.pixel_traits().num_bytes );
  EXPECT_EQ( image_pixel_traits_of<T>::static_type, vimg.pixel_traits().type );
  EXPECT_EQ( static_cast<size_t>( img.channels() ), vimg.depth() );
  EXPECT_EQ( static_cast<size_t>( img.rows ), vimg.height() );
  EXPECT_EQ( static_cast<size_t>( img.cols ), vimg.width() );
  EXPECT_EQ( img.data, vimg.first_pixel() )
    << "vital::image should share memory with cv::Mat";

  // Don't try to compare images if they don't have the same layout!
  ASSERT_FALSE( ::testing::Test::HasNonfatalFailure() );

  [&]{
    int const num_c = img.channels();
    for( int c = 0; c < num_c; ++c )
    {
      for( int j = 0; j < img.rows; ++j )
      {
        for( int i = 0; i < img.cols; ++i )
        {
#if __GNUC__ == 4 && __GNUC_MINOR__ == 9
          // GCC 4.9.x has a really screwy bug where using sizeof inside of a
          // lambda inside of a template function (e.g. here) on an expression
          // which contains an explicit template parameter (whether or not
          // said parameter depends on the outer function's template parameter)
          // causes an error. This bug gets tripped by GTEST_IS_NULL_LITERAL_.
          // Work around the issue by expanding out ASSERT_EQ and substituting
          // false for the GTEST_IS_NULL_LITERAL_ invocation.
          ASSERT_PRED_FORMAT2(
            ::testing::internal::EqHelper<false>::Compare,
            img.ptr<T>( j )[ num_c * i + c ], vimg.at<T>( i, j, c ) );
#else
          ASSERT_EQ( img.ptr<T>( j )[ num_c * i + c ], vimg.at<T>( i, j, c ) );
#endif
        }
      }
    }
  }();

  // Convert back to cv::Mat and test again
  cv::Mat img2 = ocv::image_container::vital_to_ocv( vimg, ocv::image_container::RGB_COLOR );
  ASSERT_NE( nullptr, img2.data )
    << "OpenCV re-conversion did not produce a valid cv::Mat";

  EXPECT_EQ( img.type(), img2.type() );
  ASSERT_EQ( img.channels(), img2.channels() );

  std::vector<cv::Mat> channels1( img.channels() );
  std::vector<cv::Mat> channels2( img2.channels() );
  cv::split( img, channels1 );
  cv::split( img2, channels2 );

  for ( unsigned c = 0; c < channels1.size(); ++c )
  {
    SCOPED_TRACE( "In channel " + std::to_string(c) );
    EXPECT_EQ( 0, cv::countNonZero( channels1[c] != channels2[c] ) );
  }

  EXPECT_EQ( img2.data, img.data )
    << "re-converted cv::Mat should share memory with original";
}

// ----------------------------------------------------------------------------
template <typename T>
void
run_vital_conversion_tests( kwiver::vital::image_of<T> const& img,
                            bool requires_copy = false )
{
  // convert to a cv::Mat and verify that the properties are correct
  cv::Mat ocv_img =  ocv::image_container::vital_to_ocv(img, ocv::image_container::RGB_COLOR);
  ASSERT_NE( nullptr, ocv_img.data )
    << "Vital image conversion did not produce a valid cv::Mat";

  EXPECT_EQ( cv::Mat_<T>{}.type() & 0x7, ocv_img.type() & 0x7 );
  EXPECT_EQ( img.depth(), ocv_img.channels() );
  EXPECT_EQ( img.height(), ocv_img.rows );
  EXPECT_EQ( img.width(), ocv_img.cols );
  if ( !requires_copy )
  {
    EXPECT_EQ( img.first_pixel(), reinterpret_cast<T*>( ocv_img.data ) )
      << "cv::Mat should share memory with vital::image";
  }

  [&]{
    int const num_c = ocv_img.channels();
    for( int c = 0; c < num_c; ++c )
    {
      for( int j = 0; j < ocv_img.rows; ++j )
      {
        for( int i = 0; i < ocv_img.cols; ++i )
        {
          ASSERT_EQ( img( i, j, c ), ocv_img.ptr<T>( j )[ num_c * i + c ] )
            << "Pixels differ at " << i << ", " << j << ", " << c;
        }
      }
    }
  }();

  // Convert back to vital::image and test again
  image img2 = ocv::image_container::ocv_to_vital( ocv_img, ocv::image_container::RGB_COLOR );
  EXPECT_EQ( sizeof(T), img2.pixel_traits().num_bytes );
  EXPECT_EQ( image_pixel_traits_of<T>::static_type, img2.pixel_traits().type );
  EXPECT_TRUE( equal_content( img, img2 ) );
  EXPECT_EQ( reinterpret_cast<T*>( ocv_img.data ), img2.first_pixel() );
  if ( !requires_copy )
  {
    EXPECT_EQ( img.first_pixel(), img2.first_pixel() )
      << "re-converted vital::image should share memory with original";
  }
}

} // end anonymous namespace

// ----------------------------------------------------------------------------
template <typename T>
class image_conversion : public ::testing::Test
{
};

using conversion_types =
  ::testing::Types<uint8_t, int8_t, uint16_t, int16_t, int32_t, float, double>;
TYPED_TEST_CASE(image_conversion, conversion_types);

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, ocv_to_vital_single_channel)
{
  // Create single-channel cv::Mat and convert to and from vital images
  cv::Mat_<TypeParam> img{ cv::Size{ 100, 200 } };
  populate_ocv_image<TypeParam>( img );
  run_ocv_conversion_tests<TypeParam>( img );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, ocv_to_vital_multi_channel)
{
  // Create multi-channel cv::Mat and convert to and from vital images
  cv::Mat_<cv::Vec<TypeParam, 3>> img{ cv::Size{ 100, 200 } };
  populate_ocv_image<TypeParam>( img );
  run_ocv_conversion_tests<TypeParam>( img );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, ocv_to_vital_cropped)
{
  // Create cropped cv::Mat and convert to and from vital images
  cv::Mat_<cv::Vec<TypeParam, 3>> img{ cv::Size{ 200, 300 } };
  populate_ocv_image<TypeParam>( img );
  cv::Rect window( cv::Point{ 40, 50 }, cv::Point{ 140, 250 } );
  cv::Mat_<cv::Vec<TypeParam, 3>> img_crop{ img, window };
  run_ocv_conversion_tests<TypeParam>( img );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, vital_to_ocv_single_channel)
{
  // Create vital images and convert to and from cv::Mat
  // (note: different code paths are taken depending on whether the image
  // is natively created as OpenCV or vital, so we need to test both ways)
  kwiver::vital::image_of<TypeParam> img{ 200, 300, 1 };
  populate_vital_image<TypeParam>( img );
  run_vital_conversion_tests( img );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, vital_to_ocv_multi_channel)
{
  // Create vital images and convert to and from cv::Mat
  // (note: different code paths are taken depending on whether the image
  // is natively created as OpenCV or vital, so we need to test both ways)
  kwiver::vital::image_of<TypeParam> img{ 200, 300, 3 };
  populate_vital_image<TypeParam>( img );
  run_vital_conversion_tests( img, true );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, vital_to_ocv_interleaved)
{
  // Create vital images and convert to and from cv::Mat
  // (note: different code paths are taken depending on whether the image
  // is natively created as OpenCV or vital, so we need to test both ways)
  kwiver::vital::image_of<TypeParam> img{ 200, 300, 3, true };
  populate_vital_image<TypeParam>( img );
  run_vital_conversion_tests( img );
}

// ----------------------------------------------------------------------------
template <typename T>
class image_bgr_conversion : public ::testing::Test
{
};

using image_bgr_types = ::testing::Types<uint8_t, uint16_t, float>;
TYPED_TEST_CASE(image_bgr_conversion, image_bgr_types);

// ----------------------------------------------------------------------------
TYPED_TEST(image_bgr_conversion, bgr_to_rgb)
{
  kwiver::vital::image_of<TypeParam> img{ 200, 300, 3 };
  populate_vital_image<TypeParam>( img );
  cv::Mat ocv_img =  ocv::image_container::vital_to_ocv(img, ocv::image_container::BGR_COLOR);
  {
    int const num_c = ocv_img.channels();
    for( int j = 0; j < ocv_img.rows; ++j )
    {
      for( int i = 0; i < ocv_img.cols; ++i )
      {
        for( int c = 0; c < 3; ++c )
        {
          ASSERT_EQ( img( i, j, c ), ocv_img.ptr<TypeParam>( j )[ num_c * i + (2-c) ] )
              << "Pixels differ at " << i << ", " << j << ", (" << c << "," << (2-c) << ")";
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_bgr_conversion, bgra_to_rgba)
{
  kwiver::vital::image_of<TypeParam> img{ 200, 300, 4 };
  populate_vital_image<TypeParam>( img );
  cv::Mat ocv_img =  ocv::image_container::vital_to_ocv(img, ocv::image_container::BGR_COLOR);
  {
    int const num_c = ocv_img.channels();
    for( int j = 0; j < ocv_img.rows; ++j )
    {
      for( int i = 0; i < ocv_img.cols; ++i )
      {
        for( int c = 0; c < 3; ++c )
        {
          ASSERT_EQ( img( i, j, c ), ocv_img.ptr<TypeParam>( j )[ num_c * i + (2-c) ] )
              << "Pixels differ at " << i << ", " << j << ", (" << c << "," << (2-c) << ")";
        }
        ASSERT_EQ( img( i, j, 3 ), ocv_img.ptr<TypeParam>( j )[ num_c * i + (3) ] )
              << "Pixels differ at " << i << ", " << j << ", (3,3)";
      }
    }
  }
}

// ----------------------------------------------------------------------------
template <typename T>
class get_image : public ::testing::Test
{
};

using get_image_types = ::testing::Types<
  image_type<byte, 1>,
  image_type<byte, 3>,
  image_type<uint16_t, 1>,
  image_type<uint16_t, 3>,
  image_type<float, 1>,
  image_type<float, 3>,
  image_type<double, 1>,
  image_type<double, 3>
  >;

TYPED_TEST_CASE(get_image, get_image_types);

// ----------------------------------------------------------------------------
TYPED_TEST(get_image, crop)
{
  using pix_t = typename TypeParam::pixel_type;
  cv::Mat_< cv::Vec< pix_t, TypeParam::depth > > img{
    cv::Size{ static_cast<int>( full_width ),
              static_cast<int>( full_height ) } };
  populate_ocv_image<pix_t>( img );

  image_container_sptr img_cont =
    std::make_shared<ocv::image_container>( img, ocv::image_container::RGB_COLOR );

  test_get_image_crop<pix_t>( img_cont );
}

// ----------------------------------------------------------------------------
TEST(image, bgr_to_rgb_bad_types)
{
  EXPECT_THROW( ocv::image_container::vital_to_ocv(
                  image_of<int8_t>{ 200, 300, 3 },
                  ocv::image_container::BGR_COLOR ),
                image_type_mismatch_exception );
  EXPECT_THROW( ocv::image_container::vital_to_ocv(
                  image_of<int16_t>{ 200, 300, 3 },
                  ocv::image_container::BGR_COLOR ),
                image_type_mismatch_exception );
  EXPECT_THROW( ocv::image_container::vital_to_ocv(
                  image_of<int32_t>{ 200, 300, 3 },
                  ocv::image_container::BGR_COLOR ),
                image_type_mismatch_exception );
  EXPECT_THROW( ocv::image_container::vital_to_ocv(
                  image_of<double>{ 200, 300, 3 },
                  ocv::image_container::BGR_COLOR ),
                image_type_mismatch_exception );
}

// ----------------------------------------------------------------------------
TEST(image, bad_conversions)
{
  // Some types not supported by OpenCV and should throw an exception
  EXPECT_THROW( ocv::image_container::vital_to_ocv(
                  image_of<uint32_t>( 200, 300 ), ocv::image_container::RGB_COLOR ),
                image_type_mismatch_exception );
  EXPECT_THROW( ocv::image_container::vital_to_ocv(
                  image_of<int64_t>( 200, 300 ), ocv::image_container::RGB_COLOR ),
                image_type_mismatch_exception );
  EXPECT_THROW( ocv::image_container::vital_to_ocv(
                  image_of<uint64_t>( 200, 300 ), ocv::image_container::RGB_COLOR ),
                image_type_mismatch_exception );
}
