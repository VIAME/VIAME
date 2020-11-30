// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test VXL image class functionality
 */

#include <test_tmpfn.h>

#include <arrows/tests/test_image.h>

#include <arrows/vxl/image_container.h>
#include <arrows/vxl/image_io.h>

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/util/transform_image.h>

#include <vil/vil_crop.h>

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
  ASSERT_NE(nullptr, img_io = algo::image_io::create("vxl"));

  algo::image_io* img_io_ptr = img_io.get();
  EXPECT_EQ(typeid(vxl::image_io), typeid(*img_io_ptr))
    << "Factory method did not construct the correct type";
}

namespace {

// ----------------------------------------------------------------------------
// Helper functor for use in transform_image
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

// ----------------------------------------------------------------------------
// Helper functor for use in transform_image; mimics the
// vil_convert_stretch_range_limited operation
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

// ----------------------------------------------------------------------------
// Helper function to populate the image with a pattern; the dynamic range is
// stretched between minv and maxv
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
        img(i,j,p) = static_cast<T>(value_at(i, j, p) * range + offset);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Helper function to populate the image with a pattern
template <typename T>
void
populate_vil_image(vil_image_view<T>& img)
{
  const T minv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::min() : T(0);
  const T maxv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::max() : T(1);
  populate_vil_image(img, minv, maxv);
}

} // end anonymous namespace

// ----------------------------------------------------------------------------
template <typename T, int Depth>
struct image_type
{
  using pixel_type = T;
  static constexpr unsigned int depth = Depth;
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
  image_type<uint16_t, 4>,
  image_type<float, 1>,
  image_type<float, 3>,
  image_type<double, 1>, // current VXL supports only one channel double TIFFs
  image_type<bool, 1>   // current VXL supports only one channel boolean TIFFs
  >;

TYPED_TEST_CASE(image_io, io_types);

// ----------------------------------------------------------------------------
TYPED_TEST(image_io, type)
{
  using pix_t = typename TypeParam::pixel_type;
  kwiver::vital::image_of<pix_t> img( 200, 300, TypeParam::depth );
  populate_vital_image<pix_t>( img );

  auto const image_ext = ( img.depth() == 4 ? ".png" : ".tiff" );
  auto const image_path = kwiver::testing::temp_file_name( "test-", image_ext );

  auto c = std::make_shared<simple_image_container>( img );
  vxl::image_io io;
  io.save( image_path, c );
  image_container_sptr c2 = io.load( image_path );
  kwiver::vital::image img2 = c2->get_image();
  EXPECT_EQ( img.pixel_traits(), img2.pixel_traits() );
  EXPECT_EQ( img.depth(), img2.depth() );
  EXPECT_TRUE( equal_content( img, img2 ) );
  EXPECT_EQ( 0, std::remove( image_path.c_str() ) )
    << "Failed to delete temporary image file.";
}

// ----------------------------------------------------------------------------
class image_io_stretch : public ::testing::Test
{
public:
  image_io_stretch() :
    img12{ 200, 300, 3 },
    img12_path{ kwiver::testing::temp_file_name( "test-", ".tiff" ) }
    {}

  void SetUp();
  void TearDown();

  kwiver::arrows::vxl::image_io io;
  kwiver::vital::image_of<uint16_t> img12;
  std::string const img12_path;
};

// ----------------------------------------------------------------------------
void image_io_stretch::SetUp()
{
  // Create an image with 12-bit data in a 16-bit image
  populate_vital_image<uint16_t>( img12, 0, 4095 );
  auto c = std::make_shared<simple_image_container>( img12 );

  // Save 12-bit image
  io.save( img12_path, c );
  std::cout << "wrote 12-bit test image" << std::endl;
}

// ----------------------------------------------------------------------------
void image_io_stretch::TearDown()
{
  EXPECT_EQ( 0, std::remove( img12_path.c_str() ) )
    << "Failed to delete temporary image file.";
}

// ----------------------------------------------------------------------------
TEST_F(image_io_stretch, auto_stretch)
{
  using namespace kwiver;

  vital::image_of<uint16_t> img16{ 200, 300, 3 };
  img16.copy_from( img12 );
  auto const scale = ( 65536.0 - 1e-6 ) / 4095.0;
  vital::transform_image( img16, scale_offset<uint16_t>( scale, 0 ) );

  // Load as a 16-bit image
  vital::config_block_sptr config = vital::config_block::empty_config();
  config->set_value( "auto_stretch", true );
  io.set_configuration( config );

  auto const& c = io.load( img12_path );
  auto const& img_loaded = c->get_image();

  EXPECT_EQ( img16.pixel_traits(), img_loaded.pixel_traits() );
  EXPECT_TRUE( equal_content( img16, img_loaded ) );
}

// ----------------------------------------------------------------------------
TEST_F(image_io_stretch, as_byte_auto_stretch)
{
  using namespace kwiver;

  vital::image_of<uint8_t> img8{ 200, 300,3 };
  vital::image_of<uint16_t> img16{ 200, 300, 3 };
  img16.copy_from( img12 );
  auto const scale = 255.0 / 4095.0;
  vital::transform_image( img16, scale_offset<uint16_t>( scale, 0 ) );
  cast_image( img16, img8 );

  // Load as an 8-bit image
  vital::config_block_sptr config = vital::config_block::empty_config();
  config->set_value( "auto_stretch", true );
  config->set_value( "force_byte", true );
  io.set_configuration( config );

  auto const& c = io.load( img12_path );
  auto const& img_loaded = c->get_image();

  EXPECT_EQ( img8.pixel_traits(), img_loaded.pixel_traits() );
  EXPECT_TRUE( equal_content( img8, img_loaded ) );
}

// ----------------------------------------------------------------------------
TEST_F(image_io_stretch, as_byte_no_stretch)
{
  using namespace kwiver;

  // load as an 8-bit image without stretching
  vital::image_of<uint8_t> img8t{ 200, 300, 3 };
  vital::cast_image( img12, img8t );

  vital::config_block_sptr config = vital::config_block::empty_config();
  config->set_value( "auto_stretch", false );
  config->set_value( "force_byte", true );
  io.set_configuration( config );

  auto const& c = io.load( img12_path );
  auto const& img_loaded = c->get_image();

  EXPECT_EQ( img8t.pixel_traits(), img_loaded.pixel_traits() );
  EXPECT_TRUE( equal_content( img8t, img_loaded ) );
}

// ----------------------------------------------------------------------------
TEST_F(image_io_stretch, as_byte_manual_stretch)
{
  using namespace kwiver;

  // load as an 8-bit image custom stretching
  vital::image_of<uint8_t> img8m{ 200, 300, 3 };
  vital::transform_image( img12, img8m, range_to_byte<uint16_t>( 100, 4000 ) );

  vital::config_block_sptr config = vital::config_block::empty_config();
  config->set_value( "auto_stretch", false );
  config->set_value( "force_byte", true );
  config->set_value( "manual_stretch", true );
  config->set_value( "intensity_range", "100 4000" );
  io.set_configuration( config );

  auto const& c = io.load( img12_path );
  auto const& img_loaded = c->get_image();

  EXPECT_EQ( img8m.pixel_traits(), img_loaded.pixel_traits() );
  EXPECT_TRUE( equal_content( img8m, img_loaded ) );
}

// ----------------------------------------------------------------------------
TEST_F(image_io_stretch, stretch_on_save)
{
  using namespace kwiver;

  vital::image_of<uint8_t> img8{ 200, 300,3 };
  vital::image_of<uint16_t> img16{ 200, 300, 3 };
  img16.copy_from( img12 );
  auto const scale = 255.0 / 4095.0;
  vital::transform_image( img16, scale_offset<uint16_t>( scale, 0 ) );
  cast_image( img16, img8 );

  // Test the range stretching at save time
  auto cs = std::make_shared<simple_image_container>( img12 );

  vital::config_block_sptr config = vital::config_block::empty_config();
  config->set_value( "auto_stretch", true );
  config->set_value( "force_byte", true );
  io.set_configuration( config );

  auto const& image_path = kwiver::testing::temp_file_name( "test-", ".tiff" );
  io.save( image_path, cs );

  config->set_value( "auto_stretch", false );
  config->set_value( "force_byte", false );
  io.set_configuration( config );

  auto const& cl = io.load( image_path );
  auto const& img_loaded = cl->get_image();

  EXPECT_EQ( img8.pixel_traits(), img_loaded.pixel_traits() );
  EXPECT_TRUE( equal_content( img8, img_loaded ) );

  EXPECT_EQ( 0, std::remove( image_path.c_str() ) )
    << "Failed to delete temporary image file.";
}

namespace {

// ----------------------------------------------------------------------------
template <typename T>
void
run_vil_conversion_tests( vil_image_view<T> const& img )
{
  // Convert to a vital image and verify that the properties are correct
  image const& vimg = vxl::image_container::vxl_to_vital( img );
  EXPECT_EQ( sizeof(T), vimg.pixel_traits().num_bytes );
  EXPECT_EQ( image_pixel_traits_of<T>::static_type, vimg.pixel_traits().type );
  EXPECT_EQ( img.nplanes(), vimg.depth() );
  EXPECT_EQ( img.nj(), vimg.height() );
  EXPECT_EQ( img.ni(), vimg.width() );
  EXPECT_EQ( img.top_left_ptr(), vimg.first_pixel() )
    << "vital::image should share memory with vil_image_view";

  // Don't try to compare images if they don't have the same layout!
  ASSERT_FALSE( ::testing::Test::HasNonfatalFailure() );

  [&]{
    for( unsigned int p = 0; p < img.nplanes(); ++p )
    {
      for( unsigned int j = 0; j < img.nj(); ++j )
      {
        for( unsigned int i = 0; i < img.ni(); ++i )
        {
          ASSERT_EQ( img( i, j, p ), vimg.at<T>( i, j, p ) );
        }
      }
    }
  }();

  // Convert back to VXL and test again
  vil_image_view<T> img2 = vxl::image_container::vital_to_vxl( vimg );
  ASSERT_TRUE( !!img2 )
    << "vil_image_view re-conversion did not produce a valid vil_image_view";

  EXPECT_EQ( img2.pixel_format(), img.pixel_format() );
  EXPECT_TRUE( vil_image_view_deep_equality( img, img2 ) );
  EXPECT_EQ( img2.top_left_ptr(), img.top_left_ptr() )
    << "re-converted vil_image_view should share memory with original";
}

// ----------------------------------------------------------------------------
template <typename T>
void
run_vital_conversion_tests( kwiver::vital::image_of<T> const& img )
{
  // Convert to a vil image and verify that the properties are correct
  vil_image_view<T> vimg = vxl::image_container::vital_to_vxl( img );
  ASSERT_TRUE( !!vimg )
    << "vital::image conversion did not produce a valid vil_image_view";

  EXPECT_EQ( vil_pixel_format_of( T() ), vimg.pixel_format() );
  EXPECT_EQ( img.depth(), vimg.nplanes() );
  EXPECT_EQ( img.height(), vimg.nj() );
  EXPECT_EQ( img.width(), vimg.ni() );
  EXPECT_EQ( img.first_pixel(), vimg.top_left_ptr() )
    << "vil_image_view should share memory with vital::image";

  // Don't try to compare images if they don't have the same layout!
  ASSERT_FALSE( ::testing::Test::HasNonfatalFailure() );

  [&]{
    for( unsigned int p = 0; p < vimg.nplanes(); ++p )
    {
      for( unsigned int j = 0; j < vimg.nj(); ++j )
      {
        for( unsigned int i = 0; i < vimg.ni(); ++i )
        {
          ASSERT_EQ( img( i, j, p ), vimg( i, j, p ) )
            << "Pixels differ at " << i << ", " << j << ", " << p;
        }
      }
    }
  }();

  // Convert back to vital::image and test again
  image img2 = vxl::image_container::vxl_to_vital( vimg );
  EXPECT_EQ( sizeof(T), img2.pixel_traits().num_bytes );
  EXPECT_EQ( image_pixel_traits_of<T>::static_type, img2.pixel_traits().type );
  EXPECT_TRUE( equal_content( img, img2 ) );
  EXPECT_EQ( img.first_pixel(), img2.first_pixel() )
    << "re-converted vital::image should share memory with original";
}

} // end anonymous namespace

// ----------------------------------------------------------------------------
template <typename T>
class image_conversion : public ::testing::Test
{
};

using conversion_types =
  ::testing::Types<vxl_byte, vxl_sbyte, vxl_uint_16, vxl_int_16, vxl_uint_32,
                   vxl_uint_64, vxl_int_64, float, double, bool>;
TYPED_TEST_CASE(image_conversion, conversion_types);

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, vxl_to_vital_single_channel)
{
  // Create vil_image_view and convert to and from vital images
  vil_image_view<TypeParam> img{ 100, 200, 1 };
  populate_vil_image( img );
  run_vil_conversion_tests( img );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, vxl_to_vital_multi_channel)
{
  // Create vil_image_view and convert to and from vital images
  vil_image_view<TypeParam> img{ 100, 200, 3 };
  populate_vil_image( img );
  run_vil_conversion_tests( img );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, vxl_to_vital_interleaved)
{
  // Create interleaved vil_image_view and convert to and from vital images
  vil_image_view<TypeParam> img{ 100, 200, 3, true };
  populate_vil_image( img );
  run_vil_conversion_tests( img );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, vxl_to_vital_cropped)
{
  // Create cropped vil_image_view and convert to and from vital images
  vil_image_view<TypeParam> img{ 200, 300, 3 };
  populate_vil_image(img);
  vil_image_view<TypeParam> img_crop = vil_crop( img, 50, 100, 40, 200 );
  run_vil_conversion_tests( img_crop );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, vital_to_vxl_single_channel)
{
  // Create vital images and convert to and from vil_image_view
  // (note: different code paths are taken depending on whether the image
  // is natively created as vil or vital, so we need to test both ways)
  kwiver::vital::image_of<TypeParam> img{ 200, 300, 1 };
  populate_vital_image<TypeParam>( img );
  run_vital_conversion_tests( img );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, vital_to_vxl_multi_channel)
{
  // Create vital images and convert to and from vil_image_view
  // (note: different code paths are taken depending on whether the image
  // is natively created as vil or vital, so we need to test both ways)
  kwiver::vital::image_of<TypeParam> img{ 200, 300, 3 };
  populate_vital_image<TypeParam>( img );
  run_vital_conversion_tests( img );
}

// ----------------------------------------------------------------------------
TYPED_TEST(image_conversion, vital_to_vxl_interleaved)
{
  // Create vital images and convert to and from vil_image_view
  // (note: different code paths are taken depending on whether the image
  // is natively created as vil or vital, so we need to test both ways)
  kwiver::vital::image_of<TypeParam> img{ 200, 300, 3, true };
  populate_vital_image<TypeParam>( img );
  run_vital_conversion_tests( img );
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
  vil_image_view<pix_t> img{ full_width, full_height, TypeParam::depth };
  populate_vil_image( img );

  image_container_sptr img_cont = std::make_shared<vxl::image_container>(img);

  test_get_image_crop<pix_t>( img_cont );
}
