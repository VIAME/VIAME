// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test Qt image class functionality
 */

#include <test_tmpfn.h>

#include <arrows/tests/test_image.h>

#include <arrows/qt/image_container.h>
#include <arrows/qt/image_io.h>

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/util/transform_image.h>

#include <vital/range/iota.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;
using namespace kwiver::arrows;

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST( image, create )
{
  plugin_manager::instance().load_all_plugins();

  std::shared_ptr< algo::image_io > img_io;
  ASSERT_NE( nullptr, img_io = algo::image_io::create( "qt" ) );

  algo::image_io* img_io_ptr = img_io.get();
  EXPECT_EQ( typeid( qt::image_io ), typeid( *img_io_ptr ) )
    << "Factory method did not construct the correct type";
}

namespace {

template < typename T, size_t Depth > void populate_qt_image( QImage& );

// ----------------------------------------------------------------------------
// Helper function to populate the (bitmask) image with a pattern
template <>
void
populate_qt_image< bool, 1 >( QImage& img )
{
  for( auto const j : range::iota( img.height() ) )
  {
    for( auto const i : range::iota( img.width() ) )
    {
      auto const y = value_at( i, j, 0 );
      img.setPixel( i, j, y >= 0.5 ? 1 : 0 );
    }
  }
}

// ----------------------------------------------------------------------------
// Helper function to populate the (grayscale) image with a pattern
template <>
void
populate_qt_image< byte, 1 >( QImage& img )
{
  for( auto const j : range::iota( img.height() ) )
  {
    for( auto const i : range::iota( img.width() ) )
    {
      auto const y = value_at( i, j, 0 );
      img.setPixelColor( i, j, QColor::fromRgbF( y, y, y ) );
    }
  }
}

// ----------------------------------------------------------------------------
// Helper function to populate the (RGB) image with a pattern
template <>
void
populate_qt_image< byte, 3 >( QImage& img )
{
  for( auto j : range::iota( img.height() ) )
  {
    for( auto i : range::iota( img.width() ) )
    {
      auto const r = value_at( i, j, 0 );
      auto const g = value_at( i, j, 1 );
      auto const b = value_at( i, j, 2 );
      img.setPixelColor( i, j, QColor::fromRgbF( r, g, b ) );
    }
  }
}

// ----------------------------------------------------------------------------
// Helper function to populate the (RGBA) image with a pattern
template <>
void
populate_qt_image< byte, 4 >( QImage& img )
{
  for( auto j : range::iota( img.height() ) )
  {
    for( auto i : range::iota( img.width() ) )
    {
      auto const r = value_at( i, j, 0 );
      auto const g = value_at( i, j, 1 );
      auto const b = value_at( i, j, 2 );
      auto const a = value_at( i, j, 3 );
      img.setPixelColor( i, j, QColor::fromRgbF( r, g, b, a ) );
    }
  }
}

// ----------------------------------------------------------------------------
QImage::Format
native_format( QImage::Format format )
{
  switch( format )
  {
    case QImage::Format_RGB888:
      return QImage::Format_RGB32;
    case QImage::Format_RGBA8888:
      return QImage::Format_ARGB32;
    default:
      return format;
  }
}

} // namespace (anonymous)

// ----------------------------------------------------------------------------
template < typename T, size_t Depth, QImage::Format QImageFormat >
struct image_type
{
  using pixel_type = T;

  static constexpr size_t depth = Depth;
  static constexpr QImage::Format qimage_format = QImageFormat;
};

// Declare storage for these members, which is required to use them in
// assertions
template < typename T, size_t Depth, QImage::Format QImageFormat >
constexpr size_t image_type< T, Depth, QImageFormat >::depth;

template < typename T, size_t Depth, QImage::Format QImageFormat >
constexpr QImage::Format image_type< T, Depth, QImageFormat >::qimage_format;

// ----------------------------------------------------------------------------
template < typename T >
class image_io : public ::testing::Test
{
};

using test_types = ::testing::Types<
  image_type< byte, 1, QImage::Format_Grayscale8 >,
  image_type< byte, 3, QImage::Format_RGB888 >,
  image_type< byte, 4, QImage::Format_RGBA8888 >,
  image_type< bool, 1, QImage::Format_Mono >
  >;

TYPED_TEST_CASE( image_io, test_types );

// ----------------------------------------------------------------------------
TYPED_TEST( image_io, type )
{
  using pix_t = typename TypeParam::pixel_type;

  image_of< pix_t > img{ 200, 300, TypeParam::depth };
  populate_vital_image< pix_t >( img );

  auto const image_ext = ( img.depth() == 4 ? ".png" : ".tiff" );
  auto const image_path = kwiver::testing::temp_file_name( "test-", image_ext );

  auto c = std::make_shared< simple_image_container >( img );
  qt::image_io io;
  io.save( image_path, c );

  auto img2 = io.load( image_path )->get_image();

  EXPECT_EQ( img.pixel_traits(), img2.pixel_traits() );
  EXPECT_EQ( img.depth(), img2.depth() );
  EXPECT_TRUE( equal_content( img, img2 ) );
  EXPECT_EQ( 0, std::remove( image_path.c_str() ) )
    << "Failed to delete temporary image file.";
}

namespace {

template < typename T, size_t Depth >
void compare_pixels( QImage const&, image const&, int, int );

template < typename T, size_t Depth >
void compare_pixels( image const&, QImage const&, size_t, size_t );

// ----------------------------------------------------------------------------
template <>
void
compare_pixels< bool, 1 >( QImage const& qi, image const& vi, int i, int j )
{
  auto const iu = static_cast< unsigned int >( i );
  auto const ju = static_cast< unsigned int >( j );
  auto const c = qi.pixelColor( i, j );
  EXPECT_EQ( !!c.value(), !!vi.at< byte >( iu, ju, 0 ) );
}

// ----------------------------------------------------------------------------
template <>
void
compare_pixels< byte, 1 >( QImage const& qi, image const& vi, int i, int j )
{
  auto const iu = static_cast< unsigned int >( i );
  auto const ju = static_cast< unsigned int >( j );
  auto const c = qi.pixelColor( i, j );
  EXPECT_EQ( c.value(), vi.at< byte >( iu, ju, 0 ) );
}

// ----------------------------------------------------------------------------
template <>
void
compare_pixels< byte, 3 >( QImage const& qi, image const& vi, int i, int j )
{
  auto const iu = static_cast< unsigned int >( i );
  auto const ju = static_cast< unsigned int >( j );
  auto const c = qi.pixelColor( i, j );
  EXPECT_EQ( c.red(),   vi.at< byte >( iu, ju, 0 ) );
  EXPECT_EQ( c.green(), vi.at< byte >( iu, ju, 1 ) );
  EXPECT_EQ( c.blue(),  vi.at< byte >( iu, ju, 2 ) );
}

// ----------------------------------------------------------------------------
template <>
void
compare_pixels< byte, 4 >( QImage const& qi, image const& vi, int i, int j )
{
  auto const iu = static_cast< unsigned int >( i );
  auto const ju = static_cast< unsigned int >( j );
  auto const c = qi.pixelColor( i, j );
  EXPECT_EQ( c.red(),   vi.at< byte >( iu, ju, 0 ) );
  EXPECT_EQ( c.green(), vi.at< byte >( iu, ju, 1 ) );
  EXPECT_EQ( c.blue(),  vi.at< byte >( iu, ju, 2 ) );
  EXPECT_EQ( c.alpha(), vi.at< byte >( iu, ju, 3 ) );
}

// ----------------------------------------------------------------------------
template <>
void
compare_pixels< bool, 1 >(
  image const& vi, QImage const& qi, size_t i, size_t j )
{
  auto const iu = static_cast< unsigned int >( i );
  auto const ju = static_cast< unsigned int >( j );
  auto const c =
    qi.pixelColor( static_cast< int >( i ), static_cast< int >( j ) );
  EXPECT_EQ( !!vi.at< byte >( iu, ju, 0 ), !!c.value() );
}

// ----------------------------------------------------------------------------
template <>
void
compare_pixels< byte, 1 >(
  image const& vi, QImage const& qi, size_t i, size_t j )
{
  auto const iu = static_cast< unsigned int >( i );
  auto const ju = static_cast< unsigned int >( j );
  auto const c =
    qi.pixelColor( static_cast< int >( i ), static_cast< int >( j ) );
  EXPECT_EQ( vi.at< byte >( iu, ju, 0 ), c.value() );
}

// ----------------------------------------------------------------------------
template <>
void
compare_pixels< byte, 3 >(
  image const& vi, QImage const& qi, size_t i, size_t j )
{
  auto const iu = static_cast< unsigned int >( i );
  auto const ju = static_cast< unsigned int >( j );
  auto const c =
    qi.pixelColor( static_cast< int >( i ), static_cast< int >( j ) );
  EXPECT_EQ( vi.at< byte >( iu, ju, 0 ), c.red() );
  EXPECT_EQ( vi.at< byte >( iu, ju, 1 ), c.green() );
  EXPECT_EQ( vi.at< byte >( iu, ju, 2 ), c.blue() );
}

// ----------------------------------------------------------------------------
template <>
void
compare_pixels< byte, 4 >(
  image const& vi, QImage const& qi, size_t i, size_t j )
{
  auto const iu = static_cast< unsigned int >( i );
  auto const ju = static_cast< unsigned int >( j );
  auto const c =
    qi.pixelColor( static_cast< int >( i ), static_cast< int >( j ) );
  EXPECT_EQ( vi.at< byte >( iu, ju, 0 ), c.red() );
  EXPECT_EQ( vi.at< byte >( iu, ju, 1 ), c.green() );
  EXPECT_EQ( vi.at< byte >( iu, ju, 2 ), c.blue() );
  EXPECT_EQ( vi.at< byte >( iu, ju, 3 ), c.alpha() );
}

// ----------------------------------------------------------------------------
template < typename T >
void
run_qt_conversion_tests( QImage const& img )
{
  using pix_t = typename T::pixel_type;

  // Convert to a vital image and verify that the properties are correct
  image const& vimg = qt::image_container::qt_to_vital( img );
  EXPECT_EQ( sizeof( pix_t ), vimg.pixel_traits().num_bytes );
  EXPECT_EQ( image_pixel_traits_of< pix_t >::static_type,
             vimg.pixel_traits().type );
  EXPECT_EQ( T::depth, vimg.depth() );
  EXPECT_EQ( static_cast< size_t >( img.height() ), vimg.height() );
  EXPECT_EQ( static_cast< size_t >( img.width() ), vimg.width() );

  // Because the input image is immutable, and the vital image needs mutable
  // data, the QImage backing the vital image will wind up being detached from
  // the input image, and so will never share memory with our input image
  // EXPECT_EQ( img.constBits(), vimg.first_pixel() )
  //   << "vital::image should share memory with QImage";

  // Don't try to compare images if they don't have the same layout!
  ASSERT_FALSE( ::testing::Test::HasNonfatalFailure() );

  [ & ]{
    for( auto const j : range::iota( img.height() ) )
    {
      for( auto const i : range::iota( img.width() ) )
      {
        compare_pixels< pix_t, T::depth >( img, vimg, i, j );
        ASSERT_FALSE( ::testing::Test::HasNonfatalFailure() )
          << "Pixels differ at " << i << ", " << j;
      }
    }
  }();

  // Convert back to QImage and test again
  auto img2 = qt::image_container::vital_to_qt( vimg );
  ASSERT_EQ( native_format( img.format() ), img2.format() );
  EXPECT_TRUE( img.convertToFormat( img2.format() ) == img2 );
}

// ----------------------------------------------------------------------------
template < typename T >
void
run_vital_conversion_tests( image_of< typename T::pixel_type > const& img )
{
  using pix_t = typename T::pixel_type;

  // Convert to a vil image and verify that the properties are correct
  auto qimg = qt::image_container::vital_to_qt( img );
  ASSERT_TRUE( !qimg.isNull() )
    << "vital::image conversion did not produce a valid QImage";

  EXPECT_EQ( native_format( T::qimage_format ), qimg.format() );
  EXPECT_EQ( static_cast< int >( img.height() ), qimg.height() );
  EXPECT_EQ( static_cast< int >( img.width() ), qimg.width() );
  EXPECT_NE( static_cast< void const* >( img.first_pixel() ),
             static_cast< void const* >( qimg.constBits() ) )
    << "QImage should NOT share memory with vital::image";

  // Don't try to compare images if they don't have the same layout!
  ASSERT_FALSE( ::testing::Test::HasNonfatalFailure() );

  [ & ]{
    for( auto const j : range::iota( img.height() ) )
    {
      for( auto const i : range::iota( img.width() ) )
      {
        compare_pixels< pix_t, T::depth >( img, qimg, i, j );
        ASSERT_FALSE( ::testing::Test::HasNonfatalFailure() )
          << "Pixels differ at " << i << ", " << j;
      }
    }
  }();

  // Convert back to vital::image and test again
  image img2 = qt::image_container::qt_to_vital( qimg );
  EXPECT_EQ( sizeof( pix_t ), img2.pixel_traits().num_bytes );
  EXPECT_EQ( image_pixel_traits_of< pix_t >::static_type,
             img2.pixel_traits().type );
  EXPECT_TRUE( equal_content( img, img2 ) );
  EXPECT_NE( img.first_pixel(), img2.first_pixel() )
    << "re-converted vital::image should NOT share memory with original";
}

} // namespace (anonymous)

// ----------------------------------------------------------------------------
template < typename T >
class image_conversion : public ::testing::Test
{
};

TYPED_TEST_CASE( image_conversion, test_types );

// ----------------------------------------------------------------------------
TYPED_TEST( image_conversion, qt_to_vital )
{
  using pix_t = typename TypeParam::pixel_type;

  // Create QImage and convert to and from vital images
  QImage img{ 100, 200, TypeParam::qimage_format };
  populate_qt_image< pix_t, TypeParam::depth >( img );
  run_qt_conversion_tests< TypeParam >( img );
}

// ----------------------------------------------------------------------------
TYPED_TEST( image_conversion, vital_to_qt )
{
  using pix_t = typename TypeParam::pixel_type;

  // Create vital images and convert to and from QImage
  // (note: different code paths are taken depending on whether the image
  // is natively created as Qt or vital, so we need to test both ways)
  image_of< pix_t > img{ 200, 300, TypeParam::depth };
  populate_vital_image< pix_t >( img );
  run_vital_conversion_tests< TypeParam >( img );
}

// ----------------------------------------------------------------------------
TEST( image_conversion, vital_to_qt_interleaved )
{
  using test_type_t = image_type< byte, 3, QImage::Format_RGB888 >;

  // Create vital images and convert to and from QImage
  // (note: different code paths are taken depending on whether the image
  // is natively created as Qt or vital, so we need to test both ways)
  image_of< byte > img{ 200, 300, 3, true };
  populate_vital_image< byte >( img );
  run_vital_conversion_tests< test_type_t >( img );
}
