/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// Drawing, template matching and concatenation, against OpenCV.
///
/// The drawing cases are the geometry only. `font_5x7.h` is a bitmap font
/// and OpenCV's is Hershey's, so the glyph shapes differ by construction and
/// there is nothing to compare -- the text tests below check what a font has
/// to be true of instead: that it draws inside its cell, advances by the
/// width it reports, and leaves the image alone where there is no glyph.

#include <image_ops/draw.h>
#include <image_ops/layout.h>
#include <image_ops/match.h>

#include <viame/core_types/image.h>

#include "golden_image.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace io = viame::image_ops;
namespace kv = kwiver::vital;

namespace golden_image = viame::testing::golden_image;

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

namespace {

/// A black canvas the size of the recorded window.
kv::image_of< uint8_t >
canvas( std::string const& name )
{
  auto image = golden_image::input( name );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      for( size_t p = 0; p < image.depth(); ++p )
      {
        image( i, j, p ) = 0;
      }
    }
  }

  return image;
}

/// The recorded template-match surface, scaled the way the recorder scales
/// it: a score of s is round((s + 1) * 10000).
kv::image_of< int32_t >
as_recorded_scores( kv::image_of< float > const& surface )
{
  kv::image_of< int32_t > out( surface.width(), surface.height(), 1 );

  for( size_t j = 0; j < surface.height(); ++j )
  {
    for( size_t i = 0; i < surface.width(); ++i )
    {
      out( i, j, 0 ) = static_cast< int32_t >(
        std::lround( ( static_cast< double >( surface( i, j, 0 ) ) + 1.0 ) *
                     10000.0 ) );
    }
  }

  return out;
}

/// The sub-image the recorder used as the pattern.
template < typename T >
kv::image_of< T >
sub_image( kv::image_of< T > const& image, size_t left, size_t top,
           size_t width, size_t height )
{
  kv::image_of< T > out( width, height, image.depth() );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      for( size_t p = 0; p < image.depth(); ++p )
      {
        out( i, j, p ) = image( left + i, top + j, p );
      }
    }
  }

  return out;
}

io::colour const white{ 255.0 };

} // namespace

// ----------------------------------------------------------------------------
TEST ( draw, template_matching_matches_opencv )
{
  for( auto const* name : { "match_ncc", "match_ncc_rgb" } )
  {
    auto const image = golden_image::input( name );
    auto const pattern = sub_image( image, 9, 6, 10, 8 );

    golden_image::expect_matches(
      name, as_recorded_scores( io::match_template_ncc( image, pattern ) ) );
  }
}

// ----------------------------------------------------------------------------
TEST ( draw, concatenation_matches_opencv )
{
  auto const image = golden_image::input( "hconcat" );

  // A fresh image, not a copy: `vital::image` copies share their memory, so
  // writing into a copy while reading the original reads what was just
  // written
  kv::image_of< uint8_t > mirrored_i( image.width(), image.height(), 1 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      mirrored_i( i, j, 0 ) = image( image.width() - 1 - i, j, 0 );
    }
  }

  golden_image::expect_matches(
    "hconcat", io::horizontal_concat< uint8_t >( { image, mirrored_i } ) );

  kv::image_of< uint8_t > mirrored_j( image.width(), image.height(), 1 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      mirrored_j( i, j, 0 ) = image( i, image.height() - 1 - j, 0 );
    }
  }

  golden_image::expect_matches(
    "vconcat", io::vertical_concat< uint8_t >( { image, mirrored_j } ) );
}

// ----------------------------------------------------------------------------
TEST ( draw, rectangles_match_opencv )
{
  // cv::rectangle takes inclusive corners; `rect` is half open
  io::rect const bounds{ 4, 3, 21, 16 };

  auto outlined = canvas( "draw_rect" );
  io::draw_rect( outlined, bounds, io::colour{ 200.0 }, 1 );
  golden_image::expect_matches( "draw_rect", outlined );

  auto filled = canvas( "draw_rect_filled" );
  io::draw_rect( filled, bounds, io::colour{ 200.0 }, -1 );
  golden_image::expect_matches( "draw_rect_filled", filled );
}

// ----------------------------------------------------------------------------
TEST ( draw, lines_match_opencv )
{
  auto image = canvas( "draw_lines" );

  io::draw_line( image, 1, 1, 30, 22, io::colour{ 180.0 } );
  io::draw_line( image, 30, 2, 2, 20, io::colour{ 180.0 } );
  io::draw_line( image, 0, 12, 31, 12, io::colour{ 180.0 } );
  io::draw_line( image, 16, 0, 16, 23, io::colour{ 180.0 } );

  golden_image::expect_matches( "draw_lines", image );
}

// ----------------------------------------------------------------------------
TEST ( draw, circles_match_opencv )
{
  auto outlined = canvas( "draw_circle" );
  io::draw_circle( outlined, 16, 12, 9, io::colour{ 220.0 }, 1 );
  golden_image::expect_matches( "draw_circle", outlined );

  auto filled = canvas( "draw_circle_filled" );
  io::draw_circle( filled, 16, 12, 9, io::colour{ 220.0 }, -1 );
  golden_image::expect_matches( "draw_circle_filled", filled );
}

// ----------------------------------------------------------------------------
TEST ( draw, filled_polygons_match_opencv )
{
  auto image = canvas( "draw_polygon" );

  std::vector< io::point > const shape{
    { 5, 3 }, { 28, 8 }, { 20, 21 }, { 8, 17 } };

  io::fill_polygon( image, shape, io::colour{ 150.0 } );

  golden_image::expect_matches( "draw_polygon", image );
}

// ----------------------------------------------------------------------------
/// Values a recording cannot give, starting with the ones about text.
TEST ( draw, text_stays_inside_the_size_it_reports )
{
  kv::image_of< uint8_t > image( 80, 20, 1 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      image( i, j, 0 ) = 0;
    }
  }

  std::string const text = "Fish 0.93";
  auto const size = io::text_size( text );

  io::draw_text( image, text, 3, 4, white );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      if( image( i, j, 0 ) == 0 )
      {
        continue;
      }

      EXPECT_GE( static_cast< long >( i ), 3L );
      EXPECT_LT( static_cast< long >( i ), 3L + size.right );
      EXPECT_GE( static_cast< long >( j ), 4L );
      EXPECT_LT( static_cast< long >( j ), 4L + size.bottom );
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( draw, text_scales_by_whole_multiples )
{
  auto const one = io::text_size( "AB", 1 );
  auto const three = io::text_size( "AB", 3 );

  EXPECT_EQ( one.right * 3, three.right );
  EXPECT_EQ( one.bottom * 3, three.bottom );

  // The empty string has no size, and a single glyph has no trailing gap
  EXPECT_EQ( 0, io::text_size( "", 1 ).right );
  EXPECT_EQ( io::font_width, io::text_size( "A", 1 ).right );
}

// ----------------------------------------------------------------------------
/// A space draws nothing, and every printable character draws something.
TEST ( draw, every_printable_character_has_a_glyph )
{
  bool space_is_blank = true;

  for( int i = 0; i < io::font_width; ++i )
  {
    for( int j = 0; j < io::font_height; ++j )
    {
      if( io::font_pixel( ' ', i, j ) )
      {
        space_is_blank = false;
      }
    }
  }

  EXPECT_TRUE( space_is_blank );

  for( char ch = io::font_first; ch <= io::font_last; ++ch )
  {
    if( ch == ' ' )
    {
      continue;
    }

    bool any = false;

    for( int i = 0; i < io::font_width && !any; ++i )
    {
      for( int j = 0; j < io::font_height && !any; ++j )
      {
        any = io::font_pixel( ch, i, j );
      }
    }

    EXPECT_TRUE( any ) << "'" << ch << "' has an empty glyph";
  }

  // and anything outside the range draws nothing rather than reading past
  EXPECT_FALSE( io::font_pixel( '\n', 0, 0 ) );
  EXPECT_FALSE( io::font_pixel( static_cast< char >( 200 ), 0, 0 ) );
}

// ----------------------------------------------------------------------------
/// Drawing outside the image is a no-op, not a crash.
TEST ( draw, drawing_off_the_edge_is_harmless )
{
  kv::image_of< uint8_t > image( 4, 4, 1 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      image( i, j, 0 ) = 7;
    }
  }

  io::draw_line( image, -50, -50, -10, -10, white );
  io::draw_rect( image, io::rect{ 100, 100, 120, 120 }, white );
  io::draw_circle( image, -20, -20, 5, white );
  io::draw_text( image, "off", -100, -100, white );
  io::fill_polygon( image,
                    { { -10, -10 }, { -5, -10 }, { -5, -5 } }, white );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      EXPECT_EQ( 7, image( i, j, 0 ) );
    }
  }
}

// ----------------------------------------------------------------------------
/// One colour value is broadcast to every plane; several are taken per plane.
TEST ( draw, a_colour_is_per_plane_or_broadcast )
{
  kv::image_of< uint8_t > image( 3, 1, 3 );

  io::draw_point( image, 0, 0, io::colour{ 10.0 } );
  io::draw_point( image, 1, 0, io::colour{ 10.0, 20.0, 30.0 } );
  io::draw_point( image, 2, 0, io::colour{ 40.0, 50.0 } );

  EXPECT_EQ( 10, image( 0, 0, 0 ) );
  EXPECT_EQ( 10, image( 0, 0, 1 ) );
  EXPECT_EQ( 10, image( 0, 0, 2 ) );

  EXPECT_EQ( 10, image( 1, 0, 0 ) );
  EXPECT_EQ( 20, image( 1, 0, 1 ) );
  EXPECT_EQ( 30, image( 1, 0, 2 ) );

  // Short of a plane, the last value carries
  EXPECT_EQ( 40, image( 2, 0, 0 ) );
  EXPECT_EQ( 50, image( 2, 0, 1 ) );
  EXPECT_EQ( 50, image( 2, 0, 2 ) );
}

// ----------------------------------------------------------------------------
/// A pattern taken from the image scores 1 where it came from.
TEST ( draw, a_template_matches_itself_exactly )
{
  kv::image_of< uint8_t > image( 12, 9, 1 );
  uint8_t value = 3;

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      image( i, j, 0 ) = static_cast< uint8_t >( value = value * 7 + 5 );
    }
  }

  auto const pattern = sub_image( image, 4, 3, 5, 4 );
  auto const surface = io::match_template_ncc( image, pattern );

  ASSERT_EQ( 8u, surface.width() );
  ASSERT_EQ( 6u, surface.height() );

  EXPECT_NEAR( 1.0f, surface( 4, 3, 0 ), 1e-5 );

  for( size_t j = 0; j < surface.height(); ++j )
  {
    for( size_t i = 0; i < surface.width(); ++i )
    {
      EXPECT_LE( surface( i, j, 0 ), 1.0f + 1e-5f );
      EXPECT_GE( surface( i, j, 0 ), -1.0f - 1e-5f );
    }
  }
}

// ----------------------------------------------------------------------------
/// The correlation ignores exposure, which is the reason for using it.
TEST ( draw, a_template_match_ignores_brightness_and_gain )
{
  kv::image_of< uint8_t > image( 10, 8, 1 );
  uint8_t value = 11;

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      // Kept in the middle of the range so the scaled copy does not clip
      image( i, j, 0 ) =
        static_cast< uint8_t >( 60 + ( value = value * 5 + 3 ) % 60 );
    }
  }

  auto pattern = sub_image( image, 2, 1, 4, 3 );

  for( size_t j = 0; j < pattern.height(); ++j )
  {
    for( size_t i = 0; i < pattern.width(); ++i )
    {
      // Half the contrast, shifted up: a different camera, the same scene
      pattern( i, j, 0 ) = static_cast< uint8_t >(
        20 + static_cast< int >( pattern( i, j, 0 ) ) / 2 );
    }
  }

  auto const surface = io::match_template_ncc( image, pattern );

  EXPECT_NEAR( 1.0f, surface( 2, 1, 0 ), 1e-3 );
}

// ----------------------------------------------------------------------------
TEST ( draw, refuses_what_it_cannot_do )
{
  kv::image_of< uint8_t > image( 4, 4, 1 );
  kv::image_of< uint8_t > colour( 4, 4, 3 );
  kv::image_of< uint8_t > big( 8, 8, 1 );

  EXPECT_THROW( io::match_template_ncc( image, colour ),
                std::invalid_argument );
  EXPECT_THROW( io::match_template_ncc( image, big ),
                std::invalid_argument );
  EXPECT_THROW( io::match_template_ncc( image,
                                        kv::image_of< uint8_t >( 0, 0, 1 ) ),
                std::invalid_argument );

  EXPECT_THROW( io::horizontal_concat< uint8_t >( {} ),
                std::invalid_argument );
  EXPECT_THROW( io::horizontal_concat< uint8_t >( { image, big } ),
                std::invalid_argument );
  EXPECT_THROW( io::vertical_concat< uint8_t >( { image, big } ),
                std::invalid_argument );
}
