/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// Unit tests for the in-house image codecs.
///
/// These check the codecs against what the formats say, on images small
/// enough to reason about by hand. What they do not check is agreement with
/// OpenCV on real files -- `tests/golden/codecs` does that, over eighteen
/// containers recorded from the reader being replaced, and it is the test
/// that would catch a decoder that is self-consistent and wrong.

#include <codecs/image_codec.h>
#include <codecs/tiff.h>

#include <viame/core_types/image.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace kv = kwiver::vital;
namespace codecs = viame::codecs;

namespace {

// ----------------------------------------------------------------------------
/// A file that deletes itself, so a failing test leaves nothing behind.
///
/// `mkstemp` rather than `tmpnam`: the name is created and claimed in one
/// step, so two tests running at once cannot pick the same one. The suffix
/// is appended after, because the codecs choose their encoder by extension.
class scratch_file
{
public:
  explicit scratch_file( std::string const& suffix )
  {
    std::string tmp = "/tmp/viame_codec_XXXXXX";
    auto const handle = ::mkstemp( &tmp[ 0 ] );

    if( handle < 0 )
    {
      throw std::runtime_error( "cannot make a scratch file" );
    }

    ::close( handle );
    ::remove( tmp.c_str() );

    path_ = tmp + suffix;
  }

  ~scratch_file() { std::remove( path_.c_str() ); }

  std::string const& path() const { return path_; }

private:
  std::string path_;
};

// ----------------------------------------------------------------------------
/// A gradient with a distinct value per (x, y, plane), so a transposed or
/// channel-swapped result is visible rather than plausible.
template < typename T >
kv::image_of< T >
ramp( size_t width, size_t height, size_t depth, T scale = 1 )
{
  kv::image_of< T > out( width, height, depth );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      for( size_t p = 0; p < depth; ++p )
      {
        out( i, j, p ) =
          static_cast< T >( ( i * 7 + j * 31 + p * 101 ) % 251 ) * scale;
      }
    }
  }

  return out;
}

template < typename T >
void
expect_same( kv::image const& actual, kv::image_of< T > const& expected )
{
  ASSERT_EQ( expected.width(), actual.width() );
  ASSERT_EQ( expected.height(), actual.height() );
  ASSERT_EQ( expected.depth(), actual.depth() );
  ASSERT_EQ( sizeof( T ), actual.pixel_traits().num_bytes );

  kv::image_of< T > typed( actual );

  for( size_t j = 0; j < expected.height(); ++j )
  {
    for( size_t i = 0; i < expected.width(); ++i )
    {
      for( size_t p = 0; p < expected.depth(); ++p )
      {
        ASSERT_EQ( expected( i, j, p ), typed( i, j, p ) )
          << "at (" << i << ", " << j << ", " << p << ")";
      }
    }
  }
}

} // namespace

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST ( codecs, png_round_trip_8_bit )
{
  auto const source = ramp< uint8_t >( 13, 7, 3 );
  scratch_file file( ".png" );

  codecs::write( file.path(), kv::image( source ) );
  expect_same( codecs::read( file.path() ), source );
}

// ----------------------------------------------------------------------------
TEST ( codecs, png_round_trip_gray_and_alpha )
{
  for( size_t depth : { size_t{ 1 }, size_t{ 4 } } )
  {
    auto const source = ramp< uint8_t >( 9, 5, depth );
    scratch_file file( ".png" );

    codecs::write( file.path(), kv::image( source ) );
    expect_same( codecs::read( file.path() ), source );
  }
}

// ----------------------------------------------------------------------------
TEST ( codecs, bmp_round_trip )
{
  auto const source = ramp< uint8_t >( 11, 6, 3 );
  scratch_file file( ".bmp" );

  codecs::write( file.path(), kv::image( source ) );
  expect_same( codecs::read( file.path() ), source );
}

// ----------------------------------------------------------------------------
/// JPEG is lossy, so what is checked is the geometry and that the values are
/// close, not that they are equal.
TEST ( codecs, jpeg_round_trip_is_close )
{
  auto const source = ramp< uint8_t >( 32, 16, 3 );
  scratch_file file( ".jpg" );

  codecs::write( file.path(), kv::image( source ) );
  auto const back = codecs::read( file.path() );

  ASSERT_EQ( source.width(), back.width() );
  ASSERT_EQ( source.height(), back.height() );
  ASSERT_EQ( source.depth(), back.depth() );

  kv::image_of< uint8_t > typed( back );
  double total = 0.0;

  for( size_t j = 0; j < source.height(); ++j )
  {
    for( size_t i = 0; i < source.width(); ++i )
    {
      for( size_t p = 0; p < source.depth(); ++p )
      {
        total += std::abs( int( source( i, j, p ) ) - int( typed( i, j, p ) ) );
      }
    }
  }

  // A synthetic ramp at quality 95 with 4:2:0 chroma; generous, because the
  // point here is that it decoded at all and in the right order
  auto const mean = total / ( source.width() * source.height() *
                              source.depth() );
  EXPECT_LT( mean, 20.0 ) << "mean absolute error after a JPEG round trip";
}

// ----------------------------------------------------------------------------
TEST ( codecs, tiff_round_trip_8_bit )
{
  for( size_t depth : { size_t{ 1 }, size_t{ 3 }, size_t{ 4 } } )
  {
    auto const source = ramp< uint8_t >( 10, 4, depth );
    scratch_file file( ".tif" );

    codecs::write( file.path(), kv::image( source ) );
    expect_same( codecs::read( file.path() ), source );
  }
}

// ----------------------------------------------------------------------------
TEST ( codecs, tiff_round_trip_16_bit )
{
  for( size_t depth : { size_t{ 1 }, size_t{ 3 } } )
  {
    auto const source = ramp< uint16_t >( 10, 4, depth, 200 );
    scratch_file file( ".tif" );

    codecs::write( file.path(), kv::image( source ) );
    expect_same( codecs::read( file.path() ), source );
  }
}

// ----------------------------------------------------------------------------
/// A TIFF this reader writes is one it says it can read.
TEST ( codecs, tiff_written_here_is_readable_here )
{
  auto const source = ramp< uint16_t >( 8, 3, 3, 100 );
  scratch_file file( ".tif" );

  codecs::write( file.path(), kv::image( source ) );

  std::string reason;
  EXPECT_TRUE( codecs::can_read( file.path(), reason ) ) << reason;
  EXPECT_TRUE( reason.empty() ) << reason;
  EXPECT_TRUE( codecs::tiff::unsupported_reason( file.path() ).empty() );
}

// ----------------------------------------------------------------------------
TEST ( codecs, is_tiff_recognises_both_byte_orders )
{
  uint8_t const little[] = { 'I', 'I', 42, 0 };
  uint8_t const big[] = { 'M', 'M', 0, 42 };
  uint8_t const png[] = { 0x89, 'P', 'N', 'G' };

  EXPECT_TRUE( codecs::tiff::is_tiff( little, sizeof( little ) ) );
  EXPECT_TRUE( codecs::tiff::is_tiff( big, sizeof( big ) ) );
  EXPECT_FALSE( codecs::tiff::is_tiff( png, sizeof( png ) ) );
  EXPECT_FALSE( codecs::tiff::is_tiff( little, 2 ) );
}

// ----------------------------------------------------------------------------
/// The point of `can_read`: a caller finds out before failing.
TEST ( codecs, can_read_declines_rather_than_throws )
{
  scratch_file file( ".tif" );

  {
    // A TIFF header and nothing else: well formed enough to be recognised,
    // not well formed enough to read
    std::ofstream stream( file.path(), std::ios::binary );
    uint8_t const header[] = { 'I', 'I', 42, 0, 8, 0, 0, 0, 0, 0 };
    stream.write( reinterpret_cast< char const* >( header ),
                  sizeof( header ) );
  }

  std::string reason;
  EXPECT_FALSE( codecs::can_read( file.path(), reason ) );
  EXPECT_FALSE( reason.empty() );
}

// ----------------------------------------------------------------------------
TEST ( codecs, can_write_takes_the_extension_and_the_pixels )
{
  auto const eight = kv::image( ramp< uint8_t >( 4, 4, 3 ) );
  std::string reason;

  for( auto const* name : { "a.png", "a.PNG", "a.jpg", "a.jpeg", "a.bmp",
                            "a.tif", "a.tiff" } )
  {
    EXPECT_TRUE( codecs::can_write( name, eight, reason ) ) << name;
    EXPECT_TRUE( reason.empty() ) << name << ": " << reason;
  }

  for( auto const* name : { "a.exr", "a.webp", "a" } )
  {
    EXPECT_FALSE( codecs::can_write( name, eight, reason ) ) << name;
    EXPECT_FALSE( reason.empty() ) << name;
  }
}

// ----------------------------------------------------------------------------
/// The one combination that goes to the fallback: stb writes 8 bit PNG, and
/// narrowing here would lose the range without anyone asking.
TEST ( codecs, declines_16_bit_png )
{
  auto const source = kv::image( ramp< uint16_t >( 4, 4, 1, 300 ) );
  scratch_file file( ".png" );

  std::string reason;
  EXPECT_FALSE( codecs::can_write( file.path(), source, reason ) );
  EXPECT_FALSE( reason.empty() );

  EXPECT_THROW( codecs::write( file.path(), source ), std::runtime_error );

  // 16 bit TIFF, the same pixels, is written here rather than declined
  scratch_file as_tiff( ".tif" );
  EXPECT_TRUE( codecs::can_write( as_tiff.path(), source, reason ) )
    << reason;
}

// ----------------------------------------------------------------------------
TEST ( codecs, refuses_a_float_image )
{
  auto const source = kv::image( kv::image_of< float >( 4, 4, 1 ) );
  scratch_file file( ".tif" );

  std::string reason;
  EXPECT_FALSE( codecs::can_write( file.path(), source, reason ) );
  EXPECT_THROW( codecs::write( file.path(), source ), std::runtime_error );
}

// ----------------------------------------------------------------------------
/// A 16 bit image narrowed for a container that cannot hold it saturates,
/// which is what OpenCV's writer did -- see tests/golden/codecs, which round
/// trips a 16 bit gray through BMP.
TEST ( codecs, narrowing_saturates_rather_than_shifting )
{
  kv::image_of< uint16_t > source( 3, 1, 1 );
  source( 0, 0, 0 ) = 7;
  source( 1, 0, 0 ) = 255;
  source( 2, 0, 0 ) = 60000;

  scratch_file file( ".bmp" );
  codecs::write( file.path(), kv::image( source ) );

  kv::image_of< uint8_t > back( codecs::read( file.path() ) );

  EXPECT_EQ( 7, back( 0, 0, 0 ) );
  EXPECT_EQ( 255, back( 1, 0, 0 ) );
  EXPECT_EQ( 255, back( 2, 0, 0 ) );
}
