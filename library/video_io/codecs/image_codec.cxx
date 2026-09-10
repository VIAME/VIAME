/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Reading and writing the image formats VIAME ships, without OpenCV

#include "image_codec.h"

#include "tiff.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

// The vendored codecs. Compiled here, once, rather than in a header every
// consumer includes: the implementation macro makes these translation units
// of a few thousand lines apiece.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_NO_GIF
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_PSD
#define STBI_FAILURE_USERMSG
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO
#include <stb_image_write.h>

namespace viame {

namespace codecs {

namespace {

namespace kv = kwiver::vital;

// ----------------------------------------------------------------------------
/// The extension, lower cased, with its dot.
std::string
extension_of( std::string const& filename )
{
  auto const dot = filename.find_last_of( '.' );

  if( dot == std::string::npos )
  {
    return {};
  }

  auto out = filename.substr( dot );

  std::transform( out.begin(), out.end(), out.begin(),
                  []( unsigned char c ) { return std::tolower( c ); } );

  return out;
}

bool
is_tiff_extension( std::string const& extension )
{
  return extension == ".tif" || extension == ".tiff";
}

// ----------------------------------------------------------------------------
/// Whether these bytes are a BMP whose palette is gray.
///
/// A gray BMP is stored as 8 bits per pixel indexing a 256 entry palette in
/// which every entry has R == G == B. stb expands any palette to RGB, so a
/// gray BMP decodes to three identical planes; OpenCV's reader gives one
/// plane, and `tests/golden/codecs` records that. Rather than guess from the
/// decoded pixels -- an RGB image whose channels happen to agree is not the
/// same thing -- this reads the palette, which is what actually distinguishes
/// them.
///
/// The header is fixed: 14 bytes of file header with the pixel offset at 10,
/// then a DIB header whose own size is its first field, with the bit count 14
/// bytes into it. The palette follows the DIB header, four bytes an entry,
/// blue green red reserved.
bool
is_gray_palette_bmp( std::vector< uint8_t > const& bytes )
{
  auto const u16_at = [ & ]( size_t at )
  {
    return static_cast< unsigned >( bytes[ at ] ) |
           ( static_cast< unsigned >( bytes[ at + 1 ] ) << 8 );
  };
  auto const u32_at = [ & ]( size_t at )
  {
    return static_cast< uint32_t >( bytes[ at ] ) |
           ( static_cast< uint32_t >( bytes[ at + 1 ] ) << 8 ) |
           ( static_cast< uint32_t >( bytes[ at + 2 ] ) << 16 ) |
           ( static_cast< uint32_t >( bytes[ at + 3 ] ) << 24 );
  };

  if( bytes.size() < 54 || bytes[ 0 ] != 'B' || bytes[ 1 ] != 'M' )
  {
    return false;
  }

  auto const dib_size = u32_at( 14 );

  // BITMAPCOREHEADER is 12 bytes and puts the bit count elsewhere; nothing
  // writes one any more and guessing at it is not worth it
  if( dib_size < 40 )
  {
    return false;
  }

  if( u16_at( 14 + 14 ) != 8 )
  {
    return false;   // not palettised at one byte a pixel
  }

  auto const palette_at = size_t( 14 ) + dib_size;
  auto const pixels_at = size_t( u32_at( 10 ) );

  if( pixels_at <= palette_at || pixels_at > bytes.size() )
  {
    return false;
  }

  auto const entries = ( pixels_at - palette_at ) / 4;

  if( entries == 0 )
  {
    return false;
  }

  for( size_t entry = 0; entry < entries; ++entry )
  {
    auto const at = palette_at + entry * 4;

    if( bytes[ at ] != bytes[ at + 1 ] || bytes[ at + 1 ] != bytes[ at + 2 ] )
    {
      return false;
    }
  }

  return true;
}

// ----------------------------------------------------------------------------
/// The first \p limit bytes of a file, or all of it when \p limit is zero.
std::vector< uint8_t >
slurp( std::string const& filename, size_t limit = 0 )
{
  std::ifstream stream( filename, std::ios::binary );

  if( !stream )
  {
    throw std::runtime_error( "cannot open " + filename );
  }

  stream.seekg( 0, std::ios::end );
  auto const size = stream.tellg();

  if( size < 0 )
  {
    throw std::runtime_error( "cannot size " + filename );
  }

  auto wanted = static_cast< size_t >( size );

  if( limit != 0 && limit < wanted )
  {
    wanted = limit;
  }

  std::vector< uint8_t > out( wanted );
  stream.seekg( 0, std::ios::beg );
  stream.read( reinterpret_cast< char* >( out.data() ),
               static_cast< std::streamsize >( out.size() ) );

  return out;
}

/// How much of a file `can_read` looks at.
///
/// It only has to reach the header, and reading a whole 4K TIFF to answer a
/// question about its first few hundred bytes would double the cost of every
/// image the pipelines read. JPEG puts its frame header after the tables and
/// any embedded thumbnail, which is what sets the size: 64 KB clears a
/// thumbnail comfortably, and a file whose header is further in than that
/// falls back rather than being read wrong.
constexpr size_t PROBE_BYTES = 64 * 1024;

// ----------------------------------------------------------------------------
/// stb's interleaved buffer into vital's planar image.
template < typename T >
kv::image
interleaved_to_image( T const* source, int width, int height, int channels )
{
  kv::image_of< T > out( static_cast< size_t >( width ),
                         static_cast< size_t >( height ),
                         static_cast< size_t >( channels ) );

  for( int row = 0; row < height; ++row )
  {
    for( int column = 0; column < width; ++column )
    {
      for( int channel = 0; channel < channels; ++channel )
      {
        out( column, row, channel ) =
          source[ ( size_t( row ) * width + column ) * channels + channel ];
      }
    }
  }

  return kv::image( out );
}

// ----------------------------------------------------------------------------
/// vital's planar image into an interleaved buffer.
template < typename T >
std::vector< T >
image_to_interleaved( kv::image const& image )
{
  kv::image_of< T > typed( image );

  std::vector< T > out( typed.width() * typed.height() * typed.depth() );

  size_t at = 0;

  for( size_t row = 0; row < typed.height(); ++row )
  {
    for( size_t column = 0; column < typed.width(); ++column )
    {
      for( size_t plane = 0; plane < typed.depth(); ++plane )
      {
        out[ at++ ] = typed( column, row, plane );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// A 16 bit image narrowed to 8, which is what a JPEG or a BMP can hold.
///
/// Saturated, not shifted and not rescaled: a value above 255 becomes 255
/// and everything below is kept as it is. That is what OpenCV's `imwrite`
/// does -- `tests/golden/codecs` round trips a 16 bit gray through BMP and
/// the recording is saturated, not shifted -- and it is the only one of the
/// three that leaves an already-8-bit-ranged image alone.
///
/// It also throws most of a real 16 bit sensor frame away, which is why
/// `core_image_io` has `force_byte` and the stretch options: a caller that
/// means to narrow says how.
kv::image
narrow_to_byte( kv::image const& image )
{
  kv::image_of< uint16_t > source( image );
  kv::image_of< uint8_t > out( source.width(), source.height(),
                               source.depth() );

  for( size_t row = 0; row < source.height(); ++row )
  {
    for( size_t column = 0; column < source.width(); ++column )
    {
      for( size_t plane = 0; plane < source.depth(); ++plane )
      {
        auto const value = source( column, row, plane );
        out( column, row, plane ) =
          static_cast< uint8_t >( value > 255 ? 255 : value );
      }
    }
  }

  return kv::image( out );
}

// ----------------------------------------------------------------------------
/// An 8 bit gray BMP: one byte a pixel against a 256 entry gray palette.
///
/// stb's BMP writer replicates a single channel into 24 bit BGR, so a gray
/// image written through it reads back as three planes. OpenCV wrote a
/// palettised one and `tests/golden/codecs` records that, so this writes one
/// too. The format is small enough that borrowing a whole encoder for it
/// would be the larger cost.
std::vector< uint8_t >
encode_gray_bmp( kv::image_of< uint8_t > const& image )
{
  auto const width = image.width();
  auto const height = image.height();

  // Rows are padded to a multiple of four bytes, and stored bottom up
  auto const stride = ( width + 3 ) & ~size_t( 3 );

  constexpr uint32_t FILE_HEADER = 14;
  constexpr uint32_t DIB_HEADER = 40;
  constexpr uint32_t PALETTE = 256 * 4;

  auto const pixels_at = FILE_HEADER + DIB_HEADER + PALETTE;
  auto const pixels_size = static_cast< uint32_t >( stride * height );

  std::vector< uint8_t > out;
  out.reserve( pixels_at + pixels_size );

  auto const u16 = [ &out ]( uint16_t value )
  {
    out.push_back( static_cast< uint8_t >( value & 0xFF ) );
    out.push_back( static_cast< uint8_t >( value >> 8 ) );
  };
  auto const u32 = [ &out ]( uint32_t value )
  {
    for( int shift = 0; shift < 32; shift += 8 )
    {
      out.push_back( static_cast< uint8_t >( ( value >> shift ) & 0xFF ) );
    }
  };

  out.push_back( 'B' );
  out.push_back( 'M' );
  u32( pixels_at + pixels_size );
  u32( 0 );
  u32( pixels_at );

  u32( DIB_HEADER );
  u32( static_cast< uint32_t >( width ) );
  u32( static_cast< uint32_t >( height ) );
  u16( 1 );                 // planes
  u16( 8 );                 // bits per pixel
  u32( 0 );                 // BI_RGB, uncompressed
  u32( pixels_size );
  u32( 2835 );              // 72 dpi in pixels per metre, as OpenCV writes
  u32( 2835 );
  u32( 256 );               // colours used
  u32( 256 );               // colours important

  for( unsigned entry = 0; entry < 256; ++entry )
  {
    auto const value = static_cast< uint8_t >( entry );
    out.push_back( value );   // blue
    out.push_back( value );   // green
    out.push_back( value );   // red
    out.push_back( 0 );
  }

  for( size_t row = 0; row < height; ++row )
  {
    auto const source_row = height - 1 - row;

    for( size_t column = 0; column < width; ++column )
    {
      out.push_back( image( column, source_row, 0 ) );
    }

    out.insert( out.end(), stride - width, 0 );
  }

  return out;
}

// ----------------------------------------------------------------------------
/// stb_image_write's sink, collecting into a vector.
void
collect( void* context, void* data, int size )
{
  auto* out = static_cast< std::vector< uint8_t >* >( context );
  auto const* bytes = static_cast< uint8_t const* >( data );
  out->insert( out->end(), bytes, bytes + size );
}

void
put( std::string const& filename, std::vector< uint8_t > const& bytes )
{
  std::ofstream stream( filename, std::ios::binary );

  if( !stream )
  {
    throw std::runtime_error( "cannot write " + filename );
  }

  stream.write( reinterpret_cast< char const* >( bytes.data() ),
                static_cast< std::streamsize >( bytes.size() ) );

  if( !stream )
  {
    throw std::runtime_error( "failed writing " + filename );
  }
}

} // namespace

// ----------------------------------------------------------------------------
bool
can_read( std::string const& filename, std::string& reason )
{
  reason.clear();

  std::vector< uint8_t > head;

  try
  {
    head = slurp( filename, PROBE_BYTES );
  }
  catch( std::exception const& e )
  {
    reason = e.what();
    return false;
  }

  if( tiff::is_tiff( head.data(), head.size() ) )
  {
    reason = tiff::unsupported_reason( filename );
    return reason.empty();
  }

  int width = 0;
  int height = 0;
  int channels = 0;

  if( !stbi_info_from_memory( head.data(), static_cast< int >( head.size() ),
                              &width, &height, &channels ) )
  {
    auto const* failure = stbi_failure_reason();
    reason = failure ? failure : "not a format stb recognises";
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
bool
can_write( std::string const& filename, kv::image const& image,
           std::string& reason )
{
  reason.clear();

  auto const extension = extension_of( filename );

  if( extension != ".png" && extension != ".jpg" && extension != ".jpeg" &&
      extension != ".bmp" && !is_tiff_extension( extension ) )
  {
    reason = "no encoder here for '" + extension + "'";
    return false;
  }

  auto const trait = image.pixel_traits();

  if( trait.type != kv::image_pixel_traits::UNSIGNED ||
      ( trait.num_bytes != 1 && trait.num_bytes != 2 ) )
  {
    reason = "these encoders take 8 and 16 bit unsigned images only";
    return false;
  }

  if( trait.num_bytes == 2 && extension == ".png" )
  {
    reason = "stb writes 8 bit PNG only, and narrowing a 16 bit image here "
             "would lose the range silently";
    return false;
  }

  auto const depth = image.depth();

  if( is_tiff_extension( extension ) && depth != 1 && depth != 3 &&
      depth != 4 )
  {
    reason = "TIFF here writes 1, 3 or 4 samples per pixel";
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
kv::image
read( std::string const& filename )
{
  auto const bytes = slurp( filename );

  if( tiff::is_tiff( bytes.data(), bytes.size() ) )
  {
    return tiff::read( filename );
  }

  int width = 0;
  int height = 0;
  int channels = 0;

  auto const size = static_cast< int >( bytes.size() );

  // A gray BMP is palettised, and stb expands any palette to RGB. Asking for
  // one channel gives the gray back: stb's luminance of three equal samples
  // is that sample exactly, and OpenCV's reader gives one plane here too.
  auto const wanted = is_gray_palette_bmp( bytes ) ? 1 : 0;

  if( stbi_is_16_bit_from_memory( bytes.data(), size ) )
  {
    auto* pixels = stbi_load_16_from_memory( bytes.data(), size, &width,
                                             &height, &channels, wanted );

    if( !pixels )
    {
      auto const* failure = stbi_failure_reason();
      throw std::runtime_error(
        filename + ": " + ( failure ? failure : "decode failed" ) );
    }

    auto out = interleaved_to_image< uint16_t >(
      pixels, width, height, wanted ? wanted : channels );
    stbi_image_free( pixels );
    return out;
  }

  auto* pixels = stbi_load_from_memory( bytes.data(), size, &width, &height,
                                        &channels, wanted );

  if( !pixels )
  {
    auto const* failure = stbi_failure_reason();
    throw std::runtime_error(
      filename + ": " + ( failure ? failure : "decode failed" ) );
  }

  auto out = interleaved_to_image< uint8_t >(
    pixels, width, height, wanted ? wanted : channels );
  stbi_image_free( pixels );
  return out;
}

// ----------------------------------------------------------------------------
void
write( std::string const& filename, kv::image const& image )
{
  auto const extension = extension_of( filename );

  std::string reason;

  if( !can_write( filename, image, reason ) )
  {
    throw std::runtime_error( filename + ": " + reason );
  }

  if( is_tiff_extension( extension ) )
  {
    tiff::write( filename, image );
    return;
  }

  auto const sixteen = ( image.pixel_traits().num_bytes == 2 );
  auto const source = sixteen ? narrow_to_byte( image ) : image;
  auto const pixels = image_to_interleaved< uint8_t >( source );

  auto const width = static_cast< int >( source.width() );
  auto const height = static_cast< int >( source.height() );
  auto const channels = static_cast< int >( source.depth() );

  std::vector< uint8_t > encoded;
  int ok = 0;

  if( extension == ".png" )
  {
    ok = stbi_write_png_to_func( collect, &encoded, width, height, channels,
                                 pixels.data(), width * channels );
  }
  else if( extension == ".bmp" )
  {
    if( channels == 1 )
    {
      put( filename, encode_gray_bmp( kv::image_of< uint8_t >( source ) ) );
      return;
    }

    ok = stbi_write_bmp_to_func( collect, &encoded, width, height, channels,
                                 pixels.data() );
  }
  else
  {
    // 95, which is what OpenCV's imwrite defaults to
    ok = stbi_write_jpg_to_func( collect, &encoded, width, height, channels,
                                 pixels.data(), 95 );
  }

  if( !ok )
  {
    throw std::runtime_error( "encoding failed for " + filename );
  }

  put( filename, encoded );
}

} // end namespace codecs

} // end namespace viame
