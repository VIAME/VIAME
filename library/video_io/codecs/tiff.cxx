/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Baseline TIFF, read and written in house

#include "tiff.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace viame {

namespace codecs {

namespace tiff {

namespace {

namespace kv = kwiver::vital;

// ----------------------------------------------------------------------------
// The tags this reader looks at. Everything else in the file is skipped;
// baseline TIFF requires a reader to ignore tags it does not know.
enum tag_id : uint16_t
{
  TAG_IMAGE_WIDTH      = 256,
  TAG_IMAGE_LENGTH     = 257,
  TAG_BITS_PER_SAMPLE  = 258,
  TAG_COMPRESSION      = 259,
  TAG_PHOTOMETRIC      = 262,
  TAG_STRIP_OFFSETS    = 273,
  TAG_SAMPLES_PER_PIXEL= 277,
  TAG_ROWS_PER_STRIP   = 278,
  TAG_STRIP_BYTE_COUNTS= 279,
  TAG_PLANAR_CONFIG    = 284,
  TAG_PREDICTOR        = 317,
  TAG_TILE_WIDTH       = 322,
  TAG_TILE_LENGTH      = 323,
  TAG_SAMPLE_FORMAT    = 339,
};

enum compression_id : uint32_t
{
  COMPRESSION_NONE     = 1,
  COMPRESSION_LZW      = 5,
  COMPRESSION_PACKBITS = 32773,
};

// ----------------------------------------------------------------------------
/// Whether this machine stores a uint16_t low byte first.
bool
host_is_little_endian()
{
  uint16_t const probe = 1;
  uint8_t first;
  std::memcpy( &first, &probe, 1 );
  return first == 1;
}

// ----------------------------------------------------------------------------
/// A file read whole, with the byte order its header declares.
///
/// Read whole rather than streamed: the strip offsets point anywhere in the
/// file and a TIFF is not required to put them in order, so a reader that
/// seeks is a reader that seeks backwards. The files this handles are frames,
/// not mosaics.
class reader
{
public:
  explicit reader( std::string const& filename )
  {
    std::ifstream stream( filename, std::ios::binary );

    if( !stream )
    {
      throw std::runtime_error( "cannot open " + filename );
    }

    stream.seekg( 0, std::ios::end );
    auto const size = stream.tellg();

    if( size < 8 )
    {
      throw std::runtime_error( filename + " is too short to be a TIFF" );
    }

    stream.seekg( 0, std::ios::beg );
    d_.resize( static_cast< size_t >( size ) );
    stream.read( reinterpret_cast< char* >( d_.data() ),
                 static_cast< std::streamsize >( d_.size() ) );

    if( d_[ 0 ] == 'I' && d_[ 1 ] == 'I' )
    {
      little_ = true;
    }
    else if( d_[ 0 ] == 'M' && d_[ 1 ] == 'M' )
    {
      little_ = false;
    }
    else
    {
      throw std::runtime_error( filename + " is not a TIFF" );
    }

    if( u16( 2 ) != 42 )
    {
      throw std::runtime_error( filename + " is not a TIFF" );
    }
  }

  size_t size() const { return d_.size(); }
  uint8_t const* data() const { return d_.data(); }

  uint8_t u8( size_t at ) const
  {
    check( at, 1 );
    return d_[ at ];
  }

  uint16_t u16( size_t at ) const
  {
    check( at, 2 );
    return little_
      ? static_cast< uint16_t >( d_[ at ] | ( d_[ at + 1 ] << 8 ) )
      : static_cast< uint16_t >( ( d_[ at ] << 8 ) | d_[ at + 1 ] );
  }

  uint32_t u32( size_t at ) const
  {
    check( at, 4 );
    return little_
      ? ( uint32_t( d_[ at ] ) | ( uint32_t( d_[ at + 1 ] ) << 8 ) |
          ( uint32_t( d_[ at + 2 ] ) << 16 ) |
          ( uint32_t( d_[ at + 3 ] ) << 24 ) )
      : ( ( uint32_t( d_[ at ] ) << 24 ) |
          ( uint32_t( d_[ at + 1 ] ) << 16 ) |
          ( uint32_t( d_[ at + 2 ] ) << 8 ) | uint32_t( d_[ at + 3 ] ) );
  }

  bool little_endian() const { return little_; }

private:
  void check( size_t at, size_t want ) const
  {
    if( at + want > d_.size() )
    {
      throw std::runtime_error( "TIFF reads past the end of the file" );
    }
  }

  std::vector< uint8_t > d_;
  bool little_ = true;
};

// ----------------------------------------------------------------------------
/// One IFD entry, with its values already resolved.
struct field
{
  uint16_t type = 0;
  std::vector< uint32_t > values;

  uint32_t first( uint32_t fallback = 0 ) const
  {
    return values.empty() ? fallback : values[ 0 ];
  }
};

/// The size in bytes of one value of each TIFF field type.
size_t
type_size( uint16_t type )
{
  switch( type )
  {
    case 1:  return 1;   // BYTE
    case 2:  return 1;   // ASCII
    case 3:  return 2;   // SHORT
    case 4:  return 4;   // LONG
    case 5:  return 8;   // RATIONAL
    case 6:  return 1;   // SBYTE
    case 7:  return 1;   // UNDEFINED
    case 8:  return 2;   // SSHORT
    case 9:  return 4;   // SLONG
    case 10: return 8;   // SRATIONAL
    case 11: return 4;   // FLOAT
    case 12: return 8;   // DOUBLE
    default: return 0;
  }
}

/// Read the first IFD's fields, by tag.
std::map< uint16_t, field >
read_ifd( reader const& file )
{
  std::map< uint16_t, field > fields;

  auto const ifd = file.u32( 4 );
  auto const count = file.u16( ifd );

  for( uint16_t index = 0; index < count; ++index )
  {
    auto const at = ifd + 2 + size_t( index ) * 12;
    auto const tag = file.u16( at );
    auto const type = file.u16( at + 2 );
    auto const values = file.u32( at + 4 );

    auto const width = type_size( type );

    if( width == 0 || width > 4 )
    {
      // A type this reader has no use for; baseline says skip it
      continue;
    }

    auto const bytes = width * size_t( values );
    auto const base = ( bytes <= 4 ) ? at + 8 : size_t( file.u32( at + 8 ) );

    field entry;
    entry.type = type;
    entry.values.reserve( values );

    for( uint32_t value = 0; value < values; ++value )
    {
      auto const where = base + value * width;

      switch( width )
      {
        case 1: entry.values.push_back( file.u8( where ) ); break;
        case 2: entry.values.push_back( file.u16( where ) ); break;
        default: entry.values.push_back( file.u32( where ) ); break;
      }
    }

    fields[ tag ] = std::move( entry );
  }

  return fields;
}

// ----------------------------------------------------------------------------
/// PackBits, which is TIFF's spelling of run length encoding.
///
/// A signed count byte: 0..127 means the next count+1 bytes are literal,
/// -1..-127 means the next byte repeats 1-count times, -128 is a no-op.
void
unpack_packbits( uint8_t const* source, size_t source_size,
                 std::vector< uint8_t >& out, size_t expected )
{
  size_t at = 0;

  while( at < source_size && out.size() < expected )
  {
    auto const control = static_cast< int8_t >( source[ at++ ] );

    if( control >= 0 )
    {
      auto const count = static_cast< size_t >( control ) + 1;

      if( at + count > source_size )
      {
        throw std::runtime_error( "PackBits literal runs past the strip" );
      }

      out.insert( out.end(), source + at, source + at + count );
      at += count;
    }
    else if( control != -128 )
    {
      auto const count = static_cast< size_t >( 1 - control );

      if( at >= source_size )
      {
        throw std::runtime_error( "PackBits repeat runs past the strip" );
      }

      out.insert( out.end(), count, source[ at++ ] );
    }
  }
}

// ----------------------------------------------------------------------------
/// The first byte of the string a code expands to.
///
/// The table stores a string as (prefix, last byte), so its first byte is at
/// the end of the prefix chain.
uint8_t
suffix_first( std::vector< uint16_t > const& prefix,
              std::vector< uint8_t > const& suffix, uint16_t code )
{
  constexpr uint16_t LIMIT = 4096;

  while( prefix[ code ] != LIMIT )
  {
    code = prefix[ code ];
  }

  return suffix[ code ];
}

// ----------------------------------------------------------------------------
/// TIFF's LZW, which differs from the original in two ways that matter.
///
/// Codes are packed most significant bit first, and the code width grows one
/// code *early* -- at 511, 1023 and 2047 rather than at 512, 1024 and 2048.
/// That off-by-one is in the TIFF 6.0 specification as a mistake that became
/// the format, and a decoder without it desynchronises a few hundred codes in.
void
unpack_lzw( uint8_t const* source, size_t source_size,
            std::vector< uint8_t >& out, size_t expected )
{
  constexpr uint16_t CLEAR = 256;
  constexpr uint16_t EOI = 257;
  constexpr uint16_t FIRST = 258;
  constexpr uint16_t LIMIT = 4096;

  // Entries are (prefix code, appended byte); a string is walked backwards
  // from its last byte. Flat arrays rather than vectors of vectors: the
  // table is rebuilt on every clear code, and this way that is two writes.
  std::vector< uint16_t > prefix( LIMIT, 0 );
  std::vector< uint8_t > suffix( LIMIT, 0 );
  std::vector< uint16_t > length( LIMIT, 0 );

  auto reset = [ & ]( uint16_t& next, unsigned& width )
  {
    for( uint16_t code = 0; code < 256; ++code )
    {
      prefix[ code ] = LIMIT;
      suffix[ code ] = static_cast< uint8_t >( code );
      length[ code ] = 1;
    }
    next = FIRST;
    width = 9;
  };

  std::vector< uint8_t > scratch;

  auto emit = [ & ]( uint16_t code )
  {
    scratch.clear();
    scratch.reserve( length[ code ] );

    for( uint16_t at = code; at != LIMIT; at = prefix[ at ] )
    {
      scratch.push_back( suffix[ at ] );
    }

    out.insert( out.end(), scratch.rbegin(), scratch.rend() );
  };

  uint16_t next = FIRST;
  unsigned width = 9;
  reset( next, width );

  uint32_t bits = 0;
  unsigned held = 0;
  size_t at = 0;
  uint16_t previous = LIMIT;

  while( out.size() < expected )
  {
    while( held < width )
    {
      if( at >= source_size )
      {
        return;   // truncated strip: what was decoded is what there is
      }

      bits = ( bits << 8 ) | source[ at++ ];
      held += 8;
    }

    auto const code =
      static_cast< uint16_t >( ( bits >> ( held - width ) ) &
                               ( ( 1u << width ) - 1u ) );
    held -= width;

    if( code == EOI )
    {
      return;
    }

    if( code == CLEAR )
    {
      reset( next, width );
      previous = LIMIT;
      continue;
    }

    if( previous == LIMIT )
    {
      if( code >= 256 )
      {
        throw std::runtime_error( "LZW starts with a table code" );
      }

      emit( code );
      previous = code;
      continue;
    }

    if( code < next )
    {
      emit( code );

      if( next < LIMIT )
      {
        prefix[ next ] = previous;
        suffix[ next ] = suffix_first( prefix, suffix, code );
        length[ next ] = static_cast< uint16_t >( length[ previous ] + 1 );
        ++next;
      }
    }
    else if( code == next )
    {
      // The KwKwK case: the code being defined is the one being read
      if( next < LIMIT )
      {
        prefix[ next ] = previous;
        suffix[ next ] = suffix_first( prefix, suffix, previous );
        length[ next ] = static_cast< uint16_t >( length[ previous ] + 1 );
        emit( next );
        ++next;
      }
    }
    else
    {
      throw std::runtime_error( "LZW code is past the end of the table" );
    }

    previous = code;

    // One code early, which is TIFF's mistake and TIFF's format
    if( next + 1 >= ( 1u << width ) && width < 12 )
    {
      ++width;
    }
  }
}

// ----------------------------------------------------------------------------
/// Undo horizontal differencing, which LZW's predictor 2 applies per row.
template < typename T >
void
undo_predictor( std::vector< uint8_t >& bytes, size_t width, size_t rows,
                size_t samples )
{
  auto* values = reinterpret_cast< T* >( bytes.data() );

  for( size_t row = 0; row < rows; ++row )
  {
    auto* line = values + row * width * samples;

    for( size_t column = 1; column < width; ++column )
    {
      for( size_t sample = 0; sample < samples; ++sample )
      {
        line[ column * samples + sample ] =
          static_cast< T >( line[ column * samples + sample ] +
                            line[ ( column - 1 ) * samples + sample ] );
      }
    }
  }
}

} // namespace

// ----------------------------------------------------------------------------
namespace {

/// What the reader needs from the IFD, and whether it can be honoured.
struct layout
{
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t samples = 1;
  uint32_t bits = 8;
  uint32_t compression = COMPRESSION_NONE;
  uint32_t predictor = 1;
  uint32_t rows_per_strip = 0;
  std::vector< uint32_t > strip_offsets;
  std::vector< uint32_t > strip_counts;

  /// Empty when the file can be read; otherwise why not.
  std::string refusal;
};

layout
describe( reader const& file )
{
  auto const fields = read_ifd( file );

  auto get = [ & ]( uint16_t tag ) -> field const*
  {
    auto const found = fields.find( tag );
    return found == fields.end() ? nullptr : &found->second;
  };

  layout out;

  if( get( TAG_TILE_WIDTH ) || get( TAG_TILE_LENGTH ) )
  {
    out.refusal = "tiled rather than stripped";
    return out;
  }

  auto const* width = get( TAG_IMAGE_WIDTH );
  auto const* height = get( TAG_IMAGE_LENGTH );
  auto const* offsets = get( TAG_STRIP_OFFSETS );
  auto const* counts = get( TAG_STRIP_BYTE_COUNTS );

  if( !width || !height || !offsets || !counts )
  {
    out.refusal = "no width, height or strip table";
    return out;
  }

  out.width = width->first();
  out.height = height->first();
  out.strip_offsets = offsets->values;
  out.strip_counts = counts->values;

  if( auto const* samples = get( TAG_SAMPLES_PER_PIXEL ) )
  {
    out.samples = samples->first( 1 );
  }

  if( auto const* bits = get( TAG_BITS_PER_SAMPLE ) )
  {
    out.bits = bits->first( 8 );

    for( auto const value : bits->values )
    {
      if( value != out.bits )
      {
        out.refusal = "samples of different widths";
        return out;
      }
    }
  }

  if( auto const* compression = get( TAG_COMPRESSION ) )
  {
    out.compression = compression->first( COMPRESSION_NONE );
  }

  if( auto const* predictor = get( TAG_PREDICTOR ) )
  {
    out.predictor = predictor->first( 1 );
  }

  if( auto const* rows = get( TAG_ROWS_PER_STRIP ) )
  {
    out.rows_per_strip = rows->first( out.height );
  }

  if( out.rows_per_strip == 0 || out.rows_per_strip > out.height )
  {
    out.rows_per_strip = out.height;
  }

  if( auto const* planar = get( TAG_PLANAR_CONFIG ) )
  {
    if( planar->first( 1 ) != 1 )
    {
      out.refusal = "planar rather than contiguous samples";
      return out;
    }
  }

  if( auto const* format = get( TAG_SAMPLE_FORMAT ) )
  {
    auto const value = format->first( 1 );

    if( value != 1 && value != 4 )
    {
      out.refusal = "signed or floating point samples";
      return out;
    }
  }

  if( out.bits != 8 && out.bits != 16 )
  {
    std::ostringstream reason;
    reason << out.bits << " bits per sample";
    out.refusal = reason.str();
    return out;
  }

  if( out.samples != 1 && out.samples != 3 && out.samples != 4 )
  {
    std::ostringstream reason;
    reason << out.samples << " samples per pixel";
    out.refusal = reason.str();
    return out;
  }

  if( out.compression != COMPRESSION_NONE &&
      out.compression != COMPRESSION_LZW &&
      out.compression != COMPRESSION_PACKBITS )
  {
    std::ostringstream reason;
    reason << "compression " << out.compression;
    out.refusal = reason.str();
    return out;
  }

  if( out.strip_offsets.size() != out.strip_counts.size() )
  {
    out.refusal = "strip offsets and byte counts disagree";
    return out;
  }

  if( out.predictor != 1 && out.predictor != 2 )
  {
    std::ostringstream reason;
    reason << "predictor " << out.predictor;
    out.refusal = reason.str();
    return out;
  }

  return out;
}

/// Every strip decompressed and concatenated, in row order.
std::vector< uint8_t >
decode_strips( reader const& file, layout const& shape )
{
  auto const sample_bytes = shape.bits / 8;
  auto const row_bytes = size_t( shape.width ) * shape.samples * sample_bytes;

  std::vector< uint8_t > pixels;
  pixels.reserve( row_bytes * shape.height );

  for( size_t strip = 0; strip < shape.strip_offsets.size(); ++strip )
  {
    auto const first_row = strip * size_t( shape.rows_per_strip );

    if( first_row >= shape.height )
    {
      // More strips than the height accounts for: the rest describe rows
      // that do not exist. Stopping is what libtiff does too.
      break;
    }

    auto const offset = size_t( shape.strip_offsets[ strip ] );
    auto const count = size_t( shape.strip_counts[ strip ] );

    if( offset + count > file.size() )
    {
      throw std::runtime_error( "TIFF strip runs past the end of the file" );
    }

    auto const rows_here =
      std::min< size_t >( shape.rows_per_strip, shape.height - first_row );
    auto const expected = row_bytes * rows_here;

    std::vector< uint8_t > strip_bytes;
    strip_bytes.reserve( expected );

    auto const* source = file.data() + offset;

    switch( shape.compression )
    {
      case COMPRESSION_NONE:
        strip_bytes.assign( source, source + std::min( count, expected ) );
        break;

      case COMPRESSION_PACKBITS:
        unpack_packbits( source, count, strip_bytes, expected );
        break;

      default:
        unpack_lzw( source, count, strip_bytes, expected );
        break;
    }

    strip_bytes.resize( expected, 0 );

    if( shape.predictor == 2 )
    {
      if( sample_bytes == 1 )
      {
        undo_predictor< uint8_t >( strip_bytes, shape.width, rows_here,
                                   shape.samples );
      }
      else
      {
        undo_predictor< uint16_t >( strip_bytes, shape.width, rows_here,
                                    shape.samples );
      }
    }

    pixels.insert( pixels.end(), strip_bytes.begin(), strip_bytes.end() );
  }

  pixels.resize( row_bytes * shape.height, 0 );
  return pixels;
}

/// Interleaved file bytes into vital's planar image.
template < typename T >
kv::image
to_image( std::vector< uint8_t > const& bytes, layout const& shape,
          bool file_is_little )
{
  kv::image_of< T > out( shape.width, shape.height, shape.samples );

  auto const* source = bytes.data();
  bool const swap = ( sizeof( T ) == 2 ) &&
                    ( file_is_little != host_is_little_endian() );

  for( size_t row = 0; row < shape.height; ++row )
  {
    for( size_t column = 0; column < shape.width; ++column )
    {
      for( size_t sample = 0; sample < shape.samples; ++sample )
      {
        auto const at =
          ( ( row * shape.width + column ) * shape.samples + sample ) *
          sizeof( T );

        T value;
        std::memcpy( &value, source + at, sizeof( T ) );

        if( swap )
        {
          auto const raw = static_cast< uint16_t >( value );
          value = static_cast< T >( ( raw >> 8 ) | ( raw << 8 ) );
        }

        out( column, row, sample ) = value;
      }
    }
  }

  return kv::image( out );
}

} // namespace

// ----------------------------------------------------------------------------
bool
is_tiff( void const* data, size_t size )
{
  if( size < 4 )
  {
    return false;
  }

  auto const* bytes = static_cast< uint8_t const* >( data );

  if( bytes[ 0 ] == 'I' && bytes[ 1 ] == 'I' )
  {
    return bytes[ 2 ] == 42 && bytes[ 3 ] == 0;
  }

  if( bytes[ 0 ] == 'M' && bytes[ 1 ] == 'M' )
  {
    return bytes[ 2 ] == 0 && bytes[ 3 ] == 42;
  }

  return false;
}

// ----------------------------------------------------------------------------
std::string
unsupported_reason( std::string const& filename )
{
  try
  {
    reader file( filename );
    return describe( file ).refusal;
  }
  catch( std::exception const& e )
  {
    return e.what();
  }
}

// ----------------------------------------------------------------------------
kv::image
read( std::string const& filename )
{
  reader file( filename );
  auto const shape = describe( file );

  if( !shape.refusal.empty() )
  {
    throw std::runtime_error( filename + ": " + shape.refusal );
  }

  auto const bytes = decode_strips( file, shape );

  return shape.bits == 8
    ? to_image< uint8_t >( bytes, shape, file.little_endian() )
    : to_image< uint16_t >( bytes, shape, file.little_endian() );
}

// ----------------------------------------------------------------------------
namespace {

/// Append `value` in the machine's own byte order.
template < typename T >
void
append( std::vector< uint8_t >& out, T value )
{
  auto const* bytes = reinterpret_cast< uint8_t const* >( &value );
  out.insert( out.end(), bytes, bytes + sizeof( T ) );
}

/// One IFD entry, with its value inline.
void
append_entry( std::vector< uint8_t >& out, uint16_t tag, uint16_t type,
              uint32_t count, uint32_t value )
{
  append< uint16_t >( out, tag );
  append< uint16_t >( out, type );
  append< uint32_t >( out, count );

  // A SHORT that fits sits in the high or low half depending on byte order;
  // written little endian, it is the low half.
  if( type == 3 && count == 1 )
  {
    append< uint16_t >( out, static_cast< uint16_t >( value ) );
    append< uint16_t >( out, 0 );
  }
  else
  {
    append< uint32_t >( out, value );
  }
}

/// vital's planar image into interleaved file bytes.
template < typename T >
std::vector< uint8_t >
from_image( kv::image const& image )
{
  kv::image_of< T > typed( image );

  std::vector< uint8_t > out;
  out.reserve( typed.width() * typed.height() * typed.depth() * sizeof( T ) );

  for( size_t row = 0; row < typed.height(); ++row )
  {
    for( size_t column = 0; column < typed.width(); ++column )
    {
      for( size_t plane = 0; plane < typed.depth(); ++plane )
      {
        append< T >( out, typed( column, row, plane ) );
      }
    }
  }

  return out;
}

} // namespace

// ----------------------------------------------------------------------------
void
write( std::string const& filename, kv::image const& image )
{
  auto const trait = image.pixel_traits();

  if( trait.type != kv::image_pixel_traits::UNSIGNED ||
      ( trait.num_bytes != 1 && trait.num_bytes != 2 ) )
  {
    throw std::runtime_error(
      "TIFF here writes 8 and 16 bit unsigned images only; convert first, "
      "so that whoever loses the range is the one that knows what it meant" );
  }

  auto const samples = static_cast< uint32_t >( image.depth() );

  if( samples != 1 && samples != 3 && samples != 4 )
  {
    std::ostringstream reason;
    reason << "TIFF here writes 1, 3 or 4 samples per pixel, not " << samples;
    throw std::runtime_error( reason.str() );
  }

  auto const pixels = ( trait.num_bytes == 1 )
    ? from_image< uint8_t >( image )
    : from_image< uint16_t >( image );

  // Header, then the pixels, then the IFD: writing the pixels first means
  // their offset is known before the IFD that points at them is built.
  constexpr uint32_t HEADER = 8;
  auto const pixels_at = HEADER;
  auto const ifd_at = static_cast< uint32_t >( HEADER + pixels.size() );

  // The photometric tag and, for 3 and 4 samples, an ExtraSamples for alpha
  uint16_t const entries = ( samples == 4 ) ? 11 : 10;

  std::vector< uint8_t > out;
  out.reserve( ifd_at + 2 + size_t( entries ) * 12 + 4 + samples * 2 );

  // Little endian, which is what every machine VIAME builds for is
  out.push_back( 'I' );
  out.push_back( 'I' );
  append< uint16_t >( out, 42 );
  append< uint32_t >( out, ifd_at );

  out.insert( out.end(), pixels.begin(), pixels.end() );

  // BitsPerSample needs its own array when there is more than one sample
  auto const bits_at =
    static_cast< uint32_t >( ifd_at + 2 + size_t( entries ) * 12 + 4 );
  auto const bits_value = ( samples == 1 )
    ? static_cast< uint32_t >( trait.num_bytes * 8 )
    : bits_at;

  append< uint16_t >( out, entries );

  append_entry( out, TAG_IMAGE_WIDTH, 4, 1,
                static_cast< uint32_t >( image.width() ) );
  append_entry( out, TAG_IMAGE_LENGTH, 4, 1,
                static_cast< uint32_t >( image.height() ) );
  append_entry( out, TAG_BITS_PER_SAMPLE, 3, samples, bits_value );
  append_entry( out, TAG_COMPRESSION, 3, 1, COMPRESSION_NONE );
  append_entry( out, TAG_PHOTOMETRIC, 3, 1, samples == 1 ? 1u : 2u );
  append_entry( out, TAG_STRIP_OFFSETS, 4, 1, pixels_at );
  append_entry( out, TAG_SAMPLES_PER_PIXEL, 3, 1, samples );
  append_entry( out, TAG_ROWS_PER_STRIP, 4, 1,
                static_cast< uint32_t >( image.height() ) );
  append_entry( out, TAG_STRIP_BYTE_COUNTS, 4, 1,
                static_cast< uint32_t >( pixels.size() ) );
  append_entry( out, TAG_PLANAR_CONFIG, 3, 1, 1 );

  if( samples == 4 )
  {
    // 2: unassociated alpha, which is what an RGBA image from a decoder is
    append_entry( out, 338, 3, 1, 2 );
  }

  append< uint32_t >( out, 0 );   // no second IFD

  if( samples > 1 )
  {
    for( uint32_t sample = 0; sample < samples; ++sample )
    {
      append< uint16_t >( out, static_cast< uint16_t >( trait.num_bytes * 8 ) );
    }
  }

  std::ofstream stream( filename, std::ios::binary );

  if( !stream )
  {
    throw std::runtime_error( "cannot write " + filename );
  }

  stream.write( reinterpret_cast< char const* >( out.data() ),
                static_cast< std::streamsize >( out.size() ) );

  if( !stream )
  {
    throw std::runtime_error( "failed writing " + filename );
  }
}

} // end namespace tiff

} // end namespace codecs

} // end namespace viame
