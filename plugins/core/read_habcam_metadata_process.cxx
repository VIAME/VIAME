/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Read habcam metadata
 */

#include "read_habcam_metadata_process.h"

#include <vital/vital_types.h>

#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <string>
#include <fstream>
#include <sstream>

#include <stdio.h>

#if defined( MSDOS ) || defined( WIN32 )
  #include <fcntl.h>
  #include <io.h>
#endif


namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( scan_length, unsigned, "1000",
  "Number of characters at end of file to scan for metadata." );

// =============================================================================
// Private implementation class
class read_habcam_metadata_process::priv
{
public:
  explicit priv( read_habcam_metadata_process* parent );
  ~priv();

  // Configuration settings
  int m_scan_length;

  // Other variables
  read_habcam_metadata_process* parent;
};


// Read JPG comments from file
int get_jpeg_comments( FILE *f, std::string& s )
{
  int c, m;
  unsigned ss;
  s = "";
#if defined( MSDOS ) || defined( WIN32 )
  setmode( fileno(f), O_BINARY );
#endif
  if( ferror( f ) ) return -1;
  /* A typical JPEG file has markers in these order:
   *   d8 e0_JFIF e1 e1 e2 db db fe fe c0 c4 c4 c4 c4 da d9.
   *   The first fe marker (COM, comment) was near offset 30000.
   * A typical JPEG file after filtering through jpegtran:
   *   d8 e0_JFIF fe fe db db c0 c4 c4 c4 c4 da d9.
   *   The first fe marker (COM, comment) was at offset 20.
   */
  if( (c = getc(f) ) < 0 ) return -2;  /* Truncated (empty). */
  if( c != 0xff) return -3;
  if( ( c = getc(f) ) < 0 ) return -2;  /* Truncated. */
  if( c != 0xd8) return -3;  /* Not a JPEG file, SOI expected. */

  for( ;; )
  {
    /* printf("@%ld\n", ftell(f)); */
    if( (c = getc(f) ) < 0 ) return -2;  /* Truncated. */
    if( c != 0xff ) return -3;  /* Not a JPEG file, marker expected. */
    if( (m = getc(f) ) < 0 ) return -2;  /* Truncated. */
    while( m == 0xff ) {  /* Padding. */
      if( (m = getc(f) ) < 0 ) return -2;  /* Truncated. */
    }
    if( m == 0xd8 ) return -4;  /* SOI unexpected. */
    if( m == 0xd9 ) break;  /* EOI. */
    if( m == 0xda ) break;  /* SOS. Would need special escaping to process. */
    /* printf("MARKER 0x%02x\n", m); */
    if( (c = getc(f)) < 0 ) return -2;  /* Truncated. */
    ss = (c + 0U) << 8;
    if( (c = getc(f)) < 0 ) return -2;  /* Truncated. */
    ss += c;
    if( ss < 2 ) return -5;  /* Segment too short. */
    for ( ss -= 2; ss > 0; --ss) {
      if( ( c = getc(f) ) < 0 ) return -2;  /* Truncated. */
      if( m == 0xfe ) s+=c;  /* Emit comment char. */
    }
    if( m == 0xfe ) s+='\n';  /* End of comment. */
  }
  return 0;
}


bool ends_with( std::string const &input, std::string const &ending )
{
  if( input.length() >= ending.length() )
  {
    return ( 0 == input.compare(input.length() - ending.length(), ending.length(), ending) );
  }
  else
  {
    return false;
  }
}


bool is_tiff( std::string const &file_name )
{
  return ends_with( file_name, ".tif" ) || ends_with( file_name, ".tiff" ) ||
         ends_with( file_name, ".TIF" ) || ends_with( file_name, ".TIFF" );
}


void tokenize( const std::string ascii_snippet, std::vector< std::string >& tokens )
{
  std::stringstream ss( ascii_snippet );
  std::string line;
  const std::string delims = "\n\t\v ,";
  tokens.clear();

  while( std::getline( ss, line ) )
  {
    std::size_t prev = 0, pos;
    while( ( pos = line.find_first_of( delims, prev ) ) != std::string::npos )
    {
      if( pos > prev )
      {
        std::string token = line.substr( prev, pos-prev );
        if( !token.empty() )
        {
          tokens.push_back( token );
        }
      }
      prev = pos + 1;
    }
    if( prev < line.length() )
    {
      std::string token = line.substr( prev, std::string::npos );
      if( !token.empty() )
      {
        tokens.push_back( token );
      }
    }
  }
}


// -----------------------------------------------------------------------------
read_habcam_metadata_process::priv
::priv( read_habcam_metadata_process* ptr )
  : m_scan_length( 1000 )
  , parent( ptr )
{
}


read_habcam_metadata_process::priv
::~priv()
{
}


// =============================================================================
read_habcam_metadata_process
::read_habcam_metadata_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new read_habcam_metadata_process::priv( this ) )
{
  make_ports();
  make_config();
}


read_habcam_metadata_process
::~read_habcam_metadata_process()
{
}


// -----------------------------------------------------------------------------
void
read_habcam_metadata_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( file_name, required );

  // -- outputs --
  declare_output_port_using_trait( metadata, optional );
  declare_output_port_using_trait( gsd, optional );
}

// -----------------------------------------------------------------------------
void
read_habcam_metadata_process
::make_config()
{
  declare_config_using_trait( scan_length );
}

// -----------------------------------------------------------------------------
void
read_habcam_metadata_process
::_configure()
{
  d->m_scan_length = config_value_using_trait( scan_length );
}

// -----------------------------------------------------------------------------
void
read_habcam_metadata_process
::_step()
{
  std::string file_name = grab_from_port_using_trait( file_name );

  kwiver::vital::metadata_vector output_md_vec;
  double output_gsd = -1.0;

  std::shared_ptr< kwiver::vital::metadata > output_md =
    std::make_shared< kwiver::vital::metadata >();

  if( is_tiff( file_name ) )
  {
    std::ifstream fin( file_name.c_str() );

    if( !fin )
    {
      throw std::runtime_error( "Unable to load: " + file_name );
    }

    std::stringstream buffer;
    fin.seekg( -d->m_scan_length, std::ios_base::end );
    buffer << fin.rdbuf();
    std::string ascii_snippet = buffer.str();
    fin.close();

    auto meta_start = ascii_snippet.find( "pixelformat=" );

    if( meta_start == std::string::npos )
    {
      push_to_port_using_trait( metadata, output_md_vec );
      push_to_port_using_trait( gsd, output_gsd );
      return;
    }

    ascii_snippet = ascii_snippet.substr( meta_start );

    std::vector< std::string > tokens;
    tokenize( ascii_snippet, tokens );

  #define CHECK_FIELD( STR, METAID )                              \
    {                                                             \
      std::string field = STR ;                                   \
      std::size_t pos = field.size() + 1;                         \
      if( token.substr( 0, pos ) == field + "=" )                 \
      {                                                           \
        try                                                       \
        {                                                         \
          std::string str_val = token.substr( pos );              \
                                                                  \
          if( str_val != "-99.99" )                               \
          {                                                       \
            output_md->add< METAID >( std::stod( str_val ) );     \
          }                                                       \
        }                                                         \
        catch( ... )                                              \
        {                                                         \
        }                                                         \
      }                                                           \
    }

    for( std::string token : tokens )
    {
      CHECK_FIELD( "hdg", kwiver::vital::VITAL_META_SENSOR_YAW_ANGLE );
      CHECK_FIELD( "pitch", kwiver::vital::VITAL_META_SENSOR_PITCH_ANGLE );
      CHECK_FIELD( "roll", kwiver::vital::VITAL_META_SENSOR_ROLL_ANGLE );
      CHECK_FIELD( "alt0", kwiver::vital::VITAL_META_SENSOR_ALTITUDE );
      CHECK_FIELD( "alt1", kwiver::vital::VITAL_META_SENSOR_ALTITUDE );
    }
  }
  else
  {
    std::string ascii_snippet;
    FILE *fin = fopen( file_name.c_str(), "r" );

    if( !fin )
    {
      throw std::runtime_error( "Unable to load: " + file_name );
    }
    if( get_jpeg_comments( fin, ascii_snippet ) ) // Note: returns 0 on success
    {
      throw std::runtime_error( "Unable to read metadata from: " + file_name );
    }

    std::vector< std::string > tokens;
    tokenize( ascii_snippet, tokens );
    fclose( fin );

    int image_id_ind = -1;

    for( int i = 0; i < static_cast<int>( tokens.size() ); ++i )
    {
      if( is_tiff( tokens[i] ) )
      {
        image_id_ind = i;
        break;
      }
    }

    if( image_id_ind < 0 )
    {
      throw std::runtime_error( "TIF image file string not found in: " + file_name );
    }

    if( image_id_ind + 6 <= static_cast<int>( tokens.size() ) )
    {
      output_md->add< kwiver::vital::VITAL_META_SENSOR_ALTITUDE >(
        std::stod( tokens[ image_id_ind + 2 ] ) );
      output_md->add< kwiver::vital::VITAL_META_SENSOR_YAW_ANGLE >(
        std::stod( tokens[ image_id_ind + 3 ] ) );
      output_md->add< kwiver::vital::VITAL_META_SENSOR_PITCH_ANGLE >(
        std::stod( tokens[ image_id_ind + 4 ] ) );
      output_md->add< kwiver::vital::VITAL_META_SENSOR_ROLL_ANGLE >(
        std::stod( tokens[ image_id_ind + 5 ] ) );
    }
    else
    {
      throw std::runtime_error( "Insufficient metadata fields in " + file_name );
    }
  }

  output_md_vec.push_back( output_md );

  push_to_port_using_trait( metadata, output_md_vec );
  push_to_port_using_trait( gsd, output_gsd );
}

} // end namespace core

} // end namespace viame
