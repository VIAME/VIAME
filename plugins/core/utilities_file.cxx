/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "utilities_file.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <iostream>
#include <cctype>
#include <cstring>
#include <ctime>
#include <vector>

#ifdef VIAME_ENABLE_ZLIB
#include <zlib.h>
#endif

#if WIN32 || ( __cplusplus >= 201703L && __has_include(<filesystem>) )
  #include <filesystem>
  namespace filesystem = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace filesystem = std::experimental::filesystem;
#endif

namespace viame {

// =============================================================================
// Filesystem utilities
// =============================================================================

bool does_file_exist( const std::string& location )
{
  return filesystem::exists( location ) &&
         !filesystem::is_directory( location );
}

bool does_folder_exist( const std::string& location )
{
  return filesystem::exists( location ) &&
         filesystem::is_directory( location );
}

bool list_all_subfolders( const std::string& location,
                          std::vector< std::string >& subfolders )
{
  subfolders.clear();

  if( !does_folder_exist( location ) )
  {
    return false;
  }

  filesystem::path dir( location );

  for( filesystem::directory_iterator dir_iter( dir );
       dir_iter != filesystem::directory_iterator();
       ++dir_iter )
  {
    if( filesystem::is_directory( *dir_iter ) )
    {
      subfolders.push_back( dir_iter->path().string() );
    }
  }

  return true;
}

bool list_files_in_folder( std::string location,
                           std::vector< std::string >& filepaths,
                           bool search_subfolders,
                           std::vector< std::string > extensions )
{
  filepaths.clear();

  if( !does_folder_exist( location ) )
  {
    return false;
  }

#ifdef WIN32
  if( location.back() != '\\' )
  {
    location = location + "\\";
  }
#else
  if( location.back() != '/' )
  {
    location = location + "/";
  }
#endif

  filesystem::path dir( location );

  for( filesystem::directory_iterator file_iter( dir );
       file_iter != filesystem::directory_iterator();
       ++file_iter )
  {
    if( filesystem::is_regular_file( *file_iter ) )
    {
      if( extensions.empty() )
      {
        filepaths.push_back( file_iter->path().string() );
      }
      else
      {
        for( unsigned i = 0; i < extensions.size(); i++ )
        {
          if( file_iter->path().extension() == extensions[i] )
          {
            filepaths.push_back( file_iter->path().string() );
            break;
          }
        }
      }
    }
    else if( filesystem::is_directory( *file_iter ) && search_subfolders )
    {
      std::vector< std::string > subfiles;
      list_files_in_folder( file_iter->path().string(),
        subfiles, search_subfolders, extensions );

      filepaths.insert( filepaths.end(), subfiles.begin(), subfiles.end() );
    }
  }

  return true;
}

bool create_folder( const std::string& location )
{
  filesystem::path dir( location );

  if( !filesystem::exists( dir ) )
  {
    return filesystem::create_directories( dir );
  }

  return false;
}

bool folder_contains_less_than_n_files( const std::string& folder, unsigned n )
{
  auto dir = filesystem::directory_iterator( folder );
  unsigned count = 0;

  for( auto i : dir )
  {
    (void)i; // Suppress unused variable warning
    count++;

    if( count >= n )
    {
      return false;
    }
  }

  return true;
}

// =============================================================================
// Path manipulation utilities
// =============================================================================

std::string append_path( const std::string& p1, const std::string& p2 )
{
  return p1 + "/" + p2;
}

std::string get_filename_no_path( const std::string& path )
{
  return filesystem::path( path ).filename().string();
}

std::string get_filename_with_last_path( const std::string& path )
{
  return append_path( filesystem::path( path ).parent_path().filename().string(),
                      filesystem::path( path ).filename().string() );
}

std::string replace_ext_with( const std::string& file_name, const std::string& ext )
{
  return file_name.substr( 0, file_name.find_last_of( '.' ) ) + ext;
}

std::string add_ext_unto( const std::string& path, const std::string& ext )
{
  if( !path.empty() && ( path.back() == '/' || path.back() == '\\' ) )
  {
    return path.substr( 0, path.size() - 1 ) + ext;
  }

  return path + ext;
}

std::string add_aux_ext( const std::string& file_name, unsigned id )
{
  std::size_t last_index = file_name.find_last_of( "." );
  std::string file_name_no_ext = file_name.substr( 0, last_index );
  std::string aux_addition = "_aux";

  if( id > 1 )
  {
    aux_addition += std::to_string( id );
  }

  return file_name_no_ext + aux_addition + file_name.substr( last_index );
}

bool ends_with_extension( const std::string& str, const std::string& ext )
{
  if( str.length() >= ext.length() )
  {
    return( 0 == str.compare( str.length() - ext.length(),
                              ext.length(), ext ) );
  }
  else
  {
    return false;
  }
}

bool ends_with_extension( const std::string& str,
                          const std::vector< std::string >& exts )
{
  for( const auto& ext : exts )
  {
    if( ends_with_extension( str, ext ) )
    {
      return true;
    }
  }
  return false;
}

std::string get_file_extension( const std::string& path )
{
  size_t dot_pos = path.rfind( '.' );
  if( dot_pos == std::string::npos )
  {
    return "";
  }

  std::string ext = path.substr( dot_pos );

  // Convert to lowercase
  for( auto& c : ext )
  {
    c = static_cast< char >( std::tolower( static_cast< unsigned char >( c ) ) );
  }

  return ext;
}

bool select_file_by_extension_priority(
    const std::vector< std::string >& files,
    const std::vector< std::string >& priority_exts,
    const std::vector< std::string >& allowed_exts,
    std::string& selected,
    std::string& error_msg )
{
  selected.clear();
  error_msg.clear();

  if( files.empty() )
  {
    return true;
  }

  if( files.size() == 1 )
  {
    selected = files[0];
    return true;
  }

  // Group files by extension
  std::map< std::string, std::string > files_by_ext;

  for( const auto& f : files )
  {
    std::string ext = get_file_extension( f );

    if( files_by_ext.count( ext ) )
    {
      error_msg = "multiple files with extension " + ext;
      return false;
    }
    files_by_ext[ ext ] = f;
  }

  // Build set of allowed extensions (lowercase)
  std::unordered_set< std::string > allowed_set;
  for( const auto& ext : allowed_exts )
  {
    std::string ext_lower = ext;
    for( auto& c : ext_lower )
    {
      c = static_cast< char >( std::tolower( static_cast< unsigned char >( c ) ) );
    }
    allowed_set.insert( ext_lower );
  }

  // Select based on priority
  for( const auto& ext : priority_exts )
  {
    std::string ext_lower = ext;
    for( auto& c : ext_lower )
    {
      c = static_cast< char >( std::tolower( static_cast< unsigned char >( c ) ) );
    }

    // Only consider if this extension is allowed
    if( !allowed_set.empty() && !allowed_set.count( ext_lower ) )
    {
      continue;
    }

    if( files_by_ext.count( ext_lower ) )
    {
      selected = files_by_ext[ ext_lower ];
      return true;
    }
  }

  // Fallback to first file if no priority match
  selected = files[0];
  return true;
}

std::string find_associated_file( const std::string& base_path, const std::string& ext )
{
  // Strategy 1: Replace the existing extension
  std::string replaced = replace_ext_with( base_path, ext );
  if( does_file_exist( replaced ) )
  {
    return replaced;
  }

  // Strategy 2: Add extension to the path (handles folders or extensionless files)
  std::string added = add_ext_unto( base_path, ext );
  if( does_file_exist( added ) )
  {
    return added;
  }

  return "";
}

std::string resolve_path_with_link( const std::string& path )
{
  // Check if the path exists directly
  if( does_file_exist( path ) || does_folder_exist( path ) )
  {
    return path;
  }

  // Try resolving as a .lnk file (Windows shortcut)
  std::string lnk_path = path + ".lnk";
  if( does_file_exist( lnk_path ) || does_folder_exist( lnk_path ) )
  {
    try
    {
      return filesystem::canonical( filesystem::path( lnk_path ) ).string();
    }
    catch( ... )
    {
      // If canonical fails, return original path
      return path;
    }
  }

  return path;
}

std::vector< std::string > find_files_in_folder_or_alongside(
    const std::string& folder_path,
    const std::vector< std::string >& extensions )
{
  std::vector< std::string > files;

  // Try listing files in the folder
  list_files_in_folder( folder_path, files, false, extensions );
  std::sort( files.begin(), files.end() );

  // If no files found and we have extensions, try adding first extension to folder path
  if( files.empty() && !extensions.empty() )
  {
    std::string alongside = add_ext_unto( folder_path, extensions[0] );
    if( does_file_exist( alongside ) )
    {
      files.push_back( alongside );
    }
  }

  return files;
}

std::string add_quotes( const std::string& str )
{
  return "\"" + str + "\"";
}

// =============================================================================
// String parsing utilities
// =============================================================================

std::string trim_string( std::string const& str )
{
  size_t start = str.find_first_not_of( " \t\r\n" );
  if( start == std::string::npos )
  {
    return "";
  }
  size_t end = str.find_last_not_of( " \t\r\n" );
  return str.substr( start, end - start + 1 );
}

bool trim_line( std::string const& line, std::string& trimmed, bool skip_comments )
{
  size_t start = line.find_first_not_of( " \t\r\n" );
  if( start == std::string::npos )
  {
    trimmed.clear();
    return false;
  }

  if( skip_comments && line[start] == '#' )
  {
    trimmed.clear();
    return false;
  }

  size_t end = line.find_last_not_of( " \t\r\n" );
  trimmed = line.substr( start, end - start + 1 );
  return true;
}

std::string to_lower( std::string const& str )
{
  std::string result = str;
  for( size_t i = 0; i < result.size(); ++i )
  {
    result[i] = static_cast< char >( std::tolower( static_cast< unsigned char >( result[i] ) ) );
  }
  return result;
}

bool ends_with_ci( std::string const& str, std::string const& suffix )
{
  std::string str_lower = to_lower( str );
  std::string suffix_lower = to_lower( suffix );

  if( suffix_lower.size() > str_lower.size() )
  {
    return false;
  }

  return str_lower.compare( str_lower.size() - suffix_lower.size(),
                            suffix_lower.size(), suffix_lower ) == 0;
}

void string_to_vector( const std::string& str,
                       std::vector< std::string >& out,
                       const std::string delims )
{
  out.clear();

  std::stringstream ss( str );
  std::string line;

  while( std::getline( ss, line ) )
  {
    std::size_t prev = 0, pos;
    while( ( pos = line.find_first_of( delims, prev ) ) != std::string::npos )
    {
      if( pos > prev )
      {
        std::string word = line.substr( prev, pos - prev );
        if( !word.empty() )
        {
          out.push_back( word );
        }
      }
      prev = pos + 1;
    }
    if( prev < line.length() )
    {
      std::string word = line.substr( prev, std::string::npos );
      if( !word.empty() )
      {
        out.push_back( word );
      }
    }
  }
}

void string_to_set( const std::string& str,
                    std::unordered_set< std::string >& out,
                    const std::string delims )
{
  out.clear();
  std::vector< std::string > tmp;
  string_to_vector( str, tmp, delims );
  out.insert( tmp.begin(), tmp.end() );
}

// =============================================================================
// File reading utilities
// =============================================================================

bool file_to_vector( const std::string& fn,
                     std::vector< std::string >& out,
                     bool reset )
{
  std::ifstream in( fn.c_str() );

  if( reset )
  {
    out.clear();
  }

  if( !in )
  {
    std::cerr << "Unable to open " << fn << std::endl;
    return false;
  }

  std::string line;
  while( std::getline( in, line ) )
  {
    if( !line.empty() )
    {
      out.push_back( line );
    }
  }
  return true;
}

bool load_file_list( const std::string& file,
                     std::vector< std::string >& output )
{
  std::ifstream fin( file );
  output.clear();

  if( !fin )
  {
    return false;
  }

  while( !fin.eof() )
  {
    std::string line;
    std::getline( fin, line );
    output.push_back( line );
  }

  fin.close();
  return true;
}

bool file_contains_string( const std::string& file, const std::string& key )
{
  std::ifstream fin( file );
  while( !fin.eof() )
  {
    std::string line;
    std::getline( fin, line );

    if( line.find( key ) != std::string::npos )
    {
      fin.close();
      return true;
    }
  }
  fin.close();
  return false;
}

double get_file_frame_rate( const std::string& file )
{
  std::ifstream fin( file );

  if( !fin )
  {
    return -1.0;
  }

  std::string number;

  for( unsigned i = 0; i < 4 && !fin.eof(); i++ )
  {
    std::string line;
    std::getline( fin, line );

    if( line.size() > 5 && line[0] == '#' )
    {
      for( unsigned p = 0; p < line.size() - 4; p++ )
      {
        if( line.substr( p, 4 ) == "fps:" || line.substr( p, 4 ) == "fps=" )
        {
          for( unsigned l = p + 4; l < line.size(); l++ )
          {
            if( line[l] == ' ' )
            {
              continue;
            }
            else if( std::isdigit( line[l] ) || line[l] == '.' )
            {
              number = number + line[l];
            }
            else
            {
              break;
            }
          }
        }
      }
    }
  }

  fin.close();

  if( number.empty() )
  {
    return -1.0;
  }

  return std::stof( number );
}

// =============================================================================
// Template processing utilities
// =============================================================================

bool replace_keywords_in_template_file(
    const std::string& input_file,
    const std::string& output_file,
    const std::map< std::string, std::string >& replacements )
{
  std::ifstream fin( input_file );
  if( !fin )
  {
    std::cerr << "Unable to open template file: " << input_file << std::endl;
    return false;
  }

  std::stringstream buffer;
  buffer << fin.rdbuf();
  fin.close();

  std::string content = buffer.str();

  for( const auto& pair : replacements )
  {
    const std::string& keyword = pair.first;
    const std::string& value = pair.second;

    size_t pos = 0;
    while( ( pos = content.find( keyword, pos ) ) != std::string::npos )
    {
      content.replace( pos, keyword.length(), value );
      pos += value.length();
    }
  }

  std::ofstream fout( output_file );
  if( !fout )
  {
    std::cerr << "Unable to write output file: " << output_file << std::endl;
    return false;
  }

  fout << content;
  fout.close();

  return true;
}

bool copy_file( const std::string& source, const std::string& destination )
{
  std::ifstream fin( source, std::ios::binary );
  if( !fin )
  {
    std::cerr << "Unable to open source file: " << source << std::endl;
    return false;
  }

  std::ofstream fout( destination, std::ios::binary );
  if( !fout )
  {
    std::cerr << "Unable to create destination file: " << destination << std::endl;
    fin.close();
    return false;
  }

  fout << fin.rdbuf();

  fin.close();
  fout.close();

  return true;
}

bool copy_folder( const std::string& source, const std::string& destination )
{
  if( !does_folder_exist( source ) )
  {
    std::cerr << "Source folder does not exist: " << source << std::endl;
    return false;
  }

  // Create destination folder
  if( !create_folder( destination ) )
  {
    std::cerr << "Failed to create destination folder: " << destination << std::endl;
    return false;
  }

  bool all_success = true;

  try
  {
    for( const auto& entry : filesystem::recursive_directory_iterator( source ) )
    {
      const auto& path = entry.path();
      auto relative_path = filesystem::relative( path, source );
      auto dest_path = filesystem::path( destination ) / relative_path;

      if( filesystem::is_directory( path ) )
      {
        filesystem::create_directories( dest_path );
      }
      else if( filesystem::is_regular_file( path ) )
      {
        // Check if destination path is too long (common filesystem limit)
        if( dest_path.string().length() > 250 )
        {
          // Skip files with paths that are too long
          continue;
        }

        filesystem::create_directories( dest_path.parent_path() );

        std::error_code ec;
        filesystem::copy_file( path, dest_path,
          filesystem::copy_options::overwrite_existing, ec );

        if( ec )
        {
          all_success = false;
        }
      }
    }
  }
  catch( const std::exception& e )
  {
    std::cerr << "Error copying folder: " << e.what() << std::endl;
    return false;
  }

  return all_success;
}

bool replace_keywords_in_template_to_string(
    const std::string& input_file,
    const std::map< std::string, std::string >& replacements,
    std::string& result )
{
  std::ifstream fin( input_file );
  if( !fin )
  {
    std::cerr << "Unable to open template file: " << input_file << std::endl;
    return false;
  }

  std::stringstream buffer;
  buffer << fin.rdbuf();
  fin.close();

  result = buffer.str();

  for( const auto& pair : replacements )
  {
    const std::string& keyword = pair.first;
    const std::string& value = pair.second;

    size_t pos = 0;
    while( ( pos = result.find( keyword, pos ) ) != std::string::npos )
    {
      result.replace( pos, keyword.length(), value );
      pos += value.length();
    }
  }

  return true;
}

// =============================================================================
// Zip file utilities
// =============================================================================

#ifdef VIAME_ENABLE_ZLIB

namespace {

// Helper to write little-endian values
void write_le16( std::ostream& out, uint16_t val )
{
  out.put( static_cast< char >( val & 0xFF ) );
  out.put( static_cast< char >( ( val >> 8 ) & 0xFF ) );
}

void write_le32( std::ostream& out, uint32_t val )
{
  out.put( static_cast< char >( val & 0xFF ) );
  out.put( static_cast< char >( ( val >> 8 ) & 0xFF ) );
  out.put( static_cast< char >( ( val >> 16 ) & 0xFF ) );
  out.put( static_cast< char >( ( val >> 24 ) & 0xFF ) );
}

// Convert time_t to DOS date/time format
void time_to_dos( time_t t, uint16_t& dos_date, uint16_t& dos_time )
{
  struct tm* lt = localtime( &t );
  if( !lt )
  {
    dos_date = 0;
    dos_time = 0;
    return;
  }

  dos_time = static_cast< uint16_t >(
    ( lt->tm_sec / 2 ) |
    ( lt->tm_min << 5 ) |
    ( lt->tm_hour << 11 ) );

  dos_date = static_cast< uint16_t >(
    lt->tm_mday |
    ( ( lt->tm_mon + 1 ) << 5 ) |
    ( ( lt->tm_year - 80 ) << 9 ) );
}

// Compress data using zlib deflate
bool compress_data( const std::vector< char >& input,
                    std::vector< char >& output,
                    uint32_t& crc )
{
  // Calculate CRC32 of uncompressed data
  crc = crc32( 0L, Z_NULL, 0 );
  crc = crc32( crc,
               reinterpret_cast< const Bytef* >( input.data() ),
               static_cast< uInt >( input.size() ) );

  // Compress using deflate (raw deflate, not zlib or gzip wrapper)
  z_stream strm;
  std::memset( &strm, 0, sizeof( strm ) );

  // Use raw deflate (-MAX_WBITS) for ZIP format
  if( deflateInit2( &strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                    -MAX_WBITS, 8, Z_DEFAULT_STRATEGY ) != Z_OK )
  {
    return false;
  }

  // Allocate output buffer (worst case: slightly larger than input)
  output.resize( deflateBound( &strm, static_cast< uLong >( input.size() ) ) );

  strm.next_in = reinterpret_cast< Bytef* >(
    const_cast< char* >( input.data() ) );
  strm.avail_in = static_cast< uInt >( input.size() );
  strm.next_out = reinterpret_cast< Bytef* >( output.data() );
  strm.avail_out = static_cast< uInt >( output.size() );

  int ret = deflate( &strm, Z_FINISH );
  deflateEnd( &strm );

  if( ret != Z_STREAM_END )
  {
    return false;
  }

  output.resize( strm.total_out );
  return true;
}

struct ZipEntry
{
  std::string name;
  uint32_t crc;
  uint32_t compressed_size;
  uint32_t uncompressed_size;
  uint16_t dos_time;
  uint16_t dos_date;
  uint32_t local_header_offset;
};

} // anonymous namespace

#endif // VIAME_ENABLE_ZLIB

bool create_zip_file(
    const std::string& zip_path,
    const std::map< std::string, std::string >& files_to_add,
    const std::map< std::string, std::string >& string_contents )
{
#ifdef VIAME_ENABLE_ZLIB
  std::ofstream zip_out( zip_path, std::ios::binary );
  if( !zip_out )
  {
    std::cerr << "Unable to create zip file: " << zip_path << std::endl;
    return false;
  }

  std::vector< ZipEntry > entries;
  time_t now = time( nullptr );
  uint16_t dos_date, dos_time;
  time_to_dos( now, dos_date, dos_time );

  // Write local file headers and data for file entries
  for( const auto& pair : files_to_add )
  {
    const std::string& entry_name = pair.first;
    const std::string& source_path = pair.second;

    // Read source file
    std::ifstream fin( source_path, std::ios::binary | std::ios::ate );
    if( !fin )
    {
      std::cerr << "Warning: Unable to read file for zip: " << source_path << std::endl;
      continue;
    }

    std::streamsize file_size = fin.tellg();
    fin.seekg( 0, std::ios::beg );

    std::vector< char > file_data( static_cast< size_t >( file_size ) );
    if( !fin.read( file_data.data(), file_size ) )
    {
      std::cerr << "Warning: Unable to read file data: " << source_path << std::endl;
      continue;
    }
    fin.close();

    // Compress the data
    std::vector< char > compressed_data;
    uint32_t crc;
    if( !compress_data( file_data, compressed_data, crc ) )
    {
      std::cerr << "Warning: Unable to compress file: " << source_path << std::endl;
      continue;
    }

    ZipEntry entry;
    entry.name = entry_name;
    entry.crc = crc;
    entry.compressed_size = static_cast< uint32_t >( compressed_data.size() );
    entry.uncompressed_size = static_cast< uint32_t >( file_data.size() );
    entry.dos_time = dos_time;
    entry.dos_date = dos_date;
    entry.local_header_offset = static_cast< uint32_t >( zip_out.tellp() );

    // Write local file header
    write_le32( zip_out, 0x04034b50 );  // Local file header signature
    write_le16( zip_out, 20 );           // Version needed to extract (2.0)
    write_le16( zip_out, 0 );            // General purpose bit flag
    write_le16( zip_out, 8 );            // Compression method (deflate)
    write_le16( zip_out, entry.dos_time );
    write_le16( zip_out, entry.dos_date );
    write_le32( zip_out, entry.crc );
    write_le32( zip_out, entry.compressed_size );
    write_le32( zip_out, entry.uncompressed_size );
    write_le16( zip_out, static_cast< uint16_t >( entry_name.size() ) );
    write_le16( zip_out, 0 );            // Extra field length

    // Write file name
    zip_out.write( entry_name.c_str(), entry_name.size() );

    // Write compressed data
    zip_out.write( compressed_data.data(),
                   static_cast< std::streamsize >( compressed_data.size() ) );

    entries.push_back( entry );
  }

  // Write local file headers and data for string content entries
  for( const auto& pair : string_contents )
  {
    const std::string& entry_name = pair.first;
    const std::string& content = pair.second;

    std::vector< char > file_data( content.begin(), content.end() );

    // Compress the data
    std::vector< char > compressed_data;
    uint32_t crc;
    if( !compress_data( file_data, compressed_data, crc ) )
    {
      std::cerr << "Warning: Unable to compress content for: " << entry_name << std::endl;
      continue;
    }

    ZipEntry entry;
    entry.name = entry_name;
    entry.crc = crc;
    entry.compressed_size = static_cast< uint32_t >( compressed_data.size() );
    entry.uncompressed_size = static_cast< uint32_t >( file_data.size() );
    entry.dos_time = dos_time;
    entry.dos_date = dos_date;
    entry.local_header_offset = static_cast< uint32_t >( zip_out.tellp() );

    // Write local file header
    write_le32( zip_out, 0x04034b50 );  // Local file header signature
    write_le16( zip_out, 20 );           // Version needed to extract (2.0)
    write_le16( zip_out, 0 );            // General purpose bit flag
    write_le16( zip_out, 8 );            // Compression method (deflate)
    write_le16( zip_out, entry.dos_time );
    write_le16( zip_out, entry.dos_date );
    write_le32( zip_out, entry.crc );
    write_le32( zip_out, entry.compressed_size );
    write_le32( zip_out, entry.uncompressed_size );
    write_le16( zip_out, static_cast< uint16_t >( entry_name.size() ) );
    write_le16( zip_out, 0 );            // Extra field length

    // Write file name
    zip_out.write( entry_name.c_str(), entry_name.size() );

    // Write compressed data
    zip_out.write( compressed_data.data(),
                   static_cast< std::streamsize >( compressed_data.size() ) );

    entries.push_back( entry );
  }

  // Record start of central directory
  uint32_t central_dir_offset = static_cast< uint32_t >( zip_out.tellp() );

  // Write central directory
  for( const auto& entry : entries )
  {
    write_le32( zip_out, 0x02014b50 );  // Central file header signature
    write_le16( zip_out, 20 );           // Version made by (2.0)
    write_le16( zip_out, 20 );           // Version needed to extract (2.0)
    write_le16( zip_out, 0 );            // General purpose bit flag
    write_le16( zip_out, 8 );            // Compression method (deflate)
    write_le16( zip_out, entry.dos_time );
    write_le16( zip_out, entry.dos_date );
    write_le32( zip_out, entry.crc );
    write_le32( zip_out, entry.compressed_size );
    write_le32( zip_out, entry.uncompressed_size );
    write_le16( zip_out, static_cast< uint16_t >( entry.name.size() ) );
    write_le16( zip_out, 0 );            // Extra field length
    write_le16( zip_out, 0 );            // File comment length
    write_le16( zip_out, 0 );            // Disk number start
    write_le16( zip_out, 0 );            // Internal file attributes
    write_le32( zip_out, 0 );            // External file attributes
    write_le32( zip_out, entry.local_header_offset );

    // Write file name
    zip_out.write( entry.name.c_str(), entry.name.size() );
  }

  // Calculate central directory size
  uint32_t central_dir_size = static_cast< uint32_t >( zip_out.tellp() ) - central_dir_offset;

  // Write end of central directory record
  write_le32( zip_out, 0x06054b50 );  // End of central dir signature
  write_le16( zip_out, 0 );           // Number of this disk
  write_le16( zip_out, 0 );           // Disk where central dir starts
  write_le16( zip_out, static_cast< uint16_t >( entries.size() ) );
  write_le16( zip_out, static_cast< uint16_t >( entries.size() ) );
  write_le32( zip_out, central_dir_size );
  write_le32( zip_out, central_dir_offset );
  write_le16( zip_out, 0 );           // Comment length

  zip_out.close();

  if( !zip_out )
  {
    std::cerr << "Error writing zip file: " << zip_path << std::endl;
    return false;
  }

  return true;

#else
  std::cerr << "Zip file creation requires ZLIB support" << std::endl;
  return false;
#endif
}

} // end namespace viame
