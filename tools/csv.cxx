/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "csv.h"

#include <kwiversys/SystemTools.hxx>

#include <vital/logger/logger.h>

#include <vector>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>

#if WIN32 || ( __cplusplus >= 201703L && __has_include(<filesystem>) )
  #include <filesystem>
  namespace filesystem = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace filesystem = std::experimental::filesystem;
#endif

namespace kv = kwiver::vital;

namespace viame {
namespace tools {

// =======================================================================================
// Helper functions

static double parse_fps( const std::string& line )
{
  size_t pos = line.find( "fps" );
  if( pos == std::string::npos )
  {
    return -1;
  }

  std::string output;
  bool found = false;
  bool found_period = false;

  for( size_t i = pos; i < line.size(); ++i )
  {
    if( std::isdigit( line[i] ) )
    {
      output += line[i];
      found = true;
    }
    else if( found && line[i] == '.' && !found_period )
    {
      output += line[i];
      found_period = true;
    }
    else if( !found )
    {
      continue;
    }
    else
    {
      break;
    }
  }

  return output.empty() ? -1 : std::stod( output );
}

static std::vector< std::string > split_string( const std::string& str, char delimiter )
{
  std::vector< std::string > tokens;
  std::stringstream ss( str );
  std::string token;

  while( std::getline( ss, token, delimiter ) )
  {
    tokens.push_back( token );
  }

  return tokens;
}

static std::string trim_string( const std::string& str )
{
  size_t start = str.find_first_not_of( " \t\r\n" );
  if( start == std::string::npos )
  {
    return "";
  }
  size_t end = str.find_last_not_of( " \t\r\n" );
  return str.substr( start, end - start + 1 );
}

static std::string join_strings( const std::vector< std::string >& vec, const std::string& delimiter )
{
  std::string result;
  for( size_t i = 0; i < vec.size(); ++i )
  {
    if( i > 0 )
    {
      result += delimiter;
    }
    result += vec[i];
  }
  return result;
}

static bool has_uppercase( const std::string& str )
{
  for( char c : str )
  {
    if( std::isupper( c ) )
    {
      return true;
    }
  }
  return false;
}

static std::vector< std::string > glob_files( const std::string& pattern )
{
  std::vector< std::string > result;

  // Simple glob implementation for *.csv patterns
  if( pattern.find( '*' ) != std::string::npos )
  {
    filesystem::path p( pattern );
    filesystem::path dir = p.parent_path();
    std::string filename_pattern = p.filename().string();

    if( dir.empty() )
    {
      dir = ".";
    }

    // Replace * with regex-like matching
    std::string prefix, suffix;
    size_t star_pos = filename_pattern.find( '*' );
    if( star_pos != std::string::npos )
    {
      prefix = filename_pattern.substr( 0, star_pos );
      suffix = filename_pattern.substr( star_pos + 1 );
    }

    if( filesystem::exists( dir ) && filesystem::is_directory( dir ) )
    {
      for( const auto& entry : filesystem::directory_iterator( dir ) )
      {
        if( entry.is_regular_file() )
        {
          std::string fname = entry.path().filename().string();
          bool matches = true;

          if( !prefix.empty() && fname.find( prefix ) != 0 )
          {
            matches = false;
          }
          if( !suffix.empty() )
          {
            if( fname.size() < suffix.size() ||
                fname.substr( fname.size() - suffix.size() ) != suffix )
            {
              matches = false;
            }
          }

          if( matches )
          {
            result.push_back( entry.path().string() );
          }
        }
      }
    }
  }

  std::sort( result.begin(), result.end() );
  return result;
}

static void collect_csv_files_recursive( const filesystem::path& dir,
                                         std::vector< std::string >& files )
{
  for( const auto& entry : filesystem::recursive_directory_iterator( dir ) )
  {
    if( entry.is_regular_file() )
    {
      std::string ext = entry.path().extension().string();
      std::transform( ext.begin(), ext.end(), ext.begin(), ::tolower );
      if( ext == ".csv" )
      {
        files.push_back( entry.path().string() );
      }
    }
  }
}

// =======================================================================================
csv_applet
::csv_applet()
{
}

// =======================================================================================
void
csv_applet
::add_command_options()
{
  m_cmd_options->add_options()
    ( "h,help", "Display usage information",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "i,input", "Input file or glob pattern to process",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "decrease-fid", "Decrease frame IDs in files by 1",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "increase-fid", "Increase frame IDs in files by 1",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "assign-uid", "Assign unique detection IDs to all entries in volume",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "filter-single", "Filter single state tracks",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "print-types", "Print unique list of target types",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "caps-only", "Only print types with capitalized letters in them",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "track-count", "Print total number of tracks",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "counts-per-frame", "Print total number of detections per frame",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "average-box-size", "Print average box size per type",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "conf-threshold", "Confidence threshold",
      ::cxxopts::value< double >()->default_value( "-1.0" ), "value" )
    ( "type-threshold", "Type confidence threshold",
      ::cxxopts::value< double >()->default_value( "-1.0" ), "value" )
    ( "print-filtered", "Print out tracks that were filtered out",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "print-single", "Print out video sequences only containing single states",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "lower-fid", "Lower FID if adjusting FIDs to be within some range",
      ::cxxopts::value< int >()->default_value( "0" ), "value" )
    ( "upper-fid", "Upper FID if adjusting FIDs to be within some range",
      ::cxxopts::value< int >()->default_value( "0" ), "value" )
    ( "replace-file", "If set, replace all types in this file given their synonyms",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "print-fps", "Print FPS in input files",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "comp-file", "If set, generate a comparison file contrasting types in all inputs",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ;
}

// =======================================================================================
int
csv_applet
::run()
{
  // Get logger
  kv::logger_handle_t logger = kv::get_logger( "viame.tools.csv" );

  // Get command line arguments
  auto& cmd_args = command_args();

  // Print help
  if( cmd_args[ "help" ].as< bool >() )
  {
    std::cout << "Usage: viame csv [options]\n"
              << "\nPerform filtering and analysis actions on VIAME CSV files.\n"
              << m_cmd_options->help() << std::endl;
    return EXIT_SUCCESS;
  }

  // Extract options
  std::string opt_input = cmd_args[ "input" ].as< std::string >();
  bool opt_decrease_fid = cmd_args[ "decrease-fid" ].as< bool >();
  bool opt_increase_fid = cmd_args[ "increase-fid" ].as< bool >();
  bool opt_assign_uid = cmd_args[ "assign-uid" ].as< bool >();
  bool opt_filter_single = cmd_args[ "filter-single" ].as< bool >();
  bool opt_print_types = cmd_args[ "print-types" ].as< bool >();
  bool opt_caps_only = cmd_args[ "caps-only" ].as< bool >();
  bool opt_track_count = cmd_args[ "track-count" ].as< bool >();
  bool opt_counts_per_frame = cmd_args[ "counts-per-frame" ].as< bool >();
  bool opt_average_box_size = cmd_args[ "average-box-size" ].as< bool >();
  double opt_conf_threshold = cmd_args[ "conf-threshold" ].as< double >();
  double opt_type_threshold = cmd_args[ "type-threshold" ].as< double >();
  bool opt_print_filtered = cmd_args[ "print-filtered" ].as< bool >();
  bool opt_print_single = cmd_args[ "print-single" ].as< bool >();
  int opt_lower_fid = cmd_args[ "lower-fid" ].as< int >();
  int opt_upper_fid = cmd_args[ "upper-fid" ].as< int >();
  std::string opt_replace_file = cmd_args[ "replace-file" ].as< std::string >();
  bool opt_print_fps = cmd_args[ "print-fps" ].as< bool >();
  std::string opt_comp_file = cmd_args[ "comp-file" ].as< std::string >();

  // Validate input
  if( opt_input.empty() )
  {
    std::cout << "No valid input files provided, exiting." << std::endl;
    return EXIT_SUCCESS;
  }

  // Collect input files
  std::vector< std::string > input_files;

  if( filesystem::is_directory( opt_input ) )
  {
    collect_csv_files_recursive( opt_input, input_files );
  }
  else if( opt_input.find( '*' ) != std::string::npos )
  {
    input_files = glob_files( opt_input );
  }
  else
  {
    input_files.push_back( opt_input );
  }

  // Adjust options based on dependencies
  if( opt_caps_only )
  {
    opt_print_types = true;
  }

  if( opt_print_single )
  {
    opt_track_count = true;
  }

  bool write_output = opt_filter_single || opt_increase_fid ||
    opt_decrease_fid || opt_assign_uid || !opt_replace_file.empty() ||
    opt_lower_fid > 0 || opt_upper_fid > 0;

  // Global counters and maps
  int id_counter = 1;
  std::map< std::string, int > type_counts;
  std::map< std::string, double > type_sizes;
  std::map< std::string, std::map< std::string, std::set< std::string > > > type_ids;
  std::map< std::string, std::string > repl_dict;

  int track_counter = 0;
  int state_counter = 0;

  // Load replacement file if specified
  if( !opt_replace_file.empty() )
  {
    std::ifstream fin( opt_replace_file );
    if( !fin )
    {
      std::cout << "Replace file: " << opt_replace_file << " does not exist" << std::endl;
      return EXIT_FAILURE;
    }

    std::string line;
    while( std::getline( fin, line ) )
    {
      auto parsed = split_string( line, ',' );
      if( parsed.size() >= 2 )
      {
        repl_dict[ trim_string( parsed[0] ) ] = trim_string( parsed[1] );
      }
      else if( !trim_string( line ).empty() )
      {
        std::cout << "Error parsing line: " << line << std::endl;
      }
    }
    fin.close();
  }

  // Process each input file
  for( const auto& input_file : input_files )
  {
    if( !opt_print_single )
    {
      if( opt_counts_per_frame )
      {
        std::cout << "# " << filesystem::path( input_file ).filename().string() << std::endl;
      }
      else if( opt_print_fps )
      {
        std::cout << input_file << ",";
      }
      else
      {
        std::cout << "Processing " << input_file << std::endl;
      }
    }

    std::ifstream fin( input_file );
    if( !fin )
    {
      std::cerr << "Could not open file: " << input_file << std::endl;
      continue;
    }

    std::vector< std::string > output;
    std::map< std::string, std::string > id_mappings;
    std::map< std::string, int > id_states;
    std::set< std::string > unique_ids;
    std::set< std::string > printed_ids;
    std::map< std::string, std::map< std::string, int > > frame_counts;
    std::map< std::string, std::set< std::string > > seq_ids;

    bool contains_track = false;
    double video_fps = 0;
    bool has_non_single = false;

    std::string line;
    while( std::getline( fin, line ) )
    {
      // Handle comment lines
      if( !line.empty() && ( line[0] == '#' || line.substr( 0, 9 ) == "target_id" ) )
      {
        if( opt_print_fps && line.find( "fps" ) != std::string::npos )
        {
          video_fps = parse_fps( line );
        }
        output.push_back( line );
        continue;
      }

      auto parsed_line = split_string( trim_string( line ), ',' );
      if( parsed_line.size() < 2 )
      {
        continue;
      }

      // Apply confidence threshold
      if( opt_conf_threshold > 0 && parsed_line.size() > 7 )
      {
        double conf = std::stod( parsed_line[7] );
        if( conf < opt_conf_threshold )
        {
          if( opt_print_filtered && printed_ids.find( parsed_line[0] ) == printed_ids.end() )
          {
            std::cout << "Id: " << parsed_line[0] << " filtered" << std::endl;
            printed_ids.insert( parsed_line[0] );
          }
          continue;
        }
      }

      // Track counting
      if( opt_track_count )
      {
        state_counter++;
        if( unique_ids.find( parsed_line[0] ) == unique_ids.end() )
        {
          unique_ids.insert( parsed_line[0] );
        }
        else
        {
          contains_track = true;
        }
      }

      // Frame ID adjustment
      if( opt_decrease_fid )
      {
        parsed_line[2] = std::to_string( std::stoi( parsed_line[2] ) - 1 );
      }

      if( opt_increase_fid )
      {
        parsed_line[2] = std::to_string( std::stoi( parsed_line[2] ) + 1 );
      }

      if( opt_lower_fid > 0 )
      {
        int fid = std::stoi( parsed_line[2] );
        if( fid < opt_lower_fid )
        {
          continue;
        }
        parsed_line[2] = std::to_string( fid - opt_lower_fid );
      }

      if( opt_upper_fid > 0 )
      {
        int fid = std::stoi( parsed_line[2] );
        if( fid > opt_upper_fid - opt_lower_fid )
        {
          continue;
        }
      }

      // Filter single state tracks
      if( opt_filter_single )
      {
        if( id_states.find( parsed_line[0] ) == id_states.end() )
        {
          id_states[ parsed_line[0] ] = 1;
        }
        else
        {
          id_states[ parsed_line[0] ]++;
          has_non_single = true;
        }
      }

      // Process type information
      if( parsed_line.size() > 9 )
      {
        std::string top_category;
        double top_score = -100.0;
        int attr_start = -1;

        for( size_t i = 9; i < parsed_line.size(); i += 2 )
        {
          if( parsed_line[i].empty() )
          {
            continue;
          }
          if( parsed_line[i][0] == '(' )
          {
            attr_start = static_cast< int >( i );
            break;
          }
          if( i + 1 < parsed_line.size() )
          {
            double score = std::stod( parsed_line[i + 1] );
            if( score > top_score )
            {
              top_category = parsed_line[i];
              top_score = score;
            }
          }
        }

        // Apply type threshold
        if( opt_type_threshold > 0 )
        {
          if( top_score < opt_type_threshold )
          {
            if( opt_print_filtered && printed_ids.find( parsed_line[0] ) == printed_ids.end() )
            {
              std::cout << "Id: " << parsed_line[0] << " filtered" << std::endl;
              printed_ids.insert( parsed_line[0] );
            }
            continue;
          }
        }

        // Collect type statistics
        if( opt_print_types || opt_average_box_size )
        {
          type_counts[ top_category ]++;

          if( opt_track_count )
          {
            seq_ids[ top_category ].insert( parsed_line[0] );
          }
        }

        // Counts per frame
        if( opt_counts_per_frame )
        {
          frame_counts[ parsed_line[1] ][ top_category ]++;
        }

        // Average box size calculation
        if( opt_average_box_size )
        {
          double box_width = std::stod( parsed_line[5] ) - std::stod( parsed_line[3] );
          double box_height = std::stod( parsed_line[6] ) - std::stod( parsed_line[4] );
          type_sizes[ top_category ] += ( box_width * box_height );
        }

        // Type replacement
        if( !opt_replace_file.empty() )
        {
          std::string new_cat = top_category;
          if( repl_dict.find( top_category ) != repl_dict.end() )
          {
            new_cat = repl_dict[ top_category ];
          }

          parsed_line[9] = new_cat;
          parsed_line[10] = "1.0";

          if( attr_start > 0 )
          {
            int attr_count = static_cast< int >( parsed_line.size() ) - attr_start;
            for( int j = 0; j < attr_count; ++j )
            {
              parsed_line[ 11 + j ] = parsed_line[ attr_start + j ];
            }
            parsed_line.resize( 11 + attr_count );
          }
          else if( parsed_line.size() > 11 )
          {
            parsed_line.resize( 11 );
          }
        }
      }

      // Assign unique IDs
      if( opt_assign_uid )
      {
        if( id_mappings.find( parsed_line[0] ) != id_mappings.end() )
        {
          parsed_line[0] = id_mappings[ parsed_line[0] ];
          has_non_single = true;
        }
        else
        {
          id_mappings[ parsed_line[0] ] = std::to_string( id_counter );
          parsed_line[0] = std::to_string( id_counter );
          id_counter++;
        }
      }

      // Store output line
      if( write_output )
      {
        output.push_back( join_strings( parsed_line, "," ) );
      }
    }

    fin.close();

    // Store sequence IDs for comparison file
    if( !seq_ids.empty() )
    {
      type_ids[ input_file ] = seq_ids;
    }

    // Print FPS
    if( opt_print_fps )
    {
      if( video_fps > 0 )
      {
        std::cout << video_fps << std::endl;
      }
      else
      {
        std::cout << "unlisted" << std::endl;
      }
    }

    // Update track counter
    if( opt_track_count )
    {
      track_counter += static_cast< int >( unique_ids.size() );
    }

    // Check for all single states
    if( ( opt_assign_uid || opt_filter_single ) && !has_non_single )
    {
      std::cout << "Sequence " << input_file << " has all single states" << std::endl;
    }

    // Print single detection sequences
    if( opt_print_single && !contains_track )
    {
      if( unique_ids.empty() )
      {
        std::cout << "Sequence " << input_file << " contains no detections" << std::endl;
      }
      else
      {
        std::cout << "Sequence " << input_file << " contains only detections" << std::endl;
      }
    }

    // Filter single state tracks from output
    if( opt_filter_single )
    {
      std::vector< std::string > filtered_output;
      for( const auto& out_line : output )
      {
        if( out_line.empty() || out_line[0] == '#' )
        {
          filtered_output.push_back( out_line );
          continue;
        }
        auto parts = split_string( out_line, ',' );
        if( !parts.empty() && id_states.find( parts[0] ) != id_states.end() &&
            id_states[ parts[0] ] > 1 )
        {
          filtered_output.push_back( out_line );
        }
      }
      output = filtered_output;
    }

    // Print counts per frame
    if( opt_counts_per_frame )
    {
      for( const auto& frame_pair : frame_counts )
      {
        std::string frame_str = frame_pair.first;
        for( const auto& cls_pair : frame_pair.second )
        {
          frame_str += ", " + cls_pair.first + "=" + std::to_string( cls_pair.second );
        }
        std::cout << frame_str << std::endl;
      }
    }

    // Write output file
    if( write_output )
    {
      std::ofstream fout( input_file );
      if( fout )
      {
        for( const auto& out_line : output )
        {
          fout << out_line << "\n";
        }
        fout.close();
      }
      else
      {
        std::cerr << "Could not write to file: " << input_file << std::endl;
      }
    }
  }

  // Print track count summary
  if( opt_track_count )
  {
    std::cout << "Track count: " << track_counter << " , states = " << state_counter << std::endl;
  }

  // Print types summary
  if( opt_print_types )
  {
    std::cout << "\nTypes found in files:\n" << std::endl;

    // Helper lambda to count types across all files
    auto count_type = [&type_ids]( const std::string& type_name ) -> int {
      int count = 0;
      for( const auto& fn_pair : type_ids )
      {
        const auto& seq_ids = fn_pair.second;
        if( seq_ids.find( type_name ) != seq_ids.end() )
        {
          count += static_cast< int >( seq_ids.at( type_name ).size() );
        }
      }
      return count;
    };

    for( const auto& type_pair : type_counts )
    {
      const std::string& type_name = type_pair.first;

      if( opt_caps_only && !has_uppercase( type_name ) )
      {
        continue;
      }

      if( opt_track_count )
      {
        std::cout << type_name << " " << count_type( type_name ) << std::endl;
      }
      else
      {
        std::cout << type_name << std::endl;
      }
    }
  }

  // Generate comparison file
  if( !opt_comp_file.empty() )
  {
    std::ofstream fout( opt_comp_file );
    if( fout )
    {
      fout << "file_name";
      for( const auto& type_pair : type_counts )
      {
        fout << ", " << type_pair.first;
      }
      fout << "\n";

      for( const auto& fn_pair : type_ids )
      {
        fout << fn_pair.first;
        for( const auto& type_pair : type_counts )
        {
          const std::string& type_name = type_pair.first;
          if( fn_pair.second.find( type_name ) != fn_pair.second.end() )
          {
            fout << ", " << fn_pair.second.at( type_name ).size();
          }
          else
          {
            fout << ", 0";
          }
        }
        fout << "\n";
      }
      fout.close();
    }
    else
    {
      std::cerr << "Could not write comparison file: " << opt_comp_file << std::endl;
    }
  }

  // Print average box sizes
  if( opt_average_box_size )
  {
    std::cout << "Type - Average Box Area - Total Count" << std::endl;
    for( const auto& size_pair : type_sizes )
    {
      double avg_size = size_pair.second / type_counts[ size_pair.first ];
      std::cout << size_pair.first << " " << avg_size << " "
                << type_counts[ size_pair.first ] << std::endl;
    }
  }

  return EXIT_SUCCESS;
}

} // namespace tools
} // namespace viame
