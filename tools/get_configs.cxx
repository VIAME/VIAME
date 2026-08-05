/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <kwiversys/SystemTools.hxx>
#include <kwiversys/CommandLineArguments.hxx>
#include <kwiversys/Directory.hxx>

#include <vital/kwiver-include-paths.h>

#include <vital/plugin_management/plugin_manager.h>
#include <vital/plugin_management/plugin_factory.h>
#include <vital/config/config_block.h>
#include <vital/config/config_block_io.h>
#include <vital/logger/logger.h>

#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/process.h>

#include <vector>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <algorithm>
#include <regex>
#include <cctype>
#include <cmath>

// =======================================================================================
// JSON output helpers
// =======================================================================================

std::string escape_json_string( const std::string& input )
{
  std::string output;
  output.reserve( input.size() + 10 );

  for( char c : input )
  {
    switch( c )
    {
      case '"':  output += "\\\""; break;
      case '\\': output += "\\\\"; break;
      case '\b': output += "\\b"; break;
      case '\f': output += "\\f"; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default:
        if( c < 0x20 )
        {
          char buf[8];
          snprintf( buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c) );
          output += buf;
        }
        else
        {
          output += c;
        }
        break;
    }
  }
  return output;
}

// =======================================================================================
// Class storing all input parameters and private variables for tool
// =======================================================================================
class config_extractor_vars
{
public:

  kwiversys::CommandLineArguments m_args;

  bool opt_help;
  bool opt_all_impls;
  bool opt_include_descriptions;
  std::string opt_input_path;
  std::string opt_output_file;

  config_extractor_vars()
  {
    opt_help = false;
    opt_all_impls = false;
    opt_include_descriptions = true;
  }

  virtual ~config_extractor_vars()
  {
  }
};

// =======================================================================================
// Define global variables
// =======================================================================================
static config_extractor_vars g_params;
static kwiver::vital::logger_handle_t g_logger;

// =======================================================================================
// Parameter type inference
// =======================================================================================
enum class param_type
{
  UNKNOWN,
  BOOL,
  INT,
  FLOAT,
  DOUBLE,
  STRING,
  PATH,
  ENUM
};

std::string param_type_to_string( param_type t )
{
  switch( t )
  {
    case param_type::BOOL:   return "bool";
    case param_type::INT:    return "int";
    case param_type::FLOAT:  return "float";
    case param_type::DOUBLE: return "double";
    case param_type::STRING: return "string";
    case param_type::PATH:   return "path";
    case param_type::ENUM:   return "enum";
    default:                 return "string";
  }
}

bool is_integer( const std::string& s )
{
  if( s.empty() ) return false;
  size_t start = 0;
  if( s[0] == '-' || s[0] == '+' ) start = 1;
  if( start >= s.length() ) return false;
  for( size_t i = start; i < s.length(); ++i )
  {
    if( !std::isdigit( s[i] ) ) return false;
  }
  return true;
}

bool is_float( const std::string& s )
{
  if( s.empty() ) return false;
  bool has_dot = false;
  bool has_e = false;
  size_t start = 0;
  if( s[0] == '-' || s[0] == '+' ) start = 1;
  if( start >= s.length() ) return false;

  for( size_t i = start; i < s.length(); ++i )
  {
    char c = s[i];
    if( c == '.' )
    {
      if( has_dot || has_e ) return false;
      has_dot = true;
    }
    else if( c == 'e' || c == 'E' )
    {
      if( has_e ) return false;
      has_e = true;
      if( i + 1 < s.length() && ( s[i+1] == '+' || s[i+1] == '-' ) )
      {
        ++i;
      }
    }
    else if( !std::isdigit( c ) )
    {
      return false;
    }
  }
  return has_dot || has_e;
}

bool is_bool( const std::string& s )
{
  std::string lower = s;
  std::transform( lower.begin(), lower.end(), lower.begin(), ::tolower );
  return lower == "true" || lower == "false" || lower == "yes" || lower == "no" ||
         lower == "on" || lower == "off" || lower == "1" || lower == "0";
}

bool is_path( const std::string& s )
{
  // Check if value looks like a file path
  if( s.find( '/' ) != std::string::npos || s.find( '\\' ) != std::string::npos )
  {
    return true;
  }
  // Check common file extensions
  static const std::vector< std::string > extensions = {
    ".pipe", ".conf", ".cfg", ".txt", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pgm",
    ".mp4", ".avi", ".mov", ".mpg", ".mpeg",
    ".pth", ".pt", ".weights", ".caffemodel", ".pb", ".onnx",
    ".lbl", ".names"
  };
  for( const auto& ext : extensions )
  {
    if( s.length() > ext.length() &&
        s.substr( s.length() - ext.length() ) == ext )
    {
      return true;
    }
  }
  return false;
}

param_type infer_type_from_value( const std::string& value )
{
  if( value.empty() ) return param_type::STRING;
  if( is_bool( value ) ) return param_type::BOOL;
  if( is_integer( value ) ) return param_type::INT;
  if( is_float( value ) ) return param_type::DOUBLE;
  if( is_path( value ) ) return param_type::PATH;
  return param_type::STRING;
}

param_type infer_type_from_description( const std::string& desc )
{
  std::string lower = desc;
  std::transform( lower.begin(), lower.end(), lower.begin(), ::tolower );

  // Check for boolean indicators
  if( lower.find( "true/false" ) != std::string::npos ||
      lower.find( "true or false" ) != std::string::npos ||
      lower.find( "enable" ) != std::string::npos ||
      lower.find( "disable" ) != std::string::npos ||
      lower.find( "whether to" ) != std::string::npos ||
      lower.find( "if true" ) != std::string::npos ||
      lower.find( "if false" ) != std::string::npos ||
      lower.find( "boolean" ) != std::string::npos )
  {
    return param_type::BOOL;
  }

  // Check for integer indicators
  if( lower.find( "integer" ) != std::string::npos ||
      lower.find( "number of" ) != std::string::npos ||
      lower.find( "count" ) != std::string::npos ||
      lower.find( "index" ) != std::string::npos ||
      lower.find( "pixel" ) != std::string::npos )
  {
    return param_type::INT;
  }

  // Check for float/double indicators
  if( lower.find( "float" ) != std::string::npos ||
      lower.find( "double" ) != std::string::npos ||
      lower.find( "ratio" ) != std::string::npos ||
      lower.find( "threshold" ) != std::string::npos ||
      lower.find( "probability" ) != std::string::npos ||
      lower.find( "confidence" ) != std::string::npos ||
      lower.find( "percent" ) != std::string::npos ||
      lower.find( "scale" ) != std::string::npos ||
      lower.find( "factor" ) != std::string::npos )
  {
    return param_type::DOUBLE;
  }

  // Check for path indicators
  if( lower.find( "path" ) != std::string::npos ||
      lower.find( "file" ) != std::string::npos ||
      lower.find( "directory" ) != std::string::npos ||
      lower.find( "folder" ) != std::string::npos )
  {
    return param_type::PATH;
  }

  // Check for enum indicators
  if( lower.find( "must be one of" ) != std::string::npos ||
      lower.find( "options:" ) != std::string::npos ||
      lower.find( "valid values" ) != std::string::npos )
  {
    return param_type::ENUM;
  }

  return param_type::UNKNOWN;
}

param_type infer_param_type( const std::string& value, const std::string& description )
{
  // First try to infer from description as it's more reliable
  param_type desc_type = infer_type_from_description( description );
  if( desc_type != param_type::UNKNOWN )
  {
    return desc_type;
  }

  // Fall back to inferring from value
  return infer_type_from_value( value );
}

// =======================================================================================
// Range extraction from description
// =======================================================================================
struct param_range
{
  bool has_min;
  bool has_max;
  double min_val;
  double max_val;
  std::vector< std::string > enum_values;

  param_range() : has_min( false ), has_max( false ), min_val( 0 ), max_val( 0 ) {}
};

param_range extract_range( const std::string& description, param_type type )
{
  param_range range;

  // Look for range patterns like [0, 1], (0-100), 0 to 255, etc.
  std::regex range_bracket( R"(\[(\-?[\d.]+)\s*[,\-]\s*(\-?[\d.]+)\])" );
  std::regex range_paren( R"(\((\-?[\d.]+)\s*[,\-]\s*(\-?[\d.]+)\))" );
  std::regex range_to( R"((\-?[\d.]+)\s+to\s+(\-?[\d.]+))" );
  std::regex range_between( R"(between\s+(\-?[\d.]+)\s+and\s+(\-?[\d.]+))" );
  std::regex greater_than( R"(greater than\s+(\-?[\d.]+)|>\s*(\-?[\d.]+)|>=\s*(\-?[\d.]+))" );
  std::regex less_than( R"(less than\s+(\-?[\d.]+)|<\s*(\-?[\d.]+)|<=\s*(\-?[\d.]+))" );

  std::smatch match;
  std::string lower = description;
  std::transform( lower.begin(), lower.end(), lower.begin(), ::tolower );

  // Try bracket range [min, max]
  if( std::regex_search( description, match, range_bracket ) )
  {
    try
    {
      range.min_val = std::stod( match[1].str() );
      range.max_val = std::stod( match[2].str() );
      range.has_min = true;
      range.has_max = true;
    }
    catch( ... ) {}
  }
  // Try parenthesis range (min, max)
  else if( std::regex_search( description, match, range_paren ) )
  {
    try
    {
      range.min_val = std::stod( match[1].str() );
      range.max_val = std::stod( match[2].str() );
      range.has_min = true;
      range.has_max = true;
    }
    catch( ... ) {}
  }
  // Try "X to Y" pattern
  else if( std::regex_search( lower, match, range_to ) )
  {
    try
    {
      range.min_val = std::stod( match[1].str() );
      range.max_val = std::stod( match[2].str() );
      range.has_min = true;
      range.has_max = true;
    }
    catch( ... ) {}
  }
  // Try "between X and Y" pattern
  else if( std::regex_search( lower, match, range_between ) )
  {
    try
    {
      range.min_val = std::stod( match[1].str() );
      range.max_val = std::stod( match[2].str() );
      range.has_min = true;
      range.has_max = true;
    }
    catch( ... ) {}
  }

  // Extract enum values if type is ENUM or description contains options
  if( type == param_type::ENUM || lower.find( "must be one of" ) != std::string::npos )
  {
    // Look for patterns like: "Must be one of: opt1, opt2, opt3"
    // Or bullet lists with "-" or "*"
    std::regex enum_pattern( R"(must be one of[:\s]+(.+?)(?:\.|$))" );
    std::regex option_line( R"(^\s*[\-\*]\s*(\w+))" );

    if( std::regex_search( lower, match, enum_pattern ) )
    {
      std::string options_str = match[1].str();
      std::regex word( R"(\w+)" );
      auto words_begin = std::sregex_iterator( options_str.begin(), options_str.end(), word );
      auto words_end = std::sregex_iterator();

      for( std::sregex_iterator i = words_begin; i != words_end; ++i )
      {
        std::string val = (*i).str();
        // Skip common filler words
        if( val != "or" && val != "and" && val != "the" && val != "following" )
        {
          range.enum_values.push_back( val );
        }
      }
    }

    // Also look for newline-separated options
    std::istringstream iss( description );
    std::string line;
    while( std::getline( iss, line ) )
    {
      if( std::regex_search( line, match, option_line ) )
      {
        range.enum_values.push_back( match[1].str() );
      }
    }
  }

  return range;
}

// =======================================================================================
// Structure to hold config parameter information
// =======================================================================================
struct config_param_info
{
  std::string name;
  std::string value;
  std::string description;
  param_type type;
  param_range range;
  bool is_default;
};

// =======================================================================================
// Structure to hold pipeline config information
// =======================================================================================
struct pipeline_config
{
  std::string name;
  std::string file_path;
  std::vector< config_param_info > params;
};

// =======================================================================================
// Get all registered implementation names for an algorithm type
// =======================================================================================
std::vector< std::string >
get_algorithm_implementations( const std::string& algo_type )
{
  std::vector< std::string > impls;

  auto& pm = kwiver::vital::plugin_manager::instance();
  auto factories = pm.get_factories( algo_type );

  for( auto const& fact : factories )
  {
    std::string impl_name;
    if( fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, impl_name ) )
    {
      impls.push_back( impl_name );
    }
  }

  return impls;
}

// =======================================================================================
// Get configuration for an algorithm instance
// =======================================================================================
kwiver::vital::config_block_sptr
get_algorithm_config( const std::string& algo_type, const std::string& impl_name )
{
  // Note: create_algorithm is now a template function requiring compile-time type.
  // Dynamic algorithm creation by type name at runtime is not supported.
  // This function returns nullptr; configuration is obtained from process defaults.
  (void)algo_type;
  (void)impl_name;
  return nullptr;
}

// =======================================================================================
// Parse a .pipe file to extract process definitions
// =======================================================================================
struct pipe_process_info
{
  std::string name;
  std::string type;
};

std::vector< pipe_process_info >
parse_pipe_file_for_processes( const std::string& pipe_file )
{
  std::vector< pipe_process_info > processes;

  std::ifstream file( pipe_file );
  if( !file.is_open() )
  {
    LOG_ERROR( g_logger, "Could not open file: " << pipe_file );
    return processes;
  }

  std::regex process_regex( R"(^\s*process\s+(\S+)\s*$)" );
  std::regex type_regex( R"(^\s*::\s*(\S+)\s*$)" );

  std::string line;
  std::string current_process;

  while( std::getline( file, line ) )
  {
    // Remove comments
    size_t comment_pos = line.find( '#' );
    if( comment_pos != std::string::npos )
    {
      line = line.substr( 0, comment_pos );
    }

    std::smatch match;

    // Check for process declaration
    if( std::regex_match( line, match, process_regex ) )
    {
      current_process = match[1].str();
    }
    // Check for type declaration (must follow process)
    else if( !current_process.empty() && std::regex_match( line, match, type_regex ) )
    {
      pipe_process_info info;
      info.name = current_process;
      info.type = match[1].str();
      processes.push_back( info );
      current_process.clear();
    }
    // Reset if we hit a non-empty, non-whitespace line that's not type
    else if( !current_process.empty() )
    {
      std::string trimmed = line;
      trimmed.erase( 0, trimmed.find_first_not_of( " \t" ) );
      if( !trimmed.empty() && trimmed[0] != ':' )
      {
        current_process.clear();
      }
    }
  }

  return processes;
}

// =======================================================================================
// Extract process parameters and add to flat list
// =======================================================================================
void extract_process_params(
  const std::string& proc_name,
  const std::string& proc_type,
  kwiver::vital::config_block_sptr file_config,
  std::vector< config_param_info >& params,
  bool all_impls )
{
  try
  {
    // Create empty config for process creation
    auto proc_config = kwiver::vital::config_block::empty_config();

    // Create process instance to get available config
    auto proc = sprokit::create_process( proc_type, proc_name, proc_config );

    if( !proc )
    {
      LOG_WARN( g_logger, "Could not create process: " << proc_type );
      return;
    }

    // Get all available config keys
    auto config_keys = proc->available_config();

    // Get configuration values
    for( const auto& key : config_keys )
    {
      // Skip internal keys
      if( key.find( "_" ) == 0 && key != "_non_blocking" )
      {
        continue;
      }

      config_param_info param;
      param.name = proc_name + ":" + key;

      // Get config info (default value and description)
      auto info = proc->config_info( key );
      if( info )
      {
        param.description = info->description;

        // Check if we have an override value from the file config
        std::string full_key = proc_name + kwiver::vital::config_block::block_sep() + key;
        if( file_config && file_config->has_value( full_key ) )
        {
          param.value = file_config->get_value< std::string >( full_key );
          param.is_default = ( param.value == info->def );
        }
        else
        {
          param.value = info->def;
          param.is_default = true;
        }
      }
      else
      {
        param.value = "";
        param.is_default = true;
      }

      // Infer type and extract range
      param.type = infer_param_type( param.value, param.description );
      param.range = extract_range( param.description, param.type );

      params.push_back( param );

      // Check if this is an algorithm type selector (key ends with ":type")
      if( key.length() > 5 && key.substr( key.length() - 5 ) == ":type" )
      {
        std::string algo_prefix = key.substr( 0, key.length() - 5 );
        std::string selected_impl;

        // Get selected implementation from file or default
        std::string full_key = proc_name + kwiver::vital::config_block::block_sep() + key;
        if( file_config && file_config->has_value( full_key ) )
        {
          selected_impl = file_config->get_value< std::string >( full_key );
        }
        else if( info )
        {
          selected_impl = info->def;
        }

        if( !selected_impl.empty() && selected_impl != "none" )
        {
          // Try to determine the algorithm type from the prefix
          std::string algo_type;
          size_t last_colon = algo_prefix.rfind( ':' );
          if( last_colon != std::string::npos )
          {
            algo_type = algo_prefix.substr( last_colon + 1 );
          }
          else
          {
            algo_type = algo_prefix;
          }

          // Get algorithm implementations to process
          std::vector< std::string > impls_to_process;
          impls_to_process.push_back( selected_impl );

          if( all_impls )
          {
            auto all_impl_names = get_algorithm_implementations( algo_type );
            for( const auto& impl : all_impl_names )
            {
              if( impl != selected_impl )
              {
                impls_to_process.push_back( impl );
              }
            }
          }

          // Process each implementation
          for( const auto& impl : impls_to_process )
          {
            auto impl_config = get_algorithm_config( algo_type, impl );
            if( impl_config )
            {
              auto impl_keys = impl_config->available_values();
              for( const auto& impl_key : impl_keys )
              {
                config_param_info algo_param;
                algo_param.name = proc_name + ":" + algo_prefix + ":" + impl + ":" + impl_key;

                // Check for override in file config
                std::string override_key = proc_name + kwiver::vital::config_block::block_sep()
                                         + algo_prefix + kwiver::vital::config_block::block_sep()
                                         + impl + kwiver::vital::config_block::block_sep()
                                         + impl_key;
                if( file_config && file_config->has_value( override_key ) )
                {
                  algo_param.value = file_config->get_value< std::string >( override_key );
                  algo_param.is_default = false;
                }
                else
                {
                  algo_param.value = impl_config->get_value< std::string >( impl_key, "" );
                  algo_param.is_default = true;
                }

                algo_param.description = impl_config->get_description( impl_key );
                algo_param.type = infer_param_type( algo_param.value, algo_param.description );
                algo_param.range = extract_range( algo_param.description, algo_param.type );

                params.push_back( algo_param );
              }
            }
          }
        }
      }
    }
  }
  catch( const std::exception& e )
  {
    LOG_WARN( g_logger, "Error extracting config for process " << proc_name
              << " (" << proc_type << "): " << e.what() );
  }
}

// =======================================================================================
// Extract configuration from a .pipe file
// =======================================================================================
bool extract_pipe_config( const std::string& pipe_file,
                          pipeline_config& config,
                          bool all_impls )
{
  try
  {
    // Set pipeline name from filename
    config.file_path = pipe_file;
    config.name = kwiversys::SystemTools::GetFilenameWithoutLastExtension( pipe_file );

    // Parse pipe file to find process definitions
    auto processes = parse_pipe_file_for_processes( pipe_file );

    if( processes.empty() )
    {
      LOG_WARN( g_logger, "No processes found in pipe file: " << pipe_file );
      return false;
    }

    // Read the config file to get values
    kwiver::vital::config_block_sptr file_config;
    try
    {
      file_config = kwiver::vital::read_config_file( pipe_file );
    }
    catch( const std::exception& e )
    {
      LOG_DEBUG( g_logger, "Could not read as config file: " << e.what() );
    }

    // Extract config for each process
    for( const auto& proc_info : processes )
    {
      extract_process_params( proc_info.name, proc_info.type, file_config,
                              config.params, all_impls );
    }

    return true;
  }
  catch( const std::exception& e )
  {
    LOG_ERROR( g_logger, "Error parsing pipeline file: " << e.what() );
    return false;
  }
}

// =======================================================================================
// Extract configuration from a .conf file
// =======================================================================================
bool extract_conf_config( const std::string& conf_file,
                          pipeline_config& config,
                          bool all_impls )
{
  try
  {
    // Set config name from filename
    config.file_path = conf_file;
    config.name = kwiversys::SystemTools::GetFilenameWithoutLastExtension( conf_file );

    // Read config file
    auto file_config = kwiver::vital::read_config_file( conf_file );

    if( !file_config )
    {
      LOG_ERROR( g_logger, "Could not read config file: " << conf_file );
      return false;
    }

    // Get all keys
    auto all_keys = file_config->available_values();

    // Find algorithm type selectors and process them
    std::set< std::string > processed_prefixes;

    for( const auto& key : all_keys )
    {
      if( key.length() > 5 && key.substr( key.length() - 5 ) == ":type" )
      {
        std::string algo_prefix = key.substr( 0, key.length() - 5 );

        if( processed_prefixes.count( algo_prefix ) > 0 )
        {
          continue;
        }
        processed_prefixes.insert( algo_prefix );

        std::string impl_type = file_config->get_value< std::string >( key );

        if( impl_type.empty() || impl_type == "none" )
        {
          continue;
        }

        // Add the type selector itself
        config_param_info type_param;
        type_param.name = key;
        type_param.value = impl_type;
        type_param.description = file_config->get_description( key );
        type_param.type = param_type::ENUM;
        type_param.is_default = false;
        config.params.push_back( type_param );

        // Map common prefixes to algorithm types
        static const std::map< std::string, std::string > type_mapping = {
          { "groundtruth_reader", "detected_object_set_input" },
          { "detector_trainer", "train_detector" },
          { "image_reader", "image_io" },
        };

        std::string algo_type;
        auto it = type_mapping.find( algo_prefix );
        if( it != type_mapping.end() )
        {
          algo_type = it->second;
        }
        else
        {
          // Try to extract from the prefix itself
          size_t last_colon = algo_prefix.rfind( ':' );
          if( last_colon != std::string::npos )
          {
            algo_type = algo_prefix.substr( last_colon + 1 );
          }
          else
          {
            algo_type = algo_prefix;
          }
        }

        // Get algorithm configuration
        auto algo_config = get_algorithm_config( algo_type, impl_type );

        // Add parameters from config file for this algorithm
        std::string algo_block_prefix = algo_prefix + ":" + impl_type + ":";
        std::set< std::string > file_keys;

        for( const auto& cfg_key : all_keys )
        {
          if( cfg_key.find( algo_block_prefix ) == 0 )
          {
            std::string param_key = cfg_key.substr( algo_block_prefix.length() );
            file_keys.insert( param_key );

            config_param_info param;
            param.name = cfg_key;
            param.value = file_config->get_value< std::string >( cfg_key );
            param.is_default = false;

            if( algo_config && algo_config->has_value( param_key ) )
            {
              param.description = algo_config->get_description( param_key );
            }

            param.type = infer_param_type( param.value, param.description );
            param.range = extract_range( param.description, param.type );

            config.params.push_back( param );
          }
        }

        // Add default params not in file (from algorithm config)
        if( algo_config )
        {
          auto algo_keys = algo_config->available_values();
          for( const auto& algo_key : algo_keys )
          {
            if( file_keys.count( algo_key ) == 0 )
            {
              config_param_info param;
              param.name = algo_block_prefix + algo_key;
              param.value = algo_config->get_value< std::string >( algo_key, "" );
              param.description = algo_config->get_description( algo_key );
              param.is_default = true;
              param.type = infer_param_type( param.value, param.description );
              param.range = extract_range( param.description, param.type );

              config.params.push_back( param );
            }
          }
        }

        // Get all implementations if requested
        if( all_impls && !algo_type.empty() )
        {
          auto impls = get_algorithm_implementations( algo_type );
          for( const auto& impl : impls )
          {
            if( impl == impl_type )
            {
              continue;
            }

            auto impl_config = get_algorithm_config( algo_type, impl );
            if( impl_config )
            {
              std::string impl_prefix = algo_prefix + ":" + impl + ":";
              auto impl_keys = impl_config->available_values();

              for( const auto& impl_key : impl_keys )
              {
                config_param_info param;
                param.name = impl_prefix + impl_key;
                param.value = impl_config->get_value< std::string >( impl_key, "" );
                param.description = impl_config->get_description( impl_key );
                param.is_default = true;
                param.type = infer_param_type( param.value, param.description );
                param.range = extract_range( param.description, param.type );

                config.params.push_back( param );
              }
            }
          }
        }
      }
    }

    // Add global parameters (not part of algorithm blocks)
    for( const auto& key : all_keys )
    {
      bool skip = false;
      for( const auto& prefix : processed_prefixes )
      {
        if( key.find( prefix + ":" ) == 0 || key == prefix + ":type" )
        {
          skip = true;
          break;
        }
      }

      if( !skip )
      {
        config_param_info param;
        param.name = key;
        param.value = file_config->get_value< std::string >( key );
        param.description = file_config->get_description( key );
        param.is_default = false;
        param.type = infer_param_type( param.value, param.description );
        param.range = extract_range( param.description, param.type );

        config.params.push_back( param );
      }
    }

    return true;
  }
  catch( const std::exception& e )
  {
    LOG_ERROR( g_logger, "Error parsing config file: " << e.what() );
    return false;
  }
}

// =======================================================================================
// Check if file is a training conf (vs a pipeline support conf)
// =======================================================================================
bool is_training_conf( const std::string& filename )
{
  std::string lower = filename;
  std::transform( lower.begin(), lower.end(), lower.begin(), ::tolower );
  return lower.find( "train" ) != std::string::npos;
}

// =======================================================================================
// Write JSON output for all pipelines
// =======================================================================================
void write_json_output( std::ostream& os,
                        const std::vector< pipeline_config >& pipelines,
                        bool include_desc )
{
  os << "{\n";

  for( size_t p = 0; p < pipelines.size(); ++p )
  {
    const auto& pipeline = pipelines[p];

    os << "  \"" << escape_json_string( pipeline.name ) << "\": {\n";
    os << "    \"file\": \"" << escape_json_string( pipeline.file_path ) << "\",\n";
    os << "    \"parameters\": [\n";

    for( size_t i = 0; i < pipeline.params.size(); ++i )
    {
      const auto& param = pipeline.params[i];

      os << "      {\n";
      os << "        \"name\": \"" << escape_json_string( param.name ) << "\",\n";
      os << "        \"type\": \"" << param_type_to_string( param.type ) << "\",\n";
      os << "        \"default\": \"" << escape_json_string( param.value ) << "\"";

      if( include_desc && !param.description.empty() )
      {
        os << ",\n        \"description\": \"" << escape_json_string( param.description ) << "\"";
      }

      // Add range if available
      if( param.range.has_min || param.range.has_max || !param.range.enum_values.empty() )
      {
        os << ",\n        \"range\": {";

        bool first = true;
        if( param.range.has_min )
        {
          os << "\n          \"min\": " << param.range.min_val;
          first = false;
        }
        if( param.range.has_max )
        {
          if( !first ) os << ",";
          os << "\n          \"max\": " << param.range.max_val;
          first = false;
        }
        if( !param.range.enum_values.empty() )
        {
          if( !first ) os << ",";
          os << "\n          \"values\": [";
          for( size_t e = 0; e < param.range.enum_values.size(); ++e )
          {
            if( e > 0 ) os << ", ";
            os << "\"" << escape_json_string( param.range.enum_values[e] ) << "\"";
          }
          os << "]";
        }

        os << "\n        }";
      }

      os << "\n      }";

      if( i < pipeline.params.size() - 1 )
      {
        os << ",";
      }
      os << "\n";
    }

    os << "    ]\n";
    os << "  }";

    if( p < pipelines.size() - 1 )
    {
      os << ",";
    }
    os << "\n";
  }

  os << "}\n";
}

// =======================================================================================
// Main entry point
// =======================================================================================
int main( int argc, char* argv[] )
{
  // Initialize logger
  g_logger = kwiver::vital::get_logger( "viame.tools.get_configs" );

  // Setup command line arguments
  typedef kwiversys::CommandLineArguments argT;

  g_params.m_args.Initialize( argc, argv );

  g_params.m_args.AddArgument( "--help",          argT::NO_ARGUMENT,
    &g_params.opt_help, "Display usage information" );
  g_params.m_args.AddArgument( "-h",              argT::NO_ARGUMENT,
    &g_params.opt_help, "Display usage information" );
  g_params.m_args.AddArgument( "--input",         argT::SPACE_ARGUMENT,
    &g_params.opt_input_path, "Input .pipe/.conf file or directory" );
  g_params.m_args.AddArgument( "-i",              argT::SPACE_ARGUMENT,
    &g_params.opt_input_path, "Input .pipe/.conf file or directory" );
  g_params.m_args.AddArgument( "--output",        argT::SPACE_ARGUMENT,
    &g_params.opt_output_file, "Output .json file (default: stdout)" );
  g_params.m_args.AddArgument( "-o",              argT::SPACE_ARGUMENT,
    &g_params.opt_output_file, "Output .json file (default: stdout)" );
  g_params.m_args.AddArgument( "--all-implementations", argT::NO_ARGUMENT,
    &g_params.opt_all_impls,
    "Include configs for all registered implementations, not just selected ones" );
  g_params.m_args.AddArgument( "-a",              argT::NO_ARGUMENT,
    &g_params.opt_all_impls,
    "Include configs for all registered implementations" );
  g_params.m_args.AddArgument( "--no-descriptions", argT::NO_ARGUMENT,
    &g_params.opt_include_descriptions,
    "Exclude parameter descriptions from output" );

  // Parse command line
  if( !g_params.m_args.Parse() )
  {
    LOG_ERROR( g_logger, "Problem parsing arguments" );
    return EXIT_FAILURE;
  }

  // Handle special case for --no-descriptions (it's inverted)
  for( int i = 1; i < argc; ++i )
  {
    if( std::string( argv[i] ) == "--no-descriptions" )
    {
      g_params.opt_include_descriptions = false;
      break;
    }
  }

  // Display help
  if( g_params.opt_help )
  {
    std::cout << "Usage: " << argv[0] << " [options]\n\n"
              << "Extract configuration parameters from KWIVER pipeline (.pipe) or\n"
              << "training configuration (.conf) files and output them as JSON.\n\n"
              << "The input can be a single file or a directory. When given a directory,\n"
              << "all .pipe files and training .conf files are processed into a single\n"
              << "JSON output.\n\n"
              << "Options:\n"
              << g_params.m_args.GetHelp()
              << "\nExamples:\n"
              << "  " << argv[0] << " -i detector.pipe -o detector_config.json\n"
              << "  " << argv[0] << " -i train_detector.conf -o train_config.json\n"
              << "  " << argv[0] << " -i configs/pipelines/ -o all_configs.json\n"
              << "  " << argv[0] << " -i detector.pipe -a  # Include all implementations\n"
              << std::endl;
    return EXIT_SUCCESS;
  }

  // Validate input
  if( g_params.opt_input_path.empty() )
  {
    LOG_ERROR( g_logger, "No input specified. Use --input or -i option." );
    return EXIT_FAILURE;
  }

  // Check path exists
  if( !kwiversys::SystemTools::FileExists( g_params.opt_input_path ) )
  {
    LOG_ERROR( g_logger, "Input path does not exist: " << g_params.opt_input_path );
    return EXIT_FAILURE;
  }

  // Load plugins
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  // Collect files to process
  std::vector< std::string > files_to_process;

  if( kwiversys::SystemTools::FileIsDirectory( g_params.opt_input_path ) )
  {
    // Process directory
    kwiversys::Directory dir;
    if( !dir.Load( g_params.opt_input_path ) )
    {
      LOG_ERROR( g_logger, "Could not read directory: " << g_params.opt_input_path );
      return EXIT_FAILURE;
    }

    for( unsigned long i = 0; i < dir.GetNumberOfFiles(); ++i )
    {
      std::string filename = dir.GetFile( i );
      if( filename == "." || filename == ".." )
      {
        continue;
      }

      std::string filepath = g_params.opt_input_path + "/" + filename;
      std::string ext = kwiversys::SystemTools::GetFilenameLastExtension( filename );
      std::transform( ext.begin(), ext.end(), ext.begin(), ::tolower );

      if( ext == ".pipe" )
      {
        files_to_process.push_back( filepath );
      }
      else if( ext == ".conf" && is_training_conf( filename ) )
      {
        files_to_process.push_back( filepath );
      }
    }

    // Sort files for consistent output
    std::sort( files_to_process.begin(), files_to_process.end() );
  }
  else
  {
    // Single file
    files_to_process.push_back( g_params.opt_input_path );
  }

  if( files_to_process.empty() )
  {
    LOG_ERROR( g_logger, "No .pipe or training .conf files found" );
    return EXIT_FAILURE;
  }

  // Process all files
  std::vector< pipeline_config > pipelines;

  for( const auto& file : files_to_process )
  {
    std::string ext = kwiversys::SystemTools::GetFilenameLastExtension( file );
    std::transform( ext.begin(), ext.end(), ext.begin(), ::tolower );

    pipeline_config config;
    bool success = false;

    if( ext == ".pipe" )
    {
      success = extract_pipe_config( file, config, g_params.opt_all_impls );
    }
    else if( ext == ".conf" )
    {
      success = extract_conf_config( file, config, g_params.opt_all_impls );
    }

    if( success && !config.params.empty() )
    {
      pipelines.push_back( config );
    }
    else
    {
      LOG_WARN( g_logger, "Skipping file (no params extracted): " << file );
    }
  }

  if( pipelines.empty() )
  {
    LOG_ERROR( g_logger, "No configurations extracted from any files" );
    return EXIT_FAILURE;
  }

  // Write output
  std::ostream* output = &std::cout;
  std::ofstream file_output;

  if( !g_params.opt_output_file.empty() )
  {
    file_output.open( g_params.opt_output_file );
    if( !file_output.is_open() )
    {
      LOG_ERROR( g_logger, "Could not open output file: " << g_params.opt_output_file );
      return EXIT_FAILURE;
    }
    output = &file_output;
  }

  // Write JSON
  write_json_output( *output, pipelines, g_params.opt_include_descriptions );

  if( file_output.is_open() )
  {
    file_output.close();
    LOG_INFO( g_logger, "Configuration written to: " << g_params.opt_output_file );
  }

  return EXIT_SUCCESS;
}
