/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <kwiversys/SystemTools.hxx>
#include <kwiversys/CommandLineArguments.hxx>

#include <vital/kwiver-include-paths.h>

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/config/config_block.h>
#include <vital/config/config_block_io.h>
#include <vital/algo/algorithm_factory.h>
#include <vital/logger/logger.h>

#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline_util/pipeline_builder.h>
#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline_util/pipe_declaration_types.h>

#include <vector>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <algorithm>
#include <variant>

// =======================================================================================
// JSON output helpers (simple implementation without external dependency)
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
  std::string opt_input_file;
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
  try
  {
    auto algo = kwiver::vital::create_algorithm( algo_type, impl_name );
    if( algo )
    {
      return algo->get_configuration();
    }
  }
  catch( const std::exception& e )
  {
    LOG_DEBUG( g_logger, "Could not create algorithm " << algo_type << ":" << impl_name
               << " - " << e.what() );
  }

  return nullptr;
}

// =======================================================================================
// Structure to hold config parameter information
// =======================================================================================
struct config_param_info
{
  std::string key;
  std::string value;
  std::string description;
  bool is_default;
  bool read_only;
};

// =======================================================================================
// Structure to hold process/algorithm config information
// =======================================================================================
struct component_config
{
  std::string name;
  std::string type;
  std::string impl_name;  // For algorithms, the selected implementation
  std::vector< config_param_info > params;
  std::vector< component_config > nested_algos;
};

// =======================================================================================
// Extract process configuration from pipeline
// =======================================================================================
void extract_process_config(
  const sprokit::process::name_t& proc_name,
  const sprokit::process::type_t& proc_type,
  kwiver::vital::config_block_sptr proc_config,
  component_config& output,
  bool all_impls )
{
  output.name = proc_name;
  output.type = proc_type;

  try
  {
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
      param.key = key;
      param.read_only = false;

      // Get config info (default value and description)
      auto info = proc->config_info( key );
      if( info )
      {
        param.description = info->description;

        // Check if we have an override value from the config
        if( proc_config && proc_config->has_value( key ) )
        {
          param.value = proc_config->get_value< std::string >( key );
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
        param.is_default = true;
      }

      output.params.push_back( param );

      // Check if this is an algorithm type selector (key ends with ":type")
      if( key.length() > 5 && key.substr( key.length() - 5 ) == ":type" )
      {
        std::string algo_prefix = key.substr( 0, key.length() - 5 );
        std::string algo_type;
        std::string selected_impl;

        // Get the algorithm type from description or infer from key
        // Common patterns: detector:type, filter:type, etc.
        auto colon_pos = algo_prefix.rfind( ':' );
        if( colon_pos != std::string::npos )
        {
          algo_type = algo_prefix.substr( colon_pos + 1 );
        }
        else
        {
          algo_type = algo_prefix;
        }

        // Get selected implementation
        if( proc_config && proc_config->has_value( key ) )
        {
          selected_impl = proc_config->get_value< std::string >( key );
        }
        else if( info )
        {
          selected_impl = info->def;
        }

        if( !selected_impl.empty() && selected_impl != "none" )
        {
          component_config algo_config;
          algo_config.name = algo_prefix;
          algo_config.type = algo_type;
          algo_config.impl_name = selected_impl;

          // Get all implementations if requested
          if( all_impls )
          {
            auto impls = get_algorithm_implementations( algo_type );
            for( const auto& impl : impls )
            {
              auto impl_config = get_algorithm_config( algo_type, impl );
              if( impl_config )
              {
                component_config impl_info;
                impl_info.name = impl;
                impl_info.type = algo_type;
                impl_info.impl_name = impl;

                auto impl_keys = impl_config->available_values();
                for( const auto& impl_key : impl_keys )
                {
                  config_param_info impl_param;
                  impl_param.key = impl_key;
                  impl_param.value = impl_config->get_value< std::string >( impl_key, "" );
                  impl_param.description = impl_config->get_description( impl_key );
                  impl_param.is_default = true;
                  impl_param.read_only = impl_config->is_read_only( impl_key );

                  impl_info.params.push_back( impl_param );
                }

                algo_config.nested_algos.push_back( impl_info );
              }
            }
          }
          else
          {
            // Only get config for selected implementation
            auto impl_config = get_algorithm_config( algo_type, selected_impl );
            if( impl_config )
            {
              auto impl_keys = impl_config->available_values();
              for( const auto& impl_key : impl_keys )
              {
                config_param_info algo_param;
                algo_param.key = impl_key;

                // Check for override in process config
                std::string full_key = algo_prefix + ":" + selected_impl + ":" + impl_key;
                if( proc_config && proc_config->has_value( full_key ) )
                {
                  algo_param.value = proc_config->get_value< std::string >( full_key );
                  algo_param.is_default = false;
                }
                else
                {
                  algo_param.value = impl_config->get_value< std::string >( impl_key, "" );
                  algo_param.is_default = true;
                }

                algo_param.description = impl_config->get_description( impl_key );
                algo_param.read_only = impl_config->is_read_only( impl_key );

                algo_config.params.push_back( algo_param );
              }
            }
          }

          output.nested_algos.push_back( algo_config );
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
// Write config as JSON
// =======================================================================================
void write_json_param( std::ostream& os, const config_param_info& param,
                       bool include_desc, const std::string& indent )
{
  os << indent << "{\n";
  os << indent << "  \"key\": \"" << escape_json_string( param.key ) << "\",\n";
  os << indent << "  \"value\": \"" << escape_json_string( param.value ) << "\",\n";
  os << indent << "  \"is_default\": " << ( param.is_default ? "true" : "false" );

  if( include_desc && !param.description.empty() )
  {
    os << ",\n" << indent << "  \"description\": \"" << escape_json_string( param.description ) << "\"";
  }

  if( param.read_only )
  {
    os << ",\n" << indent << "  \"read_only\": true";
  }

  os << "\n" << indent << "}";
}

void write_json_component( std::ostream& os, const component_config& comp,
                           bool include_desc, const std::string& indent )
{
  os << indent << "{\n";
  os << indent << "  \"name\": \"" << escape_json_string( comp.name ) << "\",\n";
  os << indent << "  \"type\": \"" << escape_json_string( comp.type ) << "\"";

  if( !comp.impl_name.empty() )
  {
    os << ",\n" << indent << "  \"implementation\": \"" << escape_json_string( comp.impl_name ) << "\"";
  }

  if( !comp.params.empty() )
  {
    os << ",\n" << indent << "  \"parameters\": [\n";
    for( size_t i = 0; i < comp.params.size(); ++i )
    {
      write_json_param( os, comp.params[i], include_desc, indent + "    " );
      if( i < comp.params.size() - 1 )
      {
        os << ",";
      }
      os << "\n";
    }
    os << indent << "  ]";
  }

  if( !comp.nested_algos.empty() )
  {
    os << ",\n" << indent << "  \"nested_algorithms\": [\n";
    for( size_t i = 0; i < comp.nested_algos.size(); ++i )
    {
      write_json_component( os, comp.nested_algos[i], include_desc, indent + "    " );
      if( i < comp.nested_algos.size() - 1 )
      {
        os << ",";
      }
      os << "\n";
    }
    os << indent << "  ]";
  }

  os << "\n" << indent << "}";
}

// =======================================================================================
// Extract configuration from a .pipe file
// =======================================================================================
bool extract_pipe_config( const std::string& pipe_file,
                          std::vector< component_config >& configs,
                          bool all_impls )
{
  try
  {
    sprokit::pipeline_builder builder;
    builder.load_pipeline( pipe_file );

    auto blocks = builder.pipeline_blocks();
    auto pipe_config = builder.config();

    // Find all process blocks
    for( const auto& block : blocks )
    {
      if( std::holds_alternative< sprokit::process_pipe_block >( block ) )
      {
        auto& proc_block = std::get< sprokit::process_pipe_block >( block );
        std::string proc_name = proc_block.name;
        std::string proc_type = proc_block.type;

        // Get process-specific config
        kwiver::vital::config_block_sptr proc_config_block;
        if( pipe_config )
        {
          proc_config_block = pipe_config->subblock( proc_name );
        }

        component_config comp;
        extract_process_config( proc_name, proc_type, proc_config_block, comp, all_impls );
        configs.push_back( comp );
      }
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
                          std::vector< component_config >& configs,
                          bool all_impls )
{
  try
  {
    // Read config file
    auto config = kwiver::vital::read_config_file( conf_file );

    if( !config )
    {
      LOG_ERROR( g_logger, "Could not read config file: " << conf_file );
      return false;
    }

    // Find top-level algorithm/trainer types
    auto all_keys = config->available_values();

    // Look for pattern like "detector_trainer:type" or "groundtruth_reader:type"
    std::set< std::string > processed_prefixes;

    for( const auto& key : all_keys )
    {
      if( key.length() > 5 && key.substr( key.length() - 5 ) == ":type" )
      {
        std::string algo_prefix = key.substr( 0, key.length() - 5 );

        // Skip if already processed
        if( processed_prefixes.count( algo_prefix ) > 0 )
        {
          continue;
        }
        processed_prefixes.insert( algo_prefix );

        std::string algo_type = config->get_value< std::string >( key );

        if( algo_type.empty() || algo_type == "none" )
        {
          continue;
        }

        component_config comp;
        comp.name = algo_prefix;
        comp.type = algo_prefix;  // For conf files, use prefix as type indicator
        comp.impl_name = algo_type;

        // Get algorithm configuration
        auto algo_config = get_algorithm_config( algo_prefix, algo_type );
        if( !algo_config )
        {
          // Try finding actual algorithm type from the prefix
          // Common patterns: groundtruth_reader -> detected_object_set_input
          //                  detector_trainer -> train_detector
          static const std::map< std::string, std::string > type_mapping = {
            { "groundtruth_reader", "detected_object_set_input" },
            { "detector_trainer", "train_detector" },
            { "image_reader", "image_io" },
            { "descriptor_extractor", "compute_descriptor" },
          };

          auto it = type_mapping.find( algo_prefix );
          if( it != type_mapping.end() )
          {
            algo_config = get_algorithm_config( it->second, algo_type );
            comp.type = it->second;
          }
        }

        // Add parameters from config file for this algorithm
        std::string algo_block_prefix = algo_prefix + ":" + algo_type + ":";
        for( const auto& cfg_key : all_keys )
        {
          if( cfg_key.find( algo_block_prefix ) == 0 )
          {
            config_param_info param;
            param.key = cfg_key.substr( algo_block_prefix.length() );
            param.value = config->get_value< std::string >( cfg_key );
            param.is_default = false;

            // Try to get description from algorithm config
            if( algo_config && algo_config->has_value( param.key ) )
            {
              param.description = algo_config->get_description( param.key );
            }

            comp.params.push_back( param );
          }
        }

        // If we have algorithm config, add default params not in file
        if( algo_config )
        {
          auto algo_keys = algo_config->available_values();
          std::set< std::string > existing_keys;
          for( const auto& p : comp.params )
          {
            existing_keys.insert( p.key );
          }

          for( const auto& algo_key : algo_keys )
          {
            if( existing_keys.count( algo_key ) == 0 )
            {
              config_param_info param;
              param.key = algo_key;
              param.value = algo_config->get_value< std::string >( algo_key, "" );
              param.description = algo_config->get_description( algo_key );
              param.is_default = true;
              param.read_only = algo_config->is_read_only( algo_key );

              comp.params.push_back( param );
            }
          }
        }

        // Get all implementations if requested
        if( all_impls && !comp.type.empty() )
        {
          auto impls = get_algorithm_implementations( comp.type );
          for( const auto& impl : impls )
          {
            if( impl == algo_type )
            {
              continue;  // Skip the selected one
            }

            auto impl_config = get_algorithm_config( comp.type, impl );
            if( impl_config )
            {
              component_config impl_info;
              impl_info.name = impl;
              impl_info.type = comp.type;
              impl_info.impl_name = impl;

              auto impl_keys = impl_config->available_values();
              for( const auto& impl_key : impl_keys )
              {
                config_param_info impl_param;
                impl_param.key = impl_key;
                impl_param.value = impl_config->get_value< std::string >( impl_key, "" );
                impl_param.description = impl_config->get_description( impl_key );
                impl_param.is_default = true;

                impl_info.params.push_back( impl_param );
              }

              comp.nested_algos.push_back( impl_info );
            }
          }
        }

        configs.push_back( comp );
      }
    }

    // Also extract simple key-value parameters that aren't algorithm selectors
    component_config global_config;
    global_config.name = "global";
    global_config.type = "config";

    for( const auto& key : all_keys )
    {
      // Skip algorithm blocks
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
        param.key = key;
        param.value = config->get_value< std::string >( key );
        param.description = config->get_description( key );
        param.is_default = false;

        global_config.params.push_back( param );
      }
    }

    if( !global_config.params.empty() )
    {
      configs.insert( configs.begin(), global_config );
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
    &g_params.opt_input_file, "Input .pipe or .conf file" );
  g_params.m_args.AddArgument( "-i",              argT::SPACE_ARGUMENT,
    &g_params.opt_input_file, "Input .pipe or .conf file" );
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
  // Re-read to check if flag was set
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
              << "Options:\n"
              << g_params.m_args.GetHelp()
              << "\nExamples:\n"
              << "  " << argv[0] << " -i detector.pipe -o detector_config.json\n"
              << "  " << argv[0] << " -i train_detector.conf -o train_config.json\n"
              << "  " << argv[0] << " -i detector.pipe -a  # Include all implementations\n"
              << std::endl;
    return EXIT_SUCCESS;
  }

  // Validate input file
  if( g_params.opt_input_file.empty() )
  {
    LOG_ERROR( g_logger, "No input file specified. Use --input or -i option." );
    return EXIT_FAILURE;
  }

  // Check file exists
  if( !kwiversys::SystemTools::FileExists( g_params.opt_input_file ) )
  {
    LOG_ERROR( g_logger, "Input file does not exist: " << g_params.opt_input_file );
    return EXIT_FAILURE;
  }

  // Load plugins
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  // Extract configurations
  std::vector< component_config > configs;
  bool success = false;

  // Determine file type by extension
  std::string ext = kwiversys::SystemTools::GetFilenameLastExtension( g_params.opt_input_file );
  std::transform( ext.begin(), ext.end(), ext.begin(), ::tolower );

  if( ext == ".pipe" )
  {
    success = extract_pipe_config( g_params.opt_input_file, configs, g_params.opt_all_impls );
  }
  else if( ext == ".conf" )
  {
    success = extract_conf_config( g_params.opt_input_file, configs, g_params.opt_all_impls );
  }
  else
  {
    LOG_ERROR( g_logger, "Unknown file extension: " << ext
               << ". Expected .pipe or .conf" );
    return EXIT_FAILURE;
  }

  if( !success )
  {
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
  *output << "{\n";
  *output << "  \"source_file\": \"" << escape_json_string( g_params.opt_input_file ) << "\",\n";
  *output << "  \"components\": [\n";

  for( size_t i = 0; i < configs.size(); ++i )
  {
    write_json_component( *output, configs[i], g_params.opt_include_descriptions, "    " );
    if( i < configs.size() - 1 )
    {
      *output << ",";
    }
    *output << "\n";
  }

  *output << "  ]\n";
  *output << "}\n";

  if( file_output.is_open() )
  {
    file_output.close();
    LOG_INFO( g_logger, "Configuration written to: " << g_params.opt_output_file );
  }

  return EXIT_SUCCESS;
}
