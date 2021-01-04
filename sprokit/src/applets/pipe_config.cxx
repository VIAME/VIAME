// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pipe_config.h"

#include "tool_io.h"

#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/pipeline_util/pipeline_builder.h>
#include <sprokit/pipeline_util/export_pipe.h>

namespace sprokit {
namespace tools {

// ----------------------------------------------------------------------------
pipe_config::
pipe_config()
{
}

// ----------------------------------------------------------------------------
void
pipe_config::
add_command_options()
{
  m_cmd_options->custom_help( wrap_text( "[options] pipe-file\n"
           "This program configures the specified pipeline file."
                                ));

  m_cmd_options->positional_help( "\n  pipe-file  - name of pipeline file." );

  m_cmd_options->add_options()
    ( "h,help", "Display applet usage" );

  m_cmd_options->add_options("pipe")
    ( "c,config", "File containing supplemental configuration entries. Can occur multiple times.",
      cxxopts::value<std::vector<std::string>>() )
    ( "s,setting", "Additional configuration entries in the form of VAR=VALUE. "
      "Can occur multiple times",
      cxxopts::value<std::vector<std::string>>() )
    ( "I,include", "A directory to be added to configuration include path. Can occur multiple times.",
      cxxopts::value<std::vector<std::string>>()  );

  m_cmd_options->add_options("output")
    ( "o,output", "Name of output file or '-' for stdout.",
      cxxopts::value<std::string>()->default_value("-"));

  // positional parameters
  m_cmd_options->add_options()
    ( "p,pipe-file", "Input pipeline file file", cxxopts::value<std::string>())
    ;

  m_cmd_options->parse_positional("pipe-file");
}

// ----------------------------------------------------------------------------
int
pipe_config::
run()
{
  const std::string opt_app_name = applet_name();
  auto& cmd_args = command_args();

  if ( cmd_args["help"].as<bool>() )
  {
    std::cout << m_cmd_options->help();
    return EXIT_SUCCESS;
  }

  // Load all known modules
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.load_all_plugins();

  sprokit::pipeline_builder builder;

  // Add search path to builder.
  if ( cmd_args.count("include") > 0 )
  {
    builder.add_search_path( cmd_args["include"].as<std::vector<std::string>>() );
  }

  if ( cmd_args.count("pipe-file") == 0 )
  {
    // error & exit
    std::cout << "Required pipeline file missing\n "
              << m_cmd_options->help();
    return EXIT_SUCCESS;

  }

  // Load the pipeline file.
  kwiver::vital::path_t const pipe_file( cmd_args["pipe-file"].as<std::string>() );
  builder.load_pipeline( pipe_file );

  // Must be applied after pipe file is loaded.
  // To overwrite any existing settings
  if ( cmd_args.count("config") > 0 )
  {
    std::vector< std::string > config_file_names = cmd_args["config"].as<std::vector<std::string>>();
    for ( auto config : config_file_names )
    {
      builder.load_supplement( config );
    }
  }

  // Add accumulated settings to the pipeline
  if ( cmd_args.count("setting") > 0 )
  {
    std::vector< std::string > config_settings = cmd_args["setting"].as<std::vector<std::string>>();
    for ( auto setting : config_settings )
    {
      builder.add_setting( setting );
    }
  }

  // Get handle to pipeline
  sprokit::pipeline_t const pipe = builder.pipeline();

  // get handle to config block
  kwiver::vital::config_block_sptr const conf = builder.config();

  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;

    return EXIT_FAILURE;
  }

  sprokit::ostream_t const ostr = sprokit::open_ostream( cmd_args["output"].as<std::string>() );

  sprokit::export_pipe exp( builder );

  exp.generate( *ostr.get() );

  return EXIT_SUCCESS;
}

} } // end namespace
