/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "python_script_applet.h"

#include <kwiversys/Process.h>
#include <kwiversys/SystemTools.hxx>

#include <vital/logger/logger.h>
#include <vital/util/get_paths.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace kv = kwiver::vital;

namespace viame {
namespace tools {

namespace {

// ----------------------------------------------------------------------------
std::string
join_path( const std::string& dir, const std::string& file )
{
  return dir + "/" + file;
}

// ----------------------------------------------------------------------------
/// Locate an installed tool script, returning an empty string if absent.
std::string
find_script( const std::string& name )
{
  std::vector< std::string > dirs;

  if( const char* install = std::getenv( "VIAME_INSTALL" ) )
  {
    dirs.push_back( join_path( install, "configs" ) );
  }

  dirs.push_back( join_path( kv::get_executable_path(), "../configs" ) );

  for( const auto& dir : dirs )
  {
    const std::string path = join_path( dir, name );

    if( kwiversys::SystemTools::FileExists( path, true ) )
    {
      return path;
    }
  }

  return {};
}

// ----------------------------------------------------------------------------
/// Run a command, sharing this process's streams, and return its exit code.
int
run_command( const std::vector< std::string >& args )
{
  std::vector< const char* > argv;
  argv.reserve( args.size() + 1 );

  for( const auto& arg : args )
  {
    argv.push_back( arg.c_str() );
  }
  argv.push_back( nullptr );

  kwiversysProcess* process = kwiversysProcess_New();

  if( !process )
  {
    return EXIT_FAILURE;
  }

  kwiversysProcess_SetCommand( process, argv.data() );

  // Shared streams let the script print straight through and stay interactive
  kwiversysProcess_SetPipeShared( process, kwiversysProcess_Pipe_STDIN, 1 );
  kwiversysProcess_SetPipeShared( process, kwiversysProcess_Pipe_STDOUT, 1 );
  kwiversysProcess_SetPipeShared( process, kwiversysProcess_Pipe_STDERR, 1 );

  kwiversysProcess_Execute( process );
  kwiversysProcess_WaitForExit( process, nullptr );

  int result = EXIT_FAILURE;

  if( kwiversysProcess_GetState( process ) == kwiversysProcess_State_Exited )
  {
    result = kwiversysProcess_GetExitValue( process );
  }

  kwiversysProcess_Delete( process );

  return result;
}

} // namespace

// ----------------------------------------------------------------------------
int
python_script_applet
::run()
{
  kv::logger_handle_t logger = kv::get_logger( "viame.tools.python_script" );

  const std::string script = find_script( script_name() );

  if( script.empty() )
  {
    LOG_ERROR( logger, "Unable to locate " << script_name() << ". Set "
      "VIAME_INSTALL, or run from an installed VIAME tree." );
    return EXIT_FAILURE;
  }

  std::vector< std::string > args;

#ifdef _WIN32
  args.push_back( "python.exe" );
#else
  args.push_back( "python" );
#endif

  args.push_back( script );

  // Element zero is this program's name, the rest belong to the script
  const auto& forwarded = applet_args();

  for( size_t i = 1; i < forwarded.size(); ++i )
  {
    args.push_back( forwarded[i] );
  }

  return run_command( args );
}

} // namespace tools
} // namespace viame
