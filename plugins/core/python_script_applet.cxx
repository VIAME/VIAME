/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "python_script_applet.h"

#include <viame/algorithm_framework/util/file_system.h>

#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/util/get_paths.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#if defined( _WIN32 ) || defined( _WIN64 )
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>

extern char** environ;
#endif

namespace kv = kwiver::vital;

namespace viame {

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

    if( kwiver::vital::file_is_regular( path ) )
    {
      return path;
    }
  }

  return {};
}

// ----------------------------------------------------------------------------
/// Run a command, sharing this process's streams, and return its exit code.
///
/// Sharing is the default on both platforms -- a spawned process inherits
/// the parent's standard streams unless told otherwise -- which is what lets
/// the script print straight through and stay interactive. kwiversys had to
/// ask for it explicitly because its own default was to capture.
int
run_command( const std::vector< std::string >& args )
{
  if( args.empty() )
  {
    return EXIT_FAILURE;
  }

#if defined( _WIN32 ) || defined( _WIN64 )
  // Windows takes one command line rather than a vector, and quoting is the
  // caller's problem: an argument containing a space has to arrive as one.
  std::string command_line;

  for( auto const& arg : args )
  {
    if( !command_line.empty() )
    {
      command_line.push_back( ' ' );
    }

    if( arg.find_first_of( " \t\"" ) == std::string::npos )
    {
      command_line += arg;
      continue;
    }

    command_line.push_back( '"' );

    for( char const c : arg )
    {
      if( c == '"' || c == '\\' )
      {
        command_line.push_back( '\\' );
      }

      command_line.push_back( c );
    }

    command_line.push_back( '"' );
  }

  STARTUPINFOA startup{};
  startup.cb = sizeof( startup );

  PROCESS_INFORMATION process{};

  if( !CreateProcessA(
        nullptr, command_line.data(), nullptr, nullptr, TRUE, 0, nullptr,
        nullptr, &startup, &process ) )
  {
    return EXIT_FAILURE;
  }

  WaitForSingleObject( process.hProcess, INFINITE );

  DWORD code = static_cast< DWORD >( EXIT_FAILURE );
  GetExitCodeProcess( process.hProcess, &code );

  CloseHandle( process.hProcess );
  CloseHandle( process.hThread );

  return static_cast< int >( code );
#else
  std::vector< char* > argv;
  argv.reserve( args.size() + 1 );

  for( auto const& arg : args )
  {
    // `posix_spawnp` takes `char* const*` and does not write through it.
    argv.push_back( const_cast< char* >( arg.c_str() ) );
  }

  argv.push_back( nullptr );

  pid_t child = 0;

  if( ::posix_spawnp(
        &child, argv[ 0 ], nullptr, nullptr, argv.data(), environ ) != 0 )
  {
    return EXIT_FAILURE;
  }

  int status = 0;

  while( ::waitpid( child, &status, 0 ) < 0 )
  {
    if( errno != EINTR )
    {
      return EXIT_FAILURE;
    }
  }

  // A script killed by a signal did not exit with a code; report failure
  // rather than inventing one.
  return WIFEXITED( status ) ? WEXITSTATUS( status ) : EXIT_FAILURE;
#endif
}

} // namespace

// ----------------------------------------------------------------------------
std::string
find_tool_script( const std::string& name )
{
  return find_script( name );
}

// ----------------------------------------------------------------------------
int
run_tool_script( const std::string& script,
                 const std::vector< std::string >& args )
{
  std::vector< std::string > command;

#ifdef _WIN32
  command.push_back( "python.exe" );
#else
  command.push_back( "python" );
#endif

  command.push_back( script );
  command.insert( command.end(), args.begin(), args.end() );

  return run_command( command );
}

// ----------------------------------------------------------------------------
int
python_script_applet
::run()
{
  kv::logger_handle_t logger = kv::get_logger( "viame.python_script_applet" );

  const std::string script = find_script( script_name() );

  if( script.empty() )
  {
    LOG_ERROR( logger, "Unable to locate " << script_name() << ". Set "
      "VIAME_INSTALL, or run from an installed VIAME tree." );
    return EXIT_FAILURE;
  }

  // Element zero is this program's name, the rest belong to the script
  const auto& forwarded = applet_args();

  return run_tool_script( script,
    std::vector< std::string >( forwarded.begin() + 1, forwarded.end() ) );
}

} // namespace viame
