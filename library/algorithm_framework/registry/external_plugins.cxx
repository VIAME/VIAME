/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <viame/algorithm_framework/registry/external_plugins.h>

#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include <cstdlib>

#if defined( _WIN32 )
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace viame {

char const* const plugin_path_variable = "VIAME_PLUGIN_PATH";
char const* const plugin_entry_point = "viame_register_plugin";

namespace {

#if defined( _WIN32 )
constexpr char path_separator = ';';
#else
constexpr char path_separator = ':';
#endif

using register_fn = void ( * )( kwiver::vital::plugin_loader& );

// ----------------------------------------------------------------------------
std::vector< std::string >
split_path( std::string const& value )
{
  std::vector< std::string > entries;
  std::string::size_type start = 0;

  while( start <= value.size() )
  {
    auto const end = value.find( path_separator, start );
    auto const stop = ( end == std::string::npos ) ? value.size() : end;

    // Empty entries are what a trailing or doubled separator leaves behind,
    // and an empty path would mean the current directory, which is exactly
    // the thing this is not willing to load from.
    if( stop > start )
    {
      entries.emplace_back( value.substr( start, stop - start ) );
    }

    if( end == std::string::npos )
    {
      break;
    }

    start = end + 1;
  }

  return entries;
}

// ----------------------------------------------------------------------------
/// Open a library and hand back its registration function, or nothing.
///
/// The handle is deliberately never closed. The factories the plugin
/// registers hold pointers into its code, and they outlive this function by
/// the length of the program.
register_fn
entry_point_of( std::string const& path, kwiver::vital::logger_handle_t logger )
{
#if defined( _WIN32 )
  auto* const handle = LoadLibraryA( path.c_str() );

  if( !handle )
  {
    LOG_WARN(
      logger, "Could not load plugin \"" << path << "\": error "
                                         << GetLastError() );
    return nullptr;
  }

  auto const symbol = reinterpret_cast< register_fn >(
    GetProcAddress( handle, plugin_entry_point ) );
#else
  // RTLD_GLOBAL, as the module loader used: a plugin that pulls in a library
  // a later one also wants must not make those symbols private to itself.
  auto* const handle = dlopen( path.c_str(), RTLD_LAZY | RTLD_GLOBAL );

  if( !handle )
  {
    LOG_WARN( logger, "Could not load plugin \"" << path << "\": "
                                                 << dlerror() );
    return nullptr;
  }

  auto const symbol =
    reinterpret_cast< register_fn >( dlsym( handle, plugin_entry_point ) );
#endif

  if( !symbol )
  {
    LOG_WARN(
      logger, "Plugin \"" << path << "\" exports no "
                          << plugin_entry_point << "; skipping it" );
    return nullptr;
  }

  return symbol;
}

} // namespace

// ----------------------------------------------------------------------------
std::vector< std::string >
register_external_plugins( kwiver::vital::plugin_loader& loader )
{
  std::vector< std::string > registered;

  char const* const value = std::getenv( plugin_path_variable );

  if( !value || !*value )
  {
    return registered;
  }

  auto logger = kwiver::vital::get_logger( "viame.external_plugins" );

  for( auto const& path : split_path( value ) )
  {
    auto const entry_point = entry_point_of( path, logger );

    if( !entry_point )
    {
      continue;
    }

    LOG_DEBUG( logger, "Registering external plugin \"" << path << "\"" );

    loader.set_registering_library( path );
    entry_point( loader );
    loader.set_registering_library( {} );

    registered.push_back( path );
  }

  return registered;
}

} // namespace viame
