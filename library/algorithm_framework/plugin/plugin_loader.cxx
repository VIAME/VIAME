// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "plugin_factory.h"
#include "plugin_loader.h"
#include "plugin_loader_filter.h"

#include <viame/algorithm_framework/exceptions/plugin.h>
#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/util/demangle.h>
#include <viame/algorithm_framework/util/string.h>

#include <kwiversys/Directory.hxx>
#include <kwiversys/DynamicLoader.hxx>
#include <kwiversys/SystemTools.hxx>

#include <sstream>
#include <utility>

#if __linux__
#include <dlfcn.h>
#endif

namespace kwiver {

namespace vital {

namespace {

using ST = kwiversys::SystemTools;
using DL = kwiversys::DynamicLoader;
using library_t = DL::LibraryHandle;
using function_t = DL::SymbolPointer;

} // end anon namespace

// ----------------------------------------------------------------------------
/// @brief Plugin manager private implementation.
///
class plugin_loader_impl
{
public:
  plugin_loader_impl(
    plugin_loader* parent,
    std::string init_function,
    std::string shared_lib_suffix )
    : m_parent( parent ),
      m_init_function( std::move( init_function ) ),
      m_shared_lib_suffix( std::move( shared_lib_suffix ) )
  {}

  ~plugin_loader_impl() = default;

  /// Load all modules in the currently set search path
  void load_known_modules();
  /// Load discovered module libraries in the given filesystem directory path.
  void look_in_directory( std::string const& dir_path );
  /// Attempt loading the module library file given as a filesystem path.
  void load_from_module( std::string const& path );

  // Parent loader instance this impl inst is for.
  plugin_loader* m_parent;
  // Name of the function to dynamically load from the
  const std::string m_init_function;
  const std::string m_shared_lib_suffix;

  /// Paths in which to search for module libraries
  path_list_t m_search_paths;

  // Map from interface name to vector of plugin_factory instances.
  // For consistency, "interface name" refers to the name resulting from
  // `get_interface_name<T>()`.
  plugin_map_t m_plugin_map;

  // Map to keep track of the modules we have opened and loaded.
  typedef std::map< std::string, DL::LibraryHandle > library_map_t;

  library_map_t m_library_map;

  //  Deprecated?
  /// \brief Maps module name to source file.
  ///
  /// This map is used to keep track of whch modules have been
  /// loaded. For diagnostic purposes, we also record the file that
  /// registered the module.
  plugin_module_map_t m_module_map;

  //  Deprecated?
  std::vector< plugin_filter_handle_t > m_filters;

  // Name of current module file we are processing
  std::string m_current_filename;
}; // end class plugin_loader_impl

// ----------------------------------------------------------------------------
plugin_loader
::plugin_loader(
  std::string const& init_function,
  std::string const& shared_lib_suffix )
  : m_logger( kwiver::vital::get_logger( "vital.plugin_loader" ) ),
    m_impl( new plugin_loader_impl( this, init_function, shared_lib_suffix ) )
{}

plugin_loader
::~plugin_loader() = default;

// Factory Stuff ===============================================================
/// @brief Load all known modules.
plugin_factory_vector_t const&
plugin_loader
::get_factories( std::string const& type_name ) const
{
  static plugin_factory_vector_t empty; // needed for error case

  auto const it = m_impl->m_plugin_map.find( type_name );
  if( it == m_impl->m_plugin_map.end() )
  {
    return empty;
  }

  return it->second;
}

// ----------------------------------------------------------------------------
plugin_factory_handle_t
plugin_loader
::add_factory( plugin_factory* fact )
{
  plugin_factory_handle_t fact_handle( fact );

  // Add the current file name as an attribute.
  // This method will inherently be invoked *after* calling the
  // ``plugin_loader_impl::load_from_module`` method which sets the
  // `m_impl->m_current_filename` value.
  fact->add_attribute(
    plugin_factory::PLUGIN_FILE_NAME,
    m_impl->m_current_filename );

  // Get the interface type naming, which ought to be that as returned by
  // `get_interface_name<T>()`.
  // The concrete type is expected to be the mangled `get_concrete_name<T>` and
  // is used in log messaging.
  // Also, the human-readable plugin name.
  std::string interface_type, concrete_type, plugin_name;
  // TODO: Error if any of these are not set.
  if( !fact->get_attribute( plugin_factory::INTERFACE_TYPE, interface_type ) )
  {
    VITAL_THROW(
      plugin_factory_missing_required_attrs,
      "Missing required INTERFACE_TYPE attribute." );
  }
  if( !fact->get_attribute( plugin_factory::CONCRETE_TYPE, concrete_type ) )
  {
    VITAL_THROW(
      plugin_factory_missing_required_attrs,
      "Missing required CONCRETE_TYPE attribute." );
  }
  if( !fact->get_attribute( plugin_factory::PLUGIN_NAME, plugin_name ) )
  {
    VITAL_THROW(
      plugin_factory_missing_required_attrs,
      "Missing required PLUGIN_NAME attribute." );
  }

  auto& fact_list = m_impl->m_plugin_map[ interface_type ];
  // Don't save this factory if we have already loaded it.
  if( !fact_list.empty() )
  {
    for( auto const& afact : fact_list )
    {
      std::string interf, inst, name, prev_file;
      afact->get_attribute( plugin_factory::INTERFACE_TYPE, interf );
      afact->get_attribute( plugin_factory::CONCRETE_TYPE, inst );
      afact->get_attribute( plugin_factory::PLUGIN_NAME, name );
      afact->get_attribute( plugin_factory::PLUGIN_FILE_NAME, prev_file );

      if( ( interface_type == interf ) && ( plugin_name == name ) )
      {
        std::stringstream str;
        if( concrete_type == inst )
        {
          // EXACTLY the same concrete type is being registered.
          // Only log if the paths are different.
          if( prev_file != m_impl->m_current_filename )
          {
            str << "Factory for \"" << interface_type << "\" : \""
                << demangle( concrete_type ) <<
              "\" already has been registered by "
                << prev_file << ".  This factory from "
                << m_impl->m_current_filename << " will not be registered."
                << "Using the existing factory";

            LOG_WARN( this->m_logger, str.str() );
          }
          return afact;
        }
        else
        {
          // A DIFFERENT concrete type is being registered for the same
          // PLUGIN_NAME, which should be unique among plugin factories
          // registered.
          str << "Another factory for interface \"" << interf << "\" has "
              << "already been registered under the same plugin name \""
              << name << "\". "
              << "The existing plugin type (\"" << demangle( concrete_type )
              << "\") is was registered from file \"" << prev_file << "\"."
              << "The current type being registered (\"" << demangle( inst )
              << "\") is being registered from file \""
              << m_impl->m_current_filename << "\".";
          VITAL_THROW( plugin_already_exists, str.str() );
        }
      }
    } // end foreach
  }
  // There used to be a filter step here that was never effectively utilized.
  // This filter had been a check on if the factory should be registered.
  // If this feature is desired again, we can follow a similar pattern to the
  // config stuff and defer to a static function on the implementation class.
  // Since this would likely want to return `true` most of the time the base
  // `pluggable` type could have a default static function implementation.

  // Add factory to rest of its family
  m_impl->m_plugin_map[ interface_type ].push_back( fact_handle );

  LOG_TRACE(
    m_logger,
    "Adding plugin to create interface: \"" << demangle( interface_type )
                                            << "\" with name: \"" <<
      plugin_name
                                            << "\" from derived type: \"" <<
      demangle( concrete_type )
                                            << "\" from file: " <<
      m_impl->m_current_filename );

  return fact_handle;
}

// Map Accessors ===============================================================
plugin_map_t const&
plugin_loader
::get_plugin_map() const
{
  return m_impl->m_plugin_map;
}

// Search path stuff ===========================================================
void
plugin_loader
::add_search_path( path_list_t const& path )
{
  m_impl->m_search_paths.insert(
    m_impl->m_search_paths.end(), path.begin(),
    path.end() );
  // remove any duplicate paths that were added
  erase_duplicates( m_impl->m_search_paths );
}

// ----------------------------------------------------------------------------
path_list_t const&
plugin_loader
::get_search_path() const
{
  // return vector of paths
  return this->m_impl->m_search_paths;
}

// Deprecated? =================================================================
std::vector< std::string >
plugin_loader
::get_file_list() const
{
  std::vector< std::string > retval;

  for( auto const& it : m_impl->m_library_map )
  {
    retval.push_back( it.first );
  } // end foreach

  return retval;
}

// ------------------------------------------------------------------
bool
plugin_loader
::is_module_loaded( std::string const& name ) const
{
  return ( 0 != m_impl->m_module_map.count( name ) );
}

// ------------------------------------------------------------------
void
plugin_loader
::mark_module_as_loaded( std::string const& name )
{
  m_impl->m_module_map.insert(
    std::pair< std::string, std::string >(
      name,
      m_impl->m_current_filename ) );
}

// ----------------------------------------------------------------------------
void
plugin_loader
::clear_filters()
{
  m_impl->m_filters.clear();
}

// ----------------------------------------------------------------------------
void
plugin_loader
::add_filter( plugin_filter_handle_t f )
{
  f->m_loader = this;
  m_impl->m_filters.push_back( f );
}

// ----------------------------------------------------------------------------
plugin_module_map_t const&
plugin_loader
::get_module_map() const
{
  return m_impl->m_module_map;
}

// Loading Factories ===========================================================
void
plugin_loader
::load_plugins()
{
  m_impl->load_known_modules();
}

// ----------------------------------------------------------------------------
void
plugin_loader
::load_plugins( path_list_t const& dirpath )
{
  // Iterate over path and load modules
  for( auto const& module_dir : dirpath )
  {
    m_impl->look_in_directory( module_dir );
  }
}

// ----------------------------------------------------------------------------
void
plugin_loader
::load_plugin( path_t const& file )
{
  m_impl->load_from_module( file );
}

// ----------------------------------------------------------------------------

/**
 * @brief Load all known modules.
 *
 */
void
plugin_loader_impl
::load_known_modules()
{
  // Iterate over path and load modules
  for( auto const& module_dir : m_search_paths )
  {
    look_in_directory( module_dir );
  }
}

// ----------------------------------------------------------------------------
void
plugin_loader_impl
::look_in_directory( path_t const& dir_path )
{
  // Check given path for validity
  // Preventing load from current directory via empty string (security)
  if( dir_path.empty() )
  {
    LOG_DEBUG(
      m_parent->m_logger,
      "Empty directory in the search path. Ignoring." );
    return;
  }

  if( !ST::FileExists( dir_path ) )
  {
    LOG_DEBUG(
      m_parent->m_logger,
      "Path " << dir_path << " doesn't exist. Ignoring." );
    return;
  }

  if( !ST::FileIsDirectory( dir_path ) )
  {
    LOG_DEBUG(
      m_parent->m_logger,
      "Path " << dir_path << " is not a directory. Ignoring." );
    return;
  }

  // Iterate over search-path directories, attempting module load on elements
  // that end in the configured library suffix.
  LOG_DEBUG(
    m_parent->m_logger,
    "Loading plugins from directory: " << dir_path );

  kwiversys::Directory dir;
  dir.Load( ST::CollapseFullPath( dir_path ) );

  unsigned long num_files = dir.GetNumberOfFiles();

  for( unsigned long i = 0; i < num_files; ++i )
  {
    std::string file = dir.GetPath();
    file += "/" + std::string( dir.GetFile( i ) );

    // Accept this file as a module to check if it has the correct library
    // suffix and matches a provided module name if one was provided.

    if( ST::GetFilenameLastExtension( file ) == m_shared_lib_suffix )
    {
      // Check that we're looking a file
      if( !ST::FileIsDirectory( file ) )
      {
        load_from_module( file );
      }
      else
      {
        LOG_WARN(
          m_parent->m_logger, "Encountered a directory entry " << file <<
            " which ends with the expected suffix, but is not a file" );
      }
    }
  } // end for
} // plugin_loader_impl::look_in_directory

// ----------------------------------------------------------------------------
/// \brief Load single module from shared object / DLL
///
/// @param path Name of module to load.
void
plugin_loader_impl
::load_from_module( path_t const& path )
{
  DL::LibraryHandle lib_handle;

  // Utilized in the `add_factory` method when used within the module
  // registration function to set the plugin_factory::PLUGIN_FILE_NAME
  // attribute.
  m_current_filename = path;

  LOG_DEBUG( m_parent->m_logger, "Loading plugins from: " << path );

  // DL::OpenLibrary does not specify either of the RTDL_GLOBAL/RTDL_LOCAL
  // flags.  The default behavior on Linux is RTDL_LOCAL which causes problems
  // in python plugins.  In particular, if we load a python plugin from C++, it
  // will result in undefined symbols errors for symbols that should come from
  // libpython.so.
  //
  // Here is the scenario in more general terms:
  // 1. plugin A is loaded with RTLD_LOCAL;
  // all symbols resolved during this time are "local" so if A introduces usage
  // of library X, its symbols are local to this load.
  // 2. If B is then loaded and also wants to use symbols from X, it gets told
  // "no, those are private to A" and loading fails so it only works if all
  // plugin loads use disjoint libraries (not already loaded) which we can
  // neither guarantee nor require from the plugins.
  //
  // See also explanation in
  // https://github.com/Kitware/sprokit/commit/4f33d9ff0660465552573e17d6174bb8d2934767
  //
  // Related pybind issue: https://github.com/pybind/pybind11/issues/3555
  //
  // TODO: check on  macos if this is required
#if __linux__
  lib_handle = dlopen( path.c_str(), RTLD_LAZY | RTLD_GLOBAL );
#else
  lib_handle = DL::OpenLibrary( path );
#endif
  if( !lib_handle )
  {
    LOG_WARN(
      m_parent->m_logger,
      "plugin_loader::Unable to load shared library \"" << path << "\" : "
                                                        <<
        DL::LastError() );
    return;
  }

  DL::SymbolPointer fp = DL::GetSymbolAddress( lib_handle, m_init_function );
  if( fp == nullptr )
  {
    std::string str( "Unknown error" );
    char const* last_error = DL::LastError();
    if( last_error )
    {
      str = std::string( last_error );
    }

    LOG_WARN(
      m_parent->m_logger,
      "plugin_loader:: Unable to bind to function \"" << m_init_function <<
        "()\" : "
                                                      << str );

    DL::CloseLibrary( lib_handle );
    return;
  }

  // There used to be a filter step here that was never effectively utilized.
  // This filter had been a check if the library should be loaded at all.

  // Save currently opened library in map
  m_library_map[ path ] = lib_handle;

  typedef void ( * reg_fp_t )( plugin_loader& );

  reg_fp_t reg_fp = reinterpret_cast< reg_fp_t >( fp );

  if( m_parent )
  {
    ( *reg_fp )( *m_parent ); // register plugins
  }
}

} // namespace vital

}   // end namespace
