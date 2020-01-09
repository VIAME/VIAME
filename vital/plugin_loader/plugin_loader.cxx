/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "plugin_loader.h"
#include "plugin_factory.h"

#include <vital/exceptions/plugin.h>
#include <vital/logger/logger.h>
#include <vital/util/demangle.h>
#include <vital/util/string.h>

#include <sstream>

#include <kwiversys/SystemTools.hxx>
#include <kwiversys/Directory.hxx>

namespace kwiver {
namespace vital {

namespace {

using ST =  kwiversys::SystemTools;
using DL =  kwiversys::DynamicLoader;
using library_t =  DL::LibraryHandle;
using function_t = DL::SymbolPointer;

} // end anon namespace


// ==================================================================
/**
 * @brief Plugin manager private implementation.
 *
 */
class plugin_loader_impl
{
public:
  plugin_loader_impl( plugin_loader* parent,
                      std::string const& init_function,
                      std::string const& shared_lib_suffix )
    : m_parent( parent )
    , m_init_function( init_function )
    , m_shared_lib_suffix( shared_lib_suffix )
  { }

  ~plugin_loader_impl()
  { }

  void load_known_modules();
  void look_in_directory( std::string const& directory);
  void load_from_module( std::string const& path);

  void print( std::ostream& str ) const;

  plugin_loader* m_parent;
  const std::string m_init_function;
  const std::string m_shared_lib_suffix;

  /// Paths in which to search for module libraries
  path_list_t m_search_paths;

  // Map from interface type name to vector of class loaders
  plugin_map_t m_plugin_map;

  // Map to keep track of the modules we have opened and loaded.
  typedef std::map< std::string, DL::LibraryHandle > library_map_t;
  library_map_t m_library_map;

  /**
   * \brief Maps module name to source file.
   *
   * This map is used to keep track of whch modules have been
   * loaded. For diagnostic purposes, we also record the file that
   * registered the module.
   */
  plugin_module_map_t m_module_map;

  // Name of current module file we are processing
  std::string m_current_filename;

  std::vector< plugin_filter_handle_t > m_filters;

}; // end class plugin_loader_impl


// ------------------------------------------------------------------
plugin_loader
::plugin_loader( std::string const& init_function,
                 std::string const& shared_lib_suffix )
  : m_logger( kwiver::vital::get_logger( "vital.plugin_loader" ) )
  , m_impl( new plugin_loader_impl( this, init_function, shared_lib_suffix ) )
{ }


plugin_loader
::~plugin_loader()
{ }


// ------------------------------------------------------------------
plugin_factory_vector_t const&
plugin_loader
::get_factories( std::string const& type_name ) const
{
  static plugin_factory_vector_t empty; // needed for error case

  auto const it = m_impl->m_plugin_map.find(type_name);
  if ( it == m_impl->m_plugin_map.end() )
  {
    return empty;
  }

  return it->second;
}


// ------------------------------------------------------------------
plugin_factory_handle_t
plugin_loader
::add_factory( plugin_factory* fact )
{
  plugin_factory_handle_t fact_handle( fact );

  // Add the current file name as an attribute.
  fact->add_attribute( plugin_factory::PLUGIN_FILE_NAME, m_impl->m_current_filename );

  std::string interface_type;
  fact->get_attribute( plugin_factory::INTERFACE_TYPE, interface_type );

  std::string concrete_type;
  fact->get_attribute( plugin_factory::CONCRETE_TYPE, concrete_type );

  // If the hook has declined to register the factory, just return.
  for ( auto filt : m_impl->m_filters )
  {
    if ( ! filt->add_factory( fact_handle ) )
    {
      LOG_TRACE( m_logger, "Factory filter() declined to have this factory registered"
                 << " from file \"" << m_impl->m_current_filename << "\""
                 << "\"" << demangle( interface_type )
                 << "\" for derived type: \"" << demangle( concrete_type ) << "\""
        );
      return fact_handle;
    }
  }

  // Add factory to rest of its family
  m_impl->m_plugin_map[interface_type].push_back( fact_handle );

  LOG_TRACE( m_logger,
             "Adding plugin to create interface: \"" << demangle( interface_type )
             << "\" from derived type: \"" << demangle( concrete_type )
             << "\" from file: " << m_impl->m_current_filename );

  return fact_handle;
}


// ------------------------------------------------------------------
plugin_map_t const&
plugin_loader
::get_plugin_map() const
{
  return m_impl->m_plugin_map;
}


// ------------------------------------------------------------------
void
plugin_loader
::add_search_path( path_list_t const& path)
{
  m_impl->m_search_paths.insert(m_impl->m_search_paths.end(), path.begin(), path.end() );
  // remove any duplicate paths that were added
  erase_duplicates(m_impl->m_search_paths);
}


// ------------------------------------------------------------------
path_list_t const&
plugin_loader
::get_search_path() const
{
  // return vector of paths
  return this->m_impl->m_search_paths;
}


// ------------------------------------------------------------------
std::vector< std::string >
plugin_loader
::get_file_list() const
{
  std::vector< std::string > retval;

  for( auto const it : m_impl->m_library_map )
  {
    retval.push_back( it.first );
  } // end foreach

  return retval;
}


  // ------------------------------------------------------------------
bool
plugin_loader
::is_module_loaded( std::string const& name) const
{
  return (0 != m_impl->m_module_map.count( name ));
}

// ------------------------------------------------------------------
void
plugin_loader
::mark_module_as_loaded( std::string const& name )
{
  m_impl->m_module_map.insert( std::pair< std::string, std::string >(name, m_impl->m_current_filename ) );
}


// ------------------------------------------------------------------
plugin_module_map_t const&
plugin_loader
::get_module_map() const
{
  return m_impl->m_module_map;
}


// ------------------------------------------------------------------
void
plugin_loader
::load_plugins()
{
  m_impl->load_known_modules();
}


// ------------------------------------------------------------------
void
plugin_loader
::load_plugins( path_list_t const& dirpath )
{
  // Iterate over path and load modules
  for( auto const & module_dir : dirpath )
  {
    m_impl->look_in_directory( module_dir );
  }
}


// ------------------------------------------------------------------
void
plugin_loader
::load_plugin( path_t const& file )
{
  m_impl->load_from_module( file );
}


// ==================================================================
/**
 * @brief Load all known modules.
 *
 */
void
plugin_loader_impl
::load_known_modules()
{
  // Iterate over path and load modules
  for( auto const & module_dir : m_search_paths )
  {
    look_in_directory( module_dir );
  }
}


// ------------------------------------------------------------------
void
plugin_loader_impl
::look_in_directory( path_t const& dir_path )
{
  // Check given path for validity
  // Preventing load from current directory via empty string (security)
  if ( dir_path.empty() )
  {
    LOG_DEBUG( m_parent->m_logger, "Empty directory in the search path. Ignoring." );
    return;
  }

  if ( ! ST::FileExists( dir_path ) )
  {
    LOG_DEBUG( m_parent->m_logger, "Path " << dir_path << " doesn't exist. Ignoring." );
    return;
  }

  if ( ! ST::FileIsDirectory( dir_path ) )
  {
    LOG_DEBUG( m_parent->m_logger, "Path " << dir_path << " is not a directory. Ignoring." );
    return;
  }

  // Iterate over search-path directories, attempting module load on elements
  // that end in the configured library suffix.
  LOG_DEBUG( m_parent->m_logger, "Loading plugins from directory: " << dir_path );

  kwiversys::Directory dir;
  dir.Load( dir_path );
  unsigned long num_files = dir.GetNumberOfFiles();

  for (unsigned long i = 0; i < num_files; ++i )
  {
    std::string file = dir.GetPath();
    file += "/" + std::string( dir.GetFile( i ) );

    // Accept this file as a module to check if it has the correct library
    // suffix and matches a provided module name if one was provided.

    if ( ST::GetFilenameLastExtension( file ) == m_shared_lib_suffix )
    {
      // Check that we're looking a file
      if ( ! ST::FileIsDirectory( file ) )
      {
        load_from_module( file );
      }
      else
      {
        LOG_WARN( m_parent->m_logger, "Encountered a directory entry " << file <<
                  " which ends with the expected suffix, but is not a file" );
      }
    }
  } // end for
} // plugin_loader_impl::look_in_directory


// ----------------------------------------------------------------
/**
 * \brief Load single module from shared object / DLL
 *
 * @param path Name of module to load.
 */
void
plugin_loader_impl
::load_from_module( path_t const& path )
{
  DL::LibraryHandle lib_handle;

  m_current_filename = path;

  LOG_DEBUG( m_parent->m_logger, "Loading plugins from: " << path );

  lib_handle = DL::OpenLibrary( path );
  if ( ! lib_handle )
  {
    LOG_WARN( m_parent->m_logger, "plugin_loader::Unable to load shared library \""  << path << "\" : "
              << DL::LastError() );
    return;
  }

  DL::SymbolPointer fp =
    DL::GetSymbolAddress( lib_handle, m_init_function );
  if ( 0 == fp )
  {
    std::string str("Unknown error");
    char const* last_error = DL::LastError();
    if ( last_error )
    {
      str = std::string( last_error );
    }

    LOG_INFO( m_parent->m_logger, "plugin_loader:: Unable to bind to function \"" << m_init_function << "()\" : "
              << str );

    DL::CloseLibrary( lib_handle );
    return;
  }

  // Check with the load hook to see if there are any last minute
  // objections to loading this plugin.
  for ( auto filter : m_filters )
  {
    if ( ! filter->load_plugin( path, lib_handle ) )
    {
      DL::CloseLibrary( lib_handle );
      return;
    }
  }

  // Save currently opened library in map
  m_library_map[path] = lib_handle;

  typedef void (* reg_fp_t)( plugin_loader* );

  reg_fp_t reg_fp = reinterpret_cast< reg_fp_t > ( fp );

  ( *reg_fp )( m_parent ); // register plugins
}

// ----------------------------------------------------------------------------
void plugin_loader
::clear_filters()
{
  m_impl->m_filters.clear();
}

// ----------------------------------------------------------------------------
void plugin_loader
::add_filter( plugin_filter_handle_t f )
{
  f->m_loader = this;
  m_impl->m_filters.push_back( f );
}


} } // end namespace
