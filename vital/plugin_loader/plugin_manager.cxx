/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

#include "plugin_manager.h"
#include "plugin_factory.h"

#include <vital/algorithm_plugin_manager_paths.h>

#include <vital/logger/logger.h>

#include <sstream>

#include <kwiversys/SystemTools.hxx>
#include <kwiversys/Directory.hxx>

//+ kwiversys has a demangle but it is private
#if defined __linux__
  #include <cxxabi.h>
#endif


namespace kwiver {
namespace vital {

namespace {

typedef kwiversys::SystemTools     ST;
typedef kwiversys::DynamicLoader   DL;
typedef DL::LibraryHandle          library_t;
typedef DL::SymbolPointer          function_t;


// Platform specific plugin library file (set as compile definition in CMake)
static std::string const shared_library_suffix = std::string( SHARED_LIB_SUFFIX );

} // end anon namespace


// ==================================================================
/**
 * @brief Plugin manager private implementation.
 *
 */
class plugin_manager_impl
{
public:
  plugin_manager_impl( plugin_manager* parent,
    std::string const& init_fucntion)
    : m_parent( parent ),
      m_init_function( init_fucntion ),
      m_logger( kwiver::vital::get_logger( "vital.plugin_manager" ) )
  { }

  ~plugin_manager_impl() { }

  void load_known_modules();
  void look_in_directory( std::string const& directory);
  void load_from_module( std::string const& path);

  void print( std::ostream& str ) const;

  plugin_manager* m_parent;
  const std::string m_init_function;

  /// Paths in which to search for module libraries
  typedef std::vector< path_t > search_paths_t;
  search_paths_t m_search_paths;


  // Map from interface type name to vector of class loaders
  plugin_map_t m_plugin_map;

  // Map to keep track of the modules we have opened and loaded.
  typedef std::map< std::string, DL::LibraryHandle > library_map_t;
  library_map_t m_library_map;

  // Name of current module file we are processing
  std::string m_current_filename;

  kwiver::vital::logger_handle_t m_logger;

}; // end class plugin_manager_impl


// ------------------------------------------------------------------
plugin_manager
::plugin_manager(std::string const& init_function )
  : m_impl( new plugin_manager_impl( this, init_function ) )
{ }


plugin_manager
::~plugin_manager()
{
  VITAL_FOREACH( auto entry, m_impl->m_library_map )
  {
    DL::CloseLibrary( entry.second );
  }
}


// ------------------------------------------------------------------
plugin_factory_vector_t const&
plugin_manager
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
plugin_manager
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
  if ( ! this->add_factory_hook( fact_handle ) )
  {
    LOG_TRACE( m_impl->m_logger, "add_factory_hook() declined to have this factory registered"
               << " from file \"" << m_impl->m_current_filename << "\""
               << " for interface: \"" << interface_type
               << "\" for derived type: \"" << concrete_type << "\""
      );
    return fact_handle;
  }

  // Make sure factory is not already in the list.
  // Check the two types as a signature.
  if ( m_impl->m_plugin_map.count( interface_type ) != 0)
  {
    VITAL_FOREACH( auto const fact, m_impl->m_plugin_map[interface_type] )
    {
      std::string interf;
      fact->get_attribute( plugin_factory::INTERFACE_TYPE, interf );

      std::string inst;
      fact->get_attribute( plugin_factory::CONCRETE_TYPE, inst );

      if ( (interface_type == interf) && (concrete_type == inst) )
      {
        LOG_WARN( m_impl->m_logger, "Factory for \"" << interface_type << "\" : \""
                  << concrete_type << "\" already has been registered.  This factory from "
                  << m_impl->m_current_filename << " will not be registered."
          );
        return fact_handle;
      }
    } // end foreach
  }

  // Add factory to rest of its family
  m_impl->m_plugin_map[interface_type].push_back( fact_handle );

  LOG_TRACE( m_impl->m_logger,
             "Adding plugin to create interface: " << interface_type
             << " from derived type: " << concrete_type
             << " from file: " << m_impl->m_current_filename );

  return fact_handle;
}


// ------------------------------------------------------------------
plugin_map_t const&
plugin_manager
::get_plugin_map() const
{
  return m_impl->m_plugin_map;
}


// ------------------------------------------------------------------
void
plugin_manager
::add_search_path( path_t const& path)
{
  // Split supplied path into separate items using PATH_SEPARATOR_CHAR as delimiter
  // and add to search paths.
  ST::Split( path, m_impl->m_search_paths, PATH_SEPARATOR_CHAR );

}


// ------------------------------------------------------------------
std::vector< path_t > const&
plugin_manager
::get_search_path() const
{
  //return vector of paths
  return this->m_impl->m_search_paths;
}


// ------------------------------------------------------------------
std::vector< std::string >
plugin_manager
::get_file_list() const
{
  std::vector< std::string > retval;

  VITAL_FOREACH( auto const it, m_impl->m_library_map )
  {
    retval.push_back( it.first );
  } // end foreach

  return retval;
}


// ------------------------------------------------------------------
void
plugin_manager
::load_plugins()
{
  m_impl->load_known_modules();
}


// ==================================================================
/**
 * @brief Load all known modules.
 *
 */
void
plugin_manager_impl
::load_known_modules()
{
  // Iterate over path and load modules
  VITAL_FOREACH( auto const & module_dir, m_search_paths )
  {
    look_in_directory( module_dir );
  }
}


// ------------------------------------------------------------------
void
plugin_manager_impl
::look_in_directory( path_t const& dir_path )
{
  // Check given path for validity
  // Preventing load from current directory via empty string (security)
  if ( dir_path.empty() )
  {
    LOG_DEBUG( m_logger, "Empty directory in the search path. Ignoring." );
    return;
  }

  if ( ! ST::FileExists( dir_path ) )
  {
    LOG_DEBUG( m_logger, "Path " << dir_path << " doesn't exist. Ignoring." );
    return;
  }

  if ( ! ST::FileIsDirectory( dir_path ) )
  {
    LOG_DEBUG( m_logger, "Path " << dir_path << " is not a directory. Ignoring." );
    return;
  }

  // Iterate over search-path directories, attempting module load on elements
  // that end in the configured library suffix.
  LOG_DEBUG( m_logger, "Loading modules from directory: " << dir_path );

  kwiversys::Directory dir;
  dir.Load( dir_path );
  unsigned long num_files = dir.GetNumberOfFiles();

  for (unsigned long i = 0; i < num_files; ++i )
  {
    std::string file = dir.GetPath();
    file += "/" + std::string( dir.GetFile( i ) );

    // Accept this file as a module to check if it has the correct library
    // suffix and matches a provided module name if one was provided.

    if ( ST::GetFilenameLastExtension( file ) == shared_library_suffix )
    {
      // Check that we're looking a file
      if ( ! ST::FileIsDirectory( file ) )
      {
        load_from_module( file );
      }
      else
      {
        LOG_WARN( m_logger, "Encountered a directory entry " << file <<
                  " which ends with the expected suffix, but is not a file" );
      }
    }
  } // end for
} // plugin_manager_impl::look_in_directory


// ----------------------------------------------------------------
/**
 * \brief Load single module from shared object / DLL
 *
 * @param path Name of module to load.
 */
void
plugin_manager_impl
::load_from_module( path_t const& path )
{
  DL::LibraryHandle lib_handle;

  m_current_filename = path;

  LOG_DEBUG( m_logger, "Loading plugins from: " << path );

  lib_handle = DL::OpenLibrary( path );
  if ( ! lib_handle )
  {
    LOG_WARN( m_logger, "plugin_manager::Unable to load shared library \""  << path << "\" : "
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

    LOG_WARN( m_logger, "plugin_manager:: Unable to bind to function \"" << m_init_function << "()\" : "
              << last_error );

    DL::CloseLibrary( lib_handle );
    return;
  }

  // Check with the load hook to see if there are any last minute
  // objections to loading this plugin.
  if ( ! m_parent->load_plugin_hook( path, lib_handle ) )
  {
    DL::CloseLibrary( lib_handle );
    return;
  }

  // Save currently opened library in map
  m_library_map[path] = lib_handle;

  typedef void (* reg_fp_t)( plugin_manager* );

  reg_fp_t reg_fp = reinterpret_cast< reg_fp_t > ( fp );

  ( *reg_fp )( m_parent ); // register plugins
}


// ------------------------------------------------------------------
bool
plugin_manager
::load_plugin_hook( path_t const& path, DL::LibraryHandle lib_handle ) const
{
  return true; // default is to always load
}


// ------------------------------------------------------------------
bool
plugin_manager
::add_factory_hook( plugin_factory_handle_t fact ) const
{
  return true; // default is to always register factory
}

} } // end namespace
