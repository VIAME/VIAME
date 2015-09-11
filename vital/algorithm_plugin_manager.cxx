/*ckwg +29
 * Copyright 2014-2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

/**
 * \file
 * \brief Algorithm plugin manager implementation
 * \todo look into using kwiversys for DLL support
 */

#include "algorithm_plugin_manager.h"

#include <vital/algorithm_plugin_manager_paths.h>

#include <vital/vital_apm_export.h>
#include <vital/registrar.h>
#include <vital/logger/logger.h>

#include <kwiversys/DynamicLoader.hxx>
#include <kwiversys/SystemTools.hxx>

// Need boost filesystem until we can find a simple alternative for
// iterating through a directory.
#ifndef BOOST_FILESYSTEM_VERSION
 #define BOOST_FILESYSTEM_VERSION 3
#else
 #if BOOST_FILESYSTEM_VERSION == 2
  #error "Only boost::filesystem version 3 is supported."
 #endif
#endif

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;


#include <map>
#include <string>
#include <vector>
#include <mutex>

namespace kwiver {
namespace vital {

namespace // anonymous
{

typedef kwiversys::SystemTools ST;
typedef kwiversys::DynamicLoader DL;
typedef DL::LibraryHandle library_t;
typedef DL::SymbolPointer function_t;

typedef int (* register_impls_func_t)( registrar& );


static char const* environment_variable_name( "VITAL_PLUGIN_PATH" );

// String name of the private interface function.
// See source file located @ CMake/templates/cxx/plugin_shell.cxx
static std::string const register_function_name = std::string( "private_register_algo_impls" );

// Platform specific plugin library file (set as compile definition in CMake)
static std::string const shared_library_suffix = std::string( SHARED_LIB_SUFFIX );

// Default module directory locations. Values defined in CMake configuration.
static std::string const default_module_paths= std::string( DEFAULT_MODULE_PATHS );

} // end anonymous namespace


// ---- Static ---
algorithm_plugin_manager * algorithm_plugin_manager::s_instance( 0 );

// ===========================================================================
// PluginManager Private Implementation
// ---------------------------------------------------------------------------

class algorithm_plugin_manager::impl
{
// Memeber Variables ---------------------------------------------------------


public:
  /// Paths in which to search for module libraries
  typedef std::vector< path_t > search_paths_t;
  search_paths_t m_search_paths;

  /// module libraries already loaded, keyed on filename
  typedef std::map< std::string, path_t > registered_modules_t;
  registered_modules_t registered_modules_;

  kwiver::vital::logger_handle_t m_logger;

// Member functions ----------------------------------------------------------


public:
  impl()
    : m_search_paths(),
      m_logger( kwiver::vital::get_logger( "algorithm_plugin_manager" ) )
  { }


  // ------------------------------------------------------------------
  /// Attempt loading algorithm implementations from all known search paths
  void load_from_search_paths( std::string name = std::string() )
  {
    LOG_DEBUG( m_logger, "Loading plugins in search paths" );

    // \todo: Want a way to hook into an environment variable / config file here
    //       for additional search path extension
    //       - Search order: setInCode -> EnvVar -> configFile -> defaults
    //       - create separate default_m_search_paths member var for separate storage

    for ( path_t module_dir : this->m_search_paths )
    {
      load_modules_in_directory( module_dir, name );
    }
  }


  // ------------------------------------------------------------------
  /// Attempt loading algorithm implementations from all plugin modules in dir
  /**
   * If the given path is not a valid directory, we emit a warning message
   * and return without doing anything else.
   */
  void load_modules_in_directory( path_t dir_path, std::string name = std::string() )
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
    bfs::directory_iterator dir_it( dir_path );
    while ( dir_it != bfs::directory_iterator() )
    {
      bfs::directory_entry const e = *dir_it;

      // Accept this file as a module to check if it has the correct library
      // suffix and matches a provided module name if one was provided.

      if ( ( ST::GetFilenameLastExtension( e.path().string() ) == shared_library_suffix ) &&
           ( ( name.size() == 0 ) || ( ST::GetFilenameWithoutExtension( e.path().string() ) == name ) ) )
      {
        // Check that we're looking a file
        if ( e.status().type() == bfs::regular_file )
        {
          register_from_module( e.path().string() );
        }
        else
        {
          LOG_WARN( m_logger, "Encountered a directory entry " << e.path() <<
                    " which ends with the expected suffix, but is not a file" );
        }
      }

      ++dir_it;
    }
  } // load_modules_in_directory


  // ------------------------------------------------------------------
  /// Find and execute algorithm impl registration call-back in the module
  /**
   * If the expected registration function is found, it is executed. If not,
   * the library is closed and nothing further is performed as we assume this
   * plugis just didn't provide any algorithm implementation extensions.
   *
   * \param module_path Filesystem path to the module library file to load.
   * \returns True of the module was loaded and used in some manner (i.e. still
   *          loaded). False if the module could not be loaded there was no
   *          successful interfacing.
   *
   * TODO: Just use exception handing instead of return codes and booleans.
   */
  bool register_from_module( path_t module_path )
  {
    LOG_DEBUG( m_logger, "Loading plugins from module file: " << module_path );

    //
    // Attempting module load
    //
    library_t library = DL::OpenLibrary( module_path.c_str() );
    if ( ! library )
    {
      LOG_WARN( m_logger, "Failed to open module library " << module_path <<
                 " (error: " << DL::LastError() << ")" );
      return false; // TODO: Throw exception here?
    }

    //
    // Attempt to load each available interface here
    //
    // If interface function not found, we assume this plugin doesn't provide
    // any algorithm implementations and close the library. We otherwise keep
    // it open if we are going to use things from it.
    function_t register_func = NULL;

    // Get our entry symbol
    register_func =  DL::GetSymbolAddress( library, register_function_name.c_str() );
    if ( 0 == register_func )
    {
      LOG_DEBUG( m_logger, "Failed to find algorithm impl registration function \""
                 << register_function_name  << "\"" );
      DL::CloseLibrary( library );
      return false;
    }

    register_impls_func_t const register_impls =
      reinterpret_cast< register_impls_func_t > ( register_func );

    if ( ( *register_impls )( registrar::instance() ) > 0 )
    {
      LOG_WARN( m_logger, "Algorithm implementation registration failed for one or "
                "more algorithms in plugin module, possibly due to duplicate "
                "registration: " << module_path );
      // TODO: Throw exception here?
      DL::CloseLibrary( library );
      return false;
    }
    else
    {
      LOG_DEBUG( m_logger, "Module registration complete" );

      // Adding module name to registered list. Note that existing
      // modules will be replaced.
      std::string key = ST::GetFilenameName( ST::GetFilenameWithoutLastExtension( module_path ) );
      registered_modules_[key] = module_path;
    }

    return true;
  } // register_from_module


  // ------------------------------------------------------------------
  /// Get the list of registered modules names
  std::vector< std::string > registered_module_names() const
  {
    std::vector< std::string > r_vec;

    for ( registered_modules_t::value_type p : registered_modules_ )
    {
      r_vec.push_back( p.first );
    }
    return r_vec;
  }
};


// ===========================================================================
// PluginManager Implementation
// ---------------------------------------------------------------------------

/// Private constructor
algorithm_plugin_manager
::algorithm_plugin_manager()
  : impl_( new impl() )
{
  // craft default search paths. Order of elements in the path has
  // some effect on how modules are looked up.

  // Check env variable for path specification
  const char * env_ptr = kwiversys::SystemTools::GetEnv( environment_variable_name );
  if ( 0 != env_ptr )
  {
    LOG_INFO( impl_->m_logger, "Adding path \"" << env_ptr << "\" from environment" );
    std::string const extra_module_dirs(env_ptr);

    // Split supplied path into separate items using PATH_SEPARATOR_CHAR as delimiter
    ST::Split( extra_module_dirs, impl_->m_search_paths, PATH_SEPARATOR_CHAR );
  }

  ST::Split( default_module_paths, impl_->m_search_paths, PATH_SEPARATOR_CHAR );
}


/// Private destructor
algorithm_plugin_manager
::~algorithm_plugin_manager()
{
  delete this->impl_;
}


// ------------------------------------------------------------------
/// Access singleton instance of this class
algorithm_plugin_manager&
algorithm_plugin_manager
::instance()
{
  static std::mutex local_lock;          // synchronization lock

  if (0 != s_instance)
  {
    return *s_instance;
  }

  std::lock_guard<std::mutex> lock(local_lock);
  if (0 == s_instance)
  {
    // create new object
    s_instance = new algorithm_plugin_manager();
  }

  return *s_instance;
}


// ------------------------------------------------------------------
/// (Re)Load plugin libraries found along current search paths
void
algorithm_plugin_manager
::register_plugins( std::string name )
{
  // Search for libraries to dlopen for algorithm registration
  // call-back.
  LOG_DEBUG( this->impl_->m_logger, "Dynamically loading plugin impls" );
  this->impl_->load_from_search_paths( name );
}


// ------------------------------------------------------------------
/// Add an additional directory to search for plugins in.
void
algorithm_plugin_manager
::add_search_path( path_t dirpath )
{
  this->impl_->m_search_paths.push_back( dirpath );
}


// ------------------------------------------------------------------
/// Get the list currently registered module names.
std::vector< std::string >
algorithm_plugin_manager
::registered_module_names() const
{
  return this->impl_->registered_module_names();
}

} } // end namespace
