// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for plugin manager.
 */

#include "plugin_manager.h"

#include <vital/kwiver-include-paths.h>
#include <vital/logger/logger.h>
#include <vital/plugin_loader/plugin_filter_category.h>
#include <vital/plugin_loader/plugin_filter_default.h>

#include <kwiversys/SystemTools.hxx>

#include <mutex>

namespace kwiver {
namespace vital {

namespace { // anonymous

typedef kwiversys::SystemTools ST;

static char const* environment_variable_name( "KWIVER_PLUGIN_PATH" );
static std::string const register_function_name = std::string( "register_factories" );

// Default module directory locations. Values defined in CMake configuration.
static std::string const default_module_paths = std::string( DEFAULT_MODULE_PATHS );
static std::string const shared_library_suffix = std::string( SHARED_LIB_SUFFIX );

} // end anonymous namespace

// ---- Static ----
plugin_manager* plugin_manager::s_instance( 0 );

// ==================================================================
class plugin_manager::priv
{
public:
  priv()
    : m_loader( new plugin_loader( register_function_name, shared_library_suffix ) )
    , m_logger( kwiver::vital::get_logger( "vital.plugin_manager" ) )
  {
    // Add the default filter which checks for duplicate plugins
    plugin_filter_handle_t filt = std::make_shared<kwiver::vital::plugin_filter_default>();
    m_loader->add_filter( filt );
  }

  plugin_types m_loaded; // bitmask of modules loaded
  std::unique_ptr< plugin_loader > m_loader; // the real loader object
  kwiver::vital::logger_handle_t m_logger;

  path_list_t m_search_paths;
};

// ==================================================================
plugin_manager&
plugin_manager::
instance()
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
    s_instance = new plugin_manager();
  }

  return *s_instance;
}

// ------------------------------------------------------------------
plugin_manager::
plugin_manager()
  : m_priv( new priv() )
{

  // Add search paths
  // Craft default search paths. Order of elements in the path has
  // some effect on how modules are looked up.

  // Check env variable for path specification
  add_path_from_environment( environment_variable_name );

  // Add the built-in search path
  ST::Split( default_module_paths, m_priv->m_search_paths, PATH_SEPARATOR_CHAR );
#ifdef CMAKE_INTDIR
  for ( auto& p : m_priv->m_search_paths )
  {
    ST::ReplaceString( p, "$<CONFIGURATION>", CMAKE_INTDIR );
  }
#endif

  // Add paths to the real loader
  m_priv->m_loader->add_search_path( m_priv->m_search_paths );
}

plugin_manager
::~plugin_manager()
{ }

// ------------------------------------------------------------------
void
plugin_manager
::load_all_plugins( plugin_types types )
{
  types &= ~m_priv->m_loaded;

  if ( types )
  {
    path_list_t dirpath;
    auto search_path = this->search_path();

    // Process each directory in the path and add the directories for
    // the selected plugins.
    for ( auto const& p : search_path )
    {
      // Load plugins from directories based on the options specified.
      if ( types & plugin_type::PROCESSES )
      {
        // load processes
        std::string dir_name = p + "/processes";
        dirpath.push_back( dir_name );
      }

      if ( types & plugin_type::ALGORITHMS )
      {
        // load arrows
        std::string dir_name = p + "/algorithms";
        dirpath.push_back( dir_name );
      }

      if ( types & plugin_type::APPLETS )
      {
        // load applets
        std::string dir_name = p + "/applets";
        dirpath.push_back( dir_name );
      }

      if ( types & plugin_type::EXPLORER )
      {
        // load plugin explorer stuff
        std::string dir_name = p + "/plugin_explorer";
        dirpath.push_back( dir_name );
      }

      if ( types & plugin_type::OTHERS )
      {
        // load everything else
        std::string dir_name = p + "/modules";
        dirpath.push_back( dir_name );
      }

      if ( types & plugin_type::LEGACY )
      {
        // just use the raw directory form the list
        std::string dir_name = p;
        dirpath.push_back( dir_name );
      }
    } // end for

#undef SELECT

    // Load modules from path list
    m_priv->m_loader->load_plugins( dirpath );
    m_priv->m_loaded |= types;
  }
}

// ------------------------------------------------------------------
void plugin_manager::
load_plugins( path_list_t const& dirpath )
{
    m_priv->m_loader->load_plugins( dirpath );
}

// ------------------------------------------------------------------
void plugin_manager::
add_search_path( path_t const& dirpath )
{
  path_list_t path_list;

  ST::Split( dirpath, path_list, PATH_SEPARATOR_CHAR );

  m_priv->m_loader->add_search_path( path_list );
}

// ------------------------------------------------------------------
void plugin_manager::
add_search_path( path_list_t const& dirpath )
{
  m_priv->m_loader->add_search_path( dirpath );
}

// ------------------------------------------------------------------
void plugin_manager::
add_path_from_environment( std::string env_var)
{
  // Check env variable for path specification
  const char * env_ptr = kwiversys::SystemTools::GetEnv( env_var );
  if ( 0 != env_ptr )
  {
    LOG_DEBUG( logger(), "Adding path(s) \"" << env_ptr << "\" from environment" );
    std::string const extra_module_dirs(env_ptr);

    // Split supplied path into separate items using PATH_SEPARATOR_CHAR as delimiter
    ST::Split( extra_module_dirs, m_priv->m_search_paths, PATH_SEPARATOR_CHAR );
  }
  else
  {
    LOG_DEBUG( logger(), "No additional paths on " << env_var );
  }
}

// ------------------------------------------------------------------
path_list_t const& plugin_manager::
search_path() const
{
  return m_priv->m_loader->get_search_path();
}

// ------------------------------------------------------------------
plugin_factory_handle_t plugin_manager::
add_factory( plugin_factory* fact )
{
  return m_priv->m_loader->add_factory( fact );
}

// ------------------------------------------------------------------
plugin_factory_vector_t const& plugin_manager::
get_factories( std::string const& type_name )
{
  return m_priv->m_loader->get_factories( type_name );
}

// ------------------------------------------------------------------
plugin_map_t const& plugin_manager::
plugin_map()
{
  return m_priv->m_loader->get_plugin_map();
}

// ------------------------------------------------------------------
std::vector< std::string > plugin_manager::
file_list()
{
  return m_priv->m_loader->get_file_list();
}

// ------------------------------------------------------------------
void plugin_manager::
reload_plugins()
{
  m_priv->m_loaded = plugin_types{};
  m_priv->m_loader.reset( new plugin_loader( register_function_name, shared_library_suffix ) );

  // Add paths to the real loader
  m_priv->m_loader->add_search_path( m_priv->m_search_paths );

  load_all_plugins();
}

// ------------------------------------------------------------------
bool plugin_manager::
is_module_loaded( std::string const& name) const
{
  return m_priv->m_loader->is_module_loaded( name );
}

// ------------------------------------------------------------------
void plugin_manager::
mark_module_as_loaded( std::string const& name )
{
  m_priv->m_loader->mark_module_as_loaded( name );
}

// ------------------------------------------------------------------
std::map< std::string, std::string > const& plugin_manager::
module_map() const
{
  return m_priv->m_loader->get_module_map();
}

// ------------------------------------------------------------------
kwiver::vital::logger_handle_t plugin_manager::
logger()
{
  return m_priv->m_logger;
}

// ------------------------------------------------------------------
kwiver::vital::plugin_loader*
plugin_manager::
get_loader()
{
  return m_priv->m_loader.get();
}

} } // end namespace kwiver
