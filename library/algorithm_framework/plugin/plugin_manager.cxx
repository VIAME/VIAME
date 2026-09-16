// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation for plugin manager.

#include "plugin_manager.h"

#include <viame/algorithm_framework/logger/logger.h>

#include <exception>
#include <mutex>
#include <vector>

namespace viame {

namespace {

// The registration functions of the libraries that were linked in rather than
// left in loadable modules. Each adds itself as its library loads, which is
// before `main`, so this is a function-local static rather than a namespace
// one -- it has to exist before the first caller, whatever order the loader
// runs the initialisers in.
std::vector< void ( * )( registry&, plugin_manager::plugin_types ) >&
static_registrars()
{
  static std::vector<
    void ( * )( registry&, plugin_manager::plugin_types ) > registrars;
  return registrars;
}

} // end anonymous namespace

// ----------------------------------------------------------------------------
void
plugin_manager
::add_static_registrar(
  void ( *registrar )( registry&, plugin_types ) )
{
  static_registrars().push_back( registrar );
}

// ----------------------------------------------------------------------------
// ---- Static ----
plugin_manager* plugin_manager::s_instance( nullptr );

// ----------------------------------------------------------------------------
class plugin_manager::priv
{
public:
  priv()
    : m_registry( new registry() ),
      m_logger( viame::get_logger( "viame_algorithm_framework.plugin_manager" ) )
  {}

  plugin_types m_loaded; // bitmask of kinds registered
  std::unique_ptr< registry > m_registry; // the registry itself
  viame::logger_handle_t m_logger;

  // Run the registration functions of the libraries linked into this
  // process. There is no file to find and no order to discover: the list is
  // the build's, and every function guards on its own module name. The mask
  // is the caller's, so a registrar whose kind was not asked for is skipped
  // -- which is what keeps applet dispatch from paying for the rest.
  void
  register_builtins( plugin_types types )
  {
    for( auto const registrar : static_registrars() )
    {
      registrar( *m_registry, types );
    }
  }
};

// Singleton Instance Accessor =================================================
plugin_manager&
plugin_manager
::instance()
{
  static std::mutex local_lock;          // synchronization lock

  if( s_instance != nullptr )
  {
    return *s_instance;
  }

  std::lock_guard< std::mutex > lock( local_lock );
  if( s_instance == nullptr )
  {
    // create new object
    s_instance = new plugin_manager();
  }

  return *s_instance;
}

// Protected construct/destruct ================================================
plugin_manager
::plugin_manager()
  : m_priv( new priv() )
{}

plugin_manager
::~plugin_manager()
{}

// Loading Plugins ===========================================================
void
plugin_manager
::load_all_plugins( plugin_types types )
{
  types &= ~m_priv->m_loaded;

  if( types )
  {
    m_priv->register_builtins( types );

    m_priv->m_loaded |= types;
  }
}

// Deprecated? =================================================================

// ----------------------------------------------------------------------------
plugin_factory_handle_t
plugin_manager
::add_factory( plugin_factory* fact )
{
  return m_priv->m_registry->add_factory( fact );
}

// Protected ===================================================================

// ----------------------------------------------------------------------------
plugin_factory_vector_t const&
plugin_manager
::get_factories( std::string const& type_name )
{
  return m_priv->m_registry->get_factories( type_name );
}

// ----------------------------------------------------------------------------
plugin_map_t const&
plugin_manager
::plugin_map()
{
  return m_priv->m_registry->get_plugin_map();
}

// ----------------------------------------------------------------------------
void
plugin_manager
::reload_all_plugins()
{
  m_priv->m_loaded = plugin_types{};
  m_priv->m_registry.reset( new registry() );

  load_all_plugins();
}

// ----------------------------------------------------------------------------
bool
plugin_manager
::is_module_loaded( std::string const& name ) const
{
  return m_priv->m_registry->is_module_loaded( name );
}

// ----------------------------------------------------------------------------
void
plugin_manager
::mark_module_as_loaded( module_t const& name )
{
  m_priv->m_registry->mark_module_as_loaded( name );
}

std::vector< std::string >
plugin_manager
::_impl_names( std::string const& interface_type_name ) const
{
  auto const& plugin_map = m_priv->m_registry->get_plugin_map();

  // There might not be any registered implementations for the given interface.
  // * If there is an interface key in the map but the value is an empty vector,
  //   an empty vector of strings will still be returned.
  if( !plugin_map.count( interface_type_name ) )
  {
    return {};
  }

  auto const& fact_vec = plugin_map.at( interface_type_name );
  std::vector< std::string > plugin_name_vec;
  std::string cur_name;
  for( auto const& fact : fact_vec )
  {
    // A plugin factory should *always* have a PLUGIN_NAME since it is a
    // required part of adding a factory, however technically the registrant
    // has the ability to "unset" it, so let's be cautious and guard against
    // that. In this case we choose
    if( !fact->get_attribute( plugin_factory::PLUGIN_NAME, cur_name ) ||
        cur_name.empty() )
    {
      plugin_name_vec.emplace_back( "<UNNAMED>" );
    }
    else
    {
      plugin_name_vec.push_back( cur_name );
    }
  }
  return plugin_name_vec;
}

// ----------------------------------------------------------------------------
std::map< std::string, std::string > const&
plugin_manager
::module_map() const
{
  return m_priv->m_registry->get_module_map();
}

// Private =====================================================================

// ----------------------------------------------------------------------------
viame::logger_handle_t
plugin_manager
::logger()
{
  return m_priv->m_logger;
}

// ----------------------------------------------------------------------------
viame::registry*
plugin_manager
::get_registry()
{
  return m_priv->m_registry.get();
}

} // namespace viame
