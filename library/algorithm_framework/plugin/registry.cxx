// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "plugin_factory.h"
#include "registry.h"

#include <viame/algorithm_framework/exceptions/plugin.h>
#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/util/demangle.h>

#include <sstream>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// @brief Plugin manager private implementation.
///
class registry_impl
{
public:
  registry_impl() = default;
  ~registry_impl() = default;

  // Map from interface name to vector of plugin_factory instances.
  // For consistency, "interface name" refers to the name resulting from
  // `get_interface_name<T>()`.
  plugin_map_t m_plugin_map;

  /// \brief Maps module name to the library that registered it.
  ///
  /// This map is used to keep track of which modules have registered, and
  /// is what every registration function guards on.
  plugin_module_map_t m_module_map;

  /// The library whose registration function is running.
  ///
  /// The generated registry sets it around each call, so that a factory can
  /// say where it came from. It used to be the path of the file the loader
  /// had just `dlopen`ed, which is where PLUGIN_FILE_NAME came from.
  std::string m_registering_library;
}; // end class registry_impl

// ----------------------------------------------------------------------------
registry
::registry()
  : m_logger( kwiver::vital::get_logger( "vital.registry" ) ),
    m_impl( new registry_impl() )
{}

registry
::~registry() = default;

// Factory Stuff ===============================================================
/// @brief Load all known modules.
plugin_factory_vector_t const&
registry
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
registry
::add_factory( plugin_factory* fact )
{
  plugin_factory_handle_t fact_handle( fact );

  // Where the factory came from, for the duplicate diagnostics below.
  fact->add_attribute(
    plugin_factory::PLUGIN_ORIGIN_LIBRARY,
    m_impl->m_registering_library );

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
      std::string interf, inst, name, prev_lib;
      afact->get_attribute( plugin_factory::INTERFACE_TYPE, interf );
      afact->get_attribute( plugin_factory::CONCRETE_TYPE, inst );
      afact->get_attribute( plugin_factory::PLUGIN_NAME, name );
      afact->get_attribute(
        plugin_factory::PLUGIN_ORIGIN_LIBRARY, prev_lib );

      if( ( interface_type == interf ) && ( plugin_name == name ) )
      {
        std::stringstream str;
        if( concrete_type == inst )
        {
          // EXACTLY the same concrete type is being registered.
          // Only log if the paths are different.
          if( prev_lib != m_impl->m_registering_library )
          {
            str << "Factory for \"" << interface_type << "\" : \""
                << demangle( concrete_type ) <<
              "\" already has been registered by "
                << prev_lib << ".  This factory from "
                << m_impl->m_registering_library << " will not be registered."
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
              << "\") was registered by \"" << prev_lib << "\"."
              << "The current type being registered (\"" << demangle( inst )
              << "\") is being registered by \""
              << m_impl->m_registering_library << "\".";
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
                                            << "\" from library: " <<
      m_impl->m_registering_library );

  return fact_handle;
}

// Map Accessors ===============================================================
plugin_map_t const&
registry
::get_plugin_map() const
{
  return m_impl->m_plugin_map;
}

// ------------------------------------------------------------------
bool
registry
::is_module_loaded( std::string const& name ) const
{
  return ( 0 != m_impl->m_module_map.count( name ) );
}

// ------------------------------------------------------------------
void
registry
::mark_module_as_loaded( std::string const& name )
{
  m_impl->m_module_map.insert(
    std::pair< std::string, std::string >(
      name,
      m_impl->m_registering_library ) );
}

// ----------------------------------------------------------------------------
plugin_module_map_t const&
registry
::get_module_map() const
{
  return m_impl->m_module_map;
}

// ----------------------------------------------------------------------------
void
registry
::set_registering_library( std::string const& name )
{
  m_impl->m_registering_library = name;
}

} // namespace vital

}   // end namespace
