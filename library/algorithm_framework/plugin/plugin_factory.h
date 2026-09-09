// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_PLUGIN_FACTORY_H
#define KWIVER_VITAL_PLUGIN_FACTORY_H

#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include <viame/algorithm_framework/plugin/vital_vpm_export.h>

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/exceptions/plugin.h>
#include <viame/core_types/noncopyable.h>
#include <viame/algorithm_framework/plugin/pluggable.h>
#include <viame/algorithm_framework/util/demangle.h>

namespace kwiver::vital {

class plugin_factory;

typedef std::shared_ptr< plugin_factory >         plugin_factory_handle_t;
typedef std::vector< plugin_factory_handle_t >    plugin_factory_vector_t;

/**
 * Common accessor to get an interface type's name.
 * @tparam T Interface type.
 * @return String name of the interface type.
 */
template < typename T >
std::string
get_interface_name()
{
  // See pluggable.h for static method description.
  // This is intentionally using an accessor vs. typeid(T).name() in order to
  // allow being able to get this information from a python object.
  // (of course, not through this function in that case...)
  static_assert(
    has_interface_name< T >::value,
    "The given interface type must define the static method "
    "`interface_name` in order to know how to get at, or set, "
    "it's factories." );
  return T::interface_name();
}

/// Common accessor to get a concrete type's name.
template < typename T >
std::string
get_concrete_name()
{
  return typeid( T ).name();
}

// ==================================================================

/**
 * @brief Abstract base class for plugin factory.
 *
 * NOTE: Previous implementation formed its creation methods NOT as pure-virtual
 * but with a return-null implementation. It seems that other "categories" of
 * factories hard-overwrite the "creation" function (`from_config` here) with
 * different parameterizations.
 */
class VITAL_VPM_EXPORT plugin_factory
  : public std::enable_shared_from_this< plugin_factory >,
    private kwiver::vital::noncopyable
{
public:
  /// Default constructor
  plugin_factory() = default;
  /// Default destructor
  ~plugin_factory() override = default;

  // This is the list of the global attributes that are available to
  // all customers. It is not required to have all attributes
  // present. Applications can use additional attributes that are
  // specific to the application in the application wrapper for this
  // plugin factory/manager. Do not add local scope attributes to this
  // list.
  static const std::string INTERFACE_TYPE;  // typeid name of the interface type
  static const std::string CONCRETE_TYPE;  // typeid name of the concrete type
  static const std::string PLUGIN_FILE_NAME;  // filesystem path from which this
                                              // factory was registered from.
  static const std::string PLUGIN_NAME;  // Human-readable name for plugin
                                         // implementation.
  static const std::string PLUGIN_CATEGORY;  // like if this is an algo,
                                             // process, etc.
  static const std::string PLUGIN_PROCESS_PROPERTIES;

  // User settable
  static const std::string PLUGIN_DESCRIPTION;
  static const std::string PLUGIN_VERSION;
  static const std::string PLUGIN_MODULE_NAME; // logical module name
  static const std::string PLUGIN_FACTORY_TYPE; // typename of factory class
  static const std::string PLUGIN_AUTHOR;
  static const std::string PLUGIN_ORGANIZATION;
  static const std::string PLUGIN_LICENSE;

  // plugin categories
  static const std::string APPLET_CATEGORY;
  static const std::string PROCESS_CATEGORY;
  static const std::string ALGORITHM_CATEGORY;
  static const std::string CLUSTER_CATEGORY;

  /**
   * @brief Factory encapsulating how to construct concrete types from
   *    configurations.
   *
   * This requires that concrete class types to be factory generated have a
   * static function ``T::from_config(config_block const& cb)`` that returns a
   * std::shared_ptr of a type that descends from pluggable, e.g. could be
   * dynamically cast into pluggable.
   *
   * This method returns an object of the template type if
   * possible. The type of the requested object must match the
   * interface type for this factory. If not, an exception is thrown.
   *
   * @return Object instance of the registered type.
   * @throws kwiver::vital::plugin_factory_type_creation_error
   */
  virtual pluggable_sptr from_config( config_block_sptr const cb ) const = 0;

  /**
   * @brief Populate a config block instance with the default configuration for
   *    the encapsulated type.
   *
   * This requires that the concrete implementation class define a static method
   * `T::get_default_config(config_block & cb)` that assigns into the config
   * block appropriate parameters to, when set property by an outside agent,
   * successfully construct an instance via a call to
   * `plugin_factory::from_config` above.
   *
   * @param cb config block instance to write to.
   */
  virtual void get_default_config( config_block& cb ) const = 0;

  /**
   * @brief Get attribute from factory
   *
   * @param[in] attr Attribute code
   * @param[out] val Value of attribute if present
   *
   * @return \b true if attribute is found; \b false otherwise.
   */
  bool get_attribute( std::string const& attr, std::string& val ) const;

  /**
   * @brief Add attribute to factory
   *
   * This method sets the specified attribute
   *
   * @param attr Attribute name.
   * @param val Attribute value.
   */
  plugin_factory& add_attribute(
    std::string const& attr,
    std::string const& val );

  //@{

  /**
   * @brief Iterate over all attributes some invokable `f`.
   *
   * @param f Some invokable type that takes two string parameters. These
   *    parameters will be the attribute key and value, respectively.
   */
  template < class T >
  void
  for_each_attr( T& f )
  {
    for( auto val : m_attribute_map )
    {
      f( val.first, val.second );
    }
  }

  /**
   * @brief Iterate over all attributes some const invokable `f`.
   *
   * @param f Some invokable type that takes two string parameters. These
   *    parameters will be the attribute key and value, respectively.
   */
  template < class T >
  void
  for_each_attr( T const& f ) const
  {
    for( auto const val : m_attribute_map )
    {
      f( val.first, val.second );
    }
  }

  //@}

protected:
  plugin_factory( std::string const& itype );

  std::string m_interface_type;

private:
  typedef std::map< std::string, std::string > attribute_map_t;

  attribute_map_t m_attribute_map;
};

// ----------------------------------------------------------------

/**
 * @brief Basic concrete factory templated on the concrete type to be created at
 * runtime.
 *
 * This should be used to create a real factory that can create instances of the
 * templated concrete type.
 *
 * @tparam INTERFACE Type of the interface the concrete class is based on.
 * @tparam CONCRETE Type of the concrete class created.
 */
template < class INTERFACE, class CONCRETE >
class concrete_plugin_factory
  : public plugin_factory
{
public:
  static_assert(
    std::is_base_of< pluggable, INTERFACE >::value,
    "The given interface type did not descend from the pluggable "
    "type." );
  static_assert(
    std::is_base_of< INTERFACE, CONCRETE >::value,
    "The given concrete type is not based on the given "
    "interface type." );

  /**
   * @brief Create concrete factory instance.
   *
   * This will pre-populate the interface and concrete type names based on the
   * typeid name of the template type inputs. These names are likely "mangled"
   * and ``kwiver::vital::demangle()`` may be used to try to make such names
   * more readable (if supported).
   */
  explicit concrete_plugin_factory( std::string const& plugin_name )
  {
    // Set some standard attributes
    this->add_attribute( INTERFACE_TYPE, get_interface_name< INTERFACE >() )
      .add_attribute( CONCRETE_TYPE, get_concrete_name< CONCRETE >() )
      .add_attribute( PLUGIN_NAME, plugin_name );
  }

  pluggable_sptr
  from_config( config_block_sptr const cb ) const override
  {
    static_assert(
      has_from_config< CONCRETE >::value,
      "The given concrete type does not implement the "
      "`from_config` static method. See pluggable.h for more "
      "details." );
    return CONCRETE::from_config( cb );
  }

  void
  get_default_config( config_block& cb ) const override
  {
    static_assert(
      has_get_default_config< CONCRETE >::value,
      "The given concrete type does not implement the "
      "`get_default_config` static method. See pluggable.h for "
      "more details." );
    CONCRETE::get_default_config( cb );
  }

  ~concrete_plugin_factory() override = default;
};

} // end namespace

#endif /* KWIVER_VITAL_PLUGIN_FACTORY_H */
