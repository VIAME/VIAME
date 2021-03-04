// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_PLUGIN_FACTORY_H
#define KWIVER_VITAL_PLUGIN_FACTORY_H

#include <vital/plugin_loader/vital_vpm_export.h>

#include <vital/noncopyable.h>
#include <vital/exceptions/plugin.h>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <stdexcept>
#include <typeinfo>
#include <memory>

namespace kwiver {
namespace vital {

class plugin_manager;
class plugin_factory;

typedef std::shared_ptr< plugin_factory >         plugin_factory_handle_t;
typedef std::vector< plugin_factory_handle_t >    plugin_factory_vector_t;

// ==================================================================
/**
 * @brief Abstract base class for plugin factory.
 *
 */
class VITAL_VPM_EXPORT plugin_factory
  : public std::enable_shared_from_this< plugin_factory >
  , private kwiver::vital::noncopyable
{
public:
  virtual ~plugin_factory();

  // This is the list of the global attributes that are available to
  // all customers. It is not required to have all attributes
  // present. Applications can use additional attributes that are
  // specific to the application in the application wrapper for this
  // plugin factory/manager. Do not add local scope attributes to this
  // list.
  static const std::string INTERFACE_TYPE;
  static const std::string CONCRETE_TYPE;
  static const std::string PLUGIN_FILE_NAME;
  static const std::string PLUGIN_CATEGORY;
  static const std::string PLUGIN_PROCESS_PROPERTIES;

  // User settable
  static const std::string PLUGIN_NAME;
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
  plugin_factory& add_attribute( std::string const& attr, std::string const& val );

  /**
   * @brief Returns object of registered type.
   *
   * This method returns an object of the template type if
   * possible. The type of the requested object must match the
   * interface type for this factory. If not, an exception is thrown.
   *
   * @return Object of registered type.
   * @throws kwiver::vital::plugin_factory_type_creation_error
   */
  template <class T>
  T* create_object()
  {
    // See if the type requested is the type we support.
    if ( typeid( T ).name() != m_interface_type )
    {
      std::stringstream str;
      str << "Can not create object of requested type: " <<  typeid( T ).name()
          <<"  Factory created objects of type: " << m_interface_type;
      VITAL_THROW( kwiver::vital::plugin_factory_type_creation_error, str.str() );
    }

    // Call derived class to create concrete type object
    T* new_object = reinterpret_cast< T* >( create_object_i() );
    if ( 0 == new_object )
    {
      std::stringstream str;

      str << "plugin_factory:: Unable to create object of type "
          << typeid( T ).name();
      VITAL_THROW( kwiver::vital::plugin_factory_type_creation_error, str.str() );
    }

    return new_object;
  }

  //@{
  /**
   * @brief Iterate over all attributes
   *
   * @param f Factory object
   */
  template < class T > void for_each_attr( T& f )
  {
    for( auto val : m_attribute_map )
    {
      f( val.first, val.second );
    }
  }

  template < class T > void for_each_attr( T const& f ) const
  {
    for( auto const val : m_attribute_map )
    {
      f( val.first, val.second );
    }
  }

  //@}

protected:
  plugin_factory( std::string const& itype);

  std::string m_interface_type;

private:
  // Method to create concrete object
  virtual void* create_object_i() { return 0; }

  typedef std::map< std::string, std::string > attribute_map_t;
  attribute_map_t m_attribute_map;
};

// ----------------------------------------------------------------
/**
 * @brief Factory for concrete class objects.
 *
 * @tparam T Type of the concrete class created.
 */
template< class T >
class plugin_factory_0
  : public plugin_factory
{
public:
  /**
   * @brief Create concrete factory object
   *
   * @param itype Name of the interface type
   */
  plugin_factory_0( std::string const& itype )
    : plugin_factory( itype )
  {
    // Set concrete type of factory
    this->add_attribute( CONCRETE_TYPE, typeid( T ).name() );
  }

  virtual ~plugin_factory_0() = default;

protected:
  virtual void* create_object_i()
  {
    void * new_obj = new T();
    return new_obj;
  }
};

} } // end namespace

// ==================================================================
// Support for adding factories

#define ADD_FACTORY( interface_T, conc_T)                               \
  add_factory( new kwiver::vital::plugin_factory_0< conc_T >( typeid( interface_T ).name() ) )

#endif /* KWIVER_VITAL_PLUGIN_FACTORY_H */
