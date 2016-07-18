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

#ifndef KWIVER_VITAL_PLUGIN_FACTORY_H
#define KWIVER_VITAL_PLUGIN_FACTORY_H

#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <stdexcept>
#include <typeinfo>
#include <memory>

#include <vital/vital_config.h>
#include <vital/noncopyable.h>
#include <vital/vital_foreach.h>


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
class plugin_factory
  : public std::enable_shared_from_this< plugin_factory >,
    private kwiver::vital::noncopyable
{
public:
  virtual ~plugin_factory();

  // standard set of attributes
  static const std::string INTERFACE_TYPE;
  static const std::string CONCRETE_TYPE;
  static const std::string PLUGIN_FILE_NAME;
  static const std::string PLUGIN_NAME;
  static const std::string PLUGIN_DESCRIPTION;

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
   * @throws std::runtime_error
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
      throw std::runtime_error( str.str() );
    }

    // Call derived class to create concrete type object
    T* new_object = reinterpret_cast< T* >( create_object_i() );
    if ( 0 == new_object )
    {
      std::stringstream str;

      str << "class_loader:: Unable to create object";
      throw std::runtime_error( str.str() );
    }

    return new_object;
  }

  template < class T > void for_each_attr( T& f )
  {
    VITAL_FOREACH( auto val, m_attribute_map )
    {
      f( val.first, val.second );
    }
  }

  template < class T > void for_each_attr( T const& f ) const
  {
    VITAL_FOREACH( auto const val, m_attribute_map )
    {
      f( val.first, val.second );
    }
  }


protected:
  plugin_factory( std::string const& itype);

  std::string m_interface_type;


private:
  // Method to create concrete object
  virtual void* create_object_i() = 0;

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

  virtual ~plugin_factory_0() {}

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
