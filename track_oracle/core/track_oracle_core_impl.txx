/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "track_oracle_core_impl.h"
#include <typeinfo>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <limits>
#include <algorithm>
#include <mutex>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::runtime_error;
using std::string;

namespace kwiver {
namespace track_oracle {

//
// value is true if the template parameter  can be initialized
// with zero (to give a default value)
//

template< typename T >
struct is_initializable_with_zero
{
  typedef class dummy{ char dummy_vals[2]; } yes_type;
  typedef char no_type;

  static yes_type test( int );

  static no_type test( ... );

  static T PhonyMakeT();

  static const bool value = (sizeof( test(PhonyMakeT())) == sizeof( yes_type ));
};

template< bool T_can_default_with_zero, typename T >
struct default_value_handler
{
  static T default_type_value()
  {
    return T(0);
  }
};

template<typename T >
struct default_value_handler< false, T >
{
  static T default_type_value()
  {
    return T();
  }
};


template< typename T >
field_handle_type
track_oracle_core_impl
::unlocked_create_element( const element_descriptor& e )
{
  // first double-check the name is available
  field_handle_type f = this->unlocked_lookup_by_name( e.name );
  if (f != INVALID_FIELD_HANDLE)
  {
    ostringstream oss;
    oss << "Duplicate creation of field named '" << e.name << "'\n";
    throw runtime_error( oss.str() );
  }

  // create entry in name pool; assign field handle
  f = ++this->field_count;
  this->name_pool[ e.name ] = f;
  // create entry in element pool
  element_store<T> *es = new element_store<T>( e );
  es->set_default_value( default_value_handler< is_initializable_with_zero<T>::value, T>::default_type_value() );
  this->element_pool[ f ] = es;

  // all done
  return f;
}

template< typename T >
field_handle_type
track_oracle_core_impl
::create_element( const element_descriptor& e )
{
  std::lock_guard< std::mutex > lock( this->api_lock );
  return this->unlocked_create_element<T>( e );
}

template< typename T >
pair< map<oracle_entry_handle_type, T>*, T >
track_oracle_core_impl
::lookup_table( field_handle_type field )
{
  map< field_handle_type, element_store_base* >::iterator probe = this->element_pool.find( field );
  if (probe == this->element_pool.end())
  {
    throw runtime_error( "Lost an element pool for a field?" );
  }
  element_store<T>* es_ptr = dynamic_cast< element_store<T>* >( probe->second );
  if ( ! es_ptr )
  {
    ostringstream oss;
    const element_descriptor& d = probe->second->get_descriptor();
    string my_typeid_str = typeid( static_cast<T*>(0) ).name();
    oss << "Table lookup type mismatch: field " << field << " is '" << d.name << "' type "
        << d.typeid_str << " but requested as a " << my_typeid_str << "\n";
    LOG_ERROR( main_logger, "About to throw exception '" << oss.str() << "'" );
    throw runtime_error( oss.str() );
  }
  pair< map< oracle_entry_handle_type, T>*, T> ret( &es_ptr->storage, es_ptr->get_default_value() );
  return ret;
}

template< typename T >
void
track_oracle_core_impl
::remove_field( oracle_entry_handle_type row, field_handle_type field )
{
  std::lock_guard< std::mutex > lock( this->api_lock );
  pair< map< oracle_entry_handle_type, T >*, T > probe = this->lookup_table<T>( field );
  probe.first->erase( row );
}


template< typename T >
T&
track_oracle_core_impl
::unlocked_get_field( oracle_entry_handle_type track, field_handle_type field )
{
  pair< map< oracle_entry_handle_type, T >*, T > probe = this->lookup_table<T>( field );
  map< oracle_entry_handle_type, T >& data_column = *(probe.first);
  size_t before = data_column.size();
  T& ref = data_column[ track ];
  if ( before != data_column.size() )
  {
    ref = probe.second;
  }
  return ref;
}

template< typename T >
T&
track_oracle_core_impl
::get_field( oracle_entry_handle_type track, field_handle_type field )
{
  std::lock_guard< std::mutex > lock( this->api_lock );
  return this->unlocked_get_field<T>( track, field );
}

template< typename T >
pair< bool, T >
track_oracle_core_impl
::get( oracle_entry_handle_type row, field_handle_type field )
{
  std::lock_guard< std::mutex > lock( this->api_lock );
  const pair< map< oracle_entry_handle_type, T >*, T> f_probe = this->lookup_table<T>( field );
  typename map< oracle_entry_handle_type, T >::const_iterator probe = f_probe.first->find( row );
  return
    ( probe == f_probe.first->end() )
    ? make_pair( false, f_probe.second )
    : make_pair( true, probe->second );
}


template< typename T >
oracle_entry_handle_type
track_oracle_core_impl
::lookup( field_handle_type field, const T& val, domain_handle_type domain )
{
  std::lock_guard< std::mutex > lock( this->api_lock );
  pair< map< oracle_entry_handle_type, T >*, T > probe = this->lookup_table<T>( field );
  map< oracle_entry_handle_type, T >& data_column = *(probe.first);
  if ( domain == DOMAIN_ALL )
  {
    for (typename map< oracle_entry_handle_type, T>::const_iterator i = data_column.begin();
         i != data_column.end();
         ++i )
    {
      if ( i->second == val )
      {
        return i->first;
      }
    }
  }
  else
  {
    map< domain_handle_type, handle_list_type >::iterator i = domain_pool.find( domain );
    if ( i != domain_pool.end() )
    {
      const handle_list_type& handle_list = i->second;
      for (unsigned j=0; j < handle_list.size(); ++j )
      {
        typename map< oracle_entry_handle_type, T>::const_iterator h_probe
          = data_column.find( handle_list[j] );
        if ( ( h_probe != data_column.end() ) && ( h_probe->second == val ))
        {
          return handle_list[j];
        }
      } // ...all the domain handles
    } // ...if the domain is valid
  } // ... for the domain

  return INVALID_ROW_HANDLE;
}

} // ...track_oracle
} // ...kwiver
