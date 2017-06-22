/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "track_oracle_core.h"
#include <map>


using std::pair;

namespace kwiver {
namespace track_oracle {

//
// Track oracle methods
//

template< typename T >
field_handle_type
track_oracle_core
::create_element( const element_descriptor& e )
{
  return track_oracle_core::get_instance().create_element<T>( e );
}

template< typename T >
T&
track_oracle_core
::get_field( oracle_entry_handle_type track, field_handle_type field )
{
  return track_oracle_core::get_instance().get_field<T>( track, field );
}

template< typename T >
pair< bool, T >
track_oracle_core
::get( const oracle_entry_handle_type& row, const field_handle_type& field )
{
  return track_oracle_core::get_instance().get<T>( row, field );
}

template< typename T >
void
track_oracle_core
::remove_field( oracle_entry_handle_type row, field_handle_type field )
{
  track_oracle_core::get_instance().remove_field<T>( row, field );
}

template< typename T >
oracle_entry_handle_type
track_oracle_core
::lookup( field_handle_type field, const T& val, domain_handle_type domain )
{
  return track_oracle_core::get_instance().lookup<T>( field, val, domain );
}

} // ...track_oracle
} // ...kwiver

