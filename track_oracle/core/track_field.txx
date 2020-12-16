// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <stdexcept>
#include <iostream>
#include <vector>
#include <sstream>
#include <typeinfo>

#include <track_oracle/core/track_field.h>
#include <track_oracle/core/element_store.h>
#include <track_oracle/core/track_field_output_specializations.h>
#include <track_oracle/core/track_oracle_api_types.h>

using std::make_pair;
using std::ostream;
using std::ostringstream;
using std::pair;
using std::runtime_error;
using std::string;

namespace // anon
{
using namespace ::kwiver::track_oracle;

template< typename Type >
element_store<Type>*
get_element_store( const field_handle_type& fh )
{
  element_store_base* b = track_oracle_core::get_mutable_element_store_base( fh );
  element_store<Type>* typed_b = dynamic_cast< element_store<Type>* >( b );
  if (typed_b == 0)
  {
    // whoops
    element_descriptor ed = track_oracle_core::get_element_descriptor( fh );
    throw runtime_error( "Failed to downcast element store for data term '"+ed.name+"'" );
  }
  return typed_b;
}

} // anon

namespace kwiver {
namespace track_oracle {

//
// template metaprogramming to set custom IO handlers
//

template< bool T_is_a_kwiver_io_base, typename T, typename Type >
struct special_io_handler
{
  static void set_handler( const dt::context& c )
  {
    element_store<Type>* typed_b = get_element_store<Type>( c.fh );
    kwiver_io_base<typename T::Type>* io_handler = new T;
    typed_b->set_io_handler( io_handler );
  }
};

template< typename T, typename Type >
struct special_io_handler< false, T, Type >
{
  static void set_handler( const dt::context& c)
  {
    element_store<Type>* typed_b = get_element_store<Type>( c.fh );
    if ( ! typed_b->get_io_handler() )
    {
      typed_b->set_io_handler( new kwiver_io_base<T>( c.name ) );
    }
  }
};

//
// template metaprogramming to set special default values
// (now without 'typeof')
//

template< typename T >
class is_that_a_class
{
protected:
  typedef class dummy { char dummy_vals[2]; } yes_type;
  typedef char no_type;

  template< typename X >
  static yes_type test( int X::* );

  template< typename X >
  static no_type test( ... );

public:
  static const bool value  = (sizeof(test<T>(0)) != sizeof(no_type));
};

// via http://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Member_Detector
template< typename T>
class class_has_default_value_member
{
protected:
  struct Fallback { int get_default_value; };
  struct Derived : T, Fallback {};

  template < typename U, U > struct Check;

  typedef class dummy { char dummy_vals[2]; } yes_type;
  typedef char no_type;

  template< typename U > static yes_type test( Check<int Fallback::*, &U::get_default_value>* );
  template< typename U > static no_type test( ... );

public:
  static const bool value = (sizeof( test<Derived>(0)) == sizeof( no_type ));

};

template< bool T_is_class, typename T >
struct impl_type_has_default
{
  static const bool value = class_has_default_value_member<T>::value;
};

template< typename T>
struct impl_type_has_default<false, T >
{
  static const bool value = false;
};

template< typename T  >
struct type_has_default_value_member
{
  static const bool value = impl_type_has_default< is_that_a_class<T>::value, T >::value;
};

template< bool T_has_default_value, typename T, typename Type >
struct set_default_value_handler
{
  static void set_handler( element_store<Type>* es_ptr )
  {
    es_ptr->set_default_value( T::get_default_value() );
  }
};

template< typename T, typename Type >
struct set_default_value_handler< false, T, Type >
{
  static void set_handler( element_store<Type>* )
  {
    // do nothing
  }
};

template< typename T >
track_field<T>
::track_field( const string& field_name )
  : track_field_base( field_name, 0 )
{
  this->field_handle = this->lookup_or_create_element_store( field_name );
  dt::context c( field_name, "no description" );
  c.fh = this->field_handle;
  special_io_handler< is_data_term<T>::value, T, T>::set_handler( c );
}

template< typename T >
track_field<T>
::track_field( const string& field_name, track_field_host* h )
  : track_field_base(field_name, h )
{
  this->field_handle = this->lookup_or_create_element_store( field_name );
  dt::context c( field_name, "no description" );
  c.fh = this->field_handle;
  special_io_handler< is_data_term<T>::value, T, T>::set_handler( c );
}

template< typename T >
track_field<T>
::track_field()
  : track_field_base( "", 0 )
{
  dt::context& c = T::c;
  this->name = c.name;
  if ( c.fh == INVALID_FIELD_HANDLE )
  {
    c.fh = this->lookup_or_create_element_store( c.name );
    special_io_handler< is_data_term<T>::value, T, Type>::set_handler( c );
  }
  this->field_handle = c.fh;

}

template< typename T >
track_field<T>
::track_field( track_field_host* h )
  : track_field_base( data_term_traits< true, T >::get_name(), h )
{
  dt::context& c = T::c;
  if ( c.fh == INVALID_FIELD_HANDLE )
  {
    c.fh = this->lookup_or_create_element_store( c.name );
    special_io_handler< is_data_term<T>::value, T, Type>::set_handler( c );
  }
  this->field_handle = c.fh;
}

template< typename T >
track_field<T>
::track_field( const track_field<T>& other )
  : track_field_base( other.get_field_name(), other.host )
{
  this->field_handle = other.get_field_handle();
}

template< typename T >
track_field<T>&
track_field<T>
::operator=( const track_field<T>& other )
{
  track_field_base::operator=( other );
  this->field_handle = other.get_field_handle();
  return *this;
}

template< typename T >
field_handle_type
track_field<T>
::lookup_or_create_element_store( const string& field_name )
{
  //
  // If track_field<T> finds an element store whose name==field_name,
  // AND if its type matches, then we use it.
  //
  // If we find an element_store whose name==field_name, but its type
  // DOESN'T match, then we throw.
  //
  // If we don't find an element_store whose name==field_name, we create
  // one using, for the moment, default fields for e.g. the element role.
  //

  string my_typeid_str = typeid( static_cast<Type*>(0) ).name();
  field_handle_type f = track_oracle_core::lookup_by_name( field_name );

  if ( f == INVALID_FIELD_HANDLE )
  {
    // no column by that name; create one
    f = track_oracle_core::create_element<Type>(
                       element_descriptor(
                         field_name,
                         "no description",
                         my_typeid_str,
                         element_descriptor::ADHOC));
    // if the type is a data term with a default value, set it
    element_store<Type>* es_ptr = get_element_store<Type>( f );
    set_default_value_handler< type_has_default_value_member<T>::value, T, Type >::set_handler( es_ptr );
    return f;
  }

  // we have a column by that name; check the type
  element_descriptor e = track_oracle_core::get_element_descriptor( f );

  if (e.typeid_str != my_typeid_str)
  {
    ostringstream oss;
    oss << "Track field '" << field_name
        << "': tried to create as '" << my_typeid_str
        << "' but already existed as '" << e.typeid_str << "'\n";
    throw runtime_error( oss.str() );
  }

  // types match, return the field handle
  return f;
}

template< typename T >
typename track_field<T>::Type
track_field<T>
::operator()( const oracle_entry_handle_type& row_handle ) const
{
  return track_oracle_core::get_field<Type>( row_handle, field_handle );
}

template< typename T >
typename track_field<T>::Type&
track_field<T>
::operator()( const oracle_entry_handle_type& row_handle )
{
  return track_oracle_core::get_field<Type>( row_handle, field_handle );
}

template< typename T >
typename track_field<T>::Type
track_field<T>
::operator()( void ) const
{
  if ( ! this->host )
  {
    ostringstream oss;
    oss << "Track field '" << this->name << "' dereferenced with no cursor set";
    throw runtime_error( oss.str() );
  }
  if ( ! track_oracle_core::field_has_row( this->host->get_cursor(), field_handle ))
  {
    ostringstream oss;
    oss << "Attempt to read uninitialized field " << this->name << "\n";
    throw runtime_error( oss.str() );
  }
  return track_oracle_core::get_field<Type>( this->host->get_cursor(), field_handle );
}

template< typename T >
typename track_field<T>::Type&
track_field<T>
::operator()( void )
{
  if ( ! this->host )
  {
    ostringstream oss;
    oss << "Track field '" << this->name << "' dereferenced with no cursor set";
    throw runtime_error( oss.str() );
  }
  return track_oracle_core::get_field<Type>( this->host->get_cursor(), field_handle );
}

template< typename T>
void
track_field<T>
::remove_at_row( const oracle_entry_handle_type& row )
{
  track_oracle_core::remove_field<typename track_field<T>::Type>( row, this->field_handle );
}

template< typename T>
oracle_entry_handle_type
track_field<T>
::lookup( const typename track_field<T>::Type& val, domain_handle_type domain )
{
  return track_oracle_core::lookup<Type>( this->field_handle, val, domain );
}

template< typename T>
oracle_entry_handle_type
track_field<T>
::lookup( const typename track_field<T>::Type& val, const track_handle_type& src_track )
{
  frame_handle_list_type frames = track_oracle_core::get_frames( src_track );

  // just do a linear search for now
  for (unsigned i=0; i<frames.size(); ++i)
  {
    if ( (track_oracle_core::field_has_row( frames[i].row, this->field_handle)) &&
         (track_oracle_core::get_field<Type>( frames[i].row, this->field_handle) == val) )
    {
      return frames[i].row;
    }
  }

  // not there!
  return INVALID_ROW_HANDLE;
}

template<typename T>
bool
track_field<T>
::exists( const oracle_entry_handle_type& row ) const
{
  return track_oracle_core::field_has_row( row, this->field_handle );
}

template<typename T>
bool
track_field<T>
::exists( void ) const
{
  if ( ! this->host )
  {
    ostringstream oss;
    oss << "Track field '" << this->name << "' dereferenced with no cursor set";
    throw runtime_error( oss.str() );
  }
  return track_oracle_core::field_has_row( this->host->get_cursor(), this->field_handle );
}

template< typename T >
pair< bool, typename track_field<T>::Type >
track_field<T>
::get( oracle_entry_handle_type row ) const
{
  return track_oracle_core::get<Type>( row, this->field_handle );
}

template< typename T >
track_field<T>*
track_field<T>
::clone() const
{
  return new track_field<T>( *this );
}

template< typename T >
void
track_field<T>
::copy_value( const oracle_entry_handle_type& src,
              const oracle_entry_handle_type& dst ) const
{
  pair< bool, typename track_field<T>::Type > probe = track_oracle_core::get<Type>( src, field_handle );
  if ( probe.first )
  {
    track_oracle_core::get_field<Type>( dst, field_handle ) = probe.second;
  }
}

template< typename T >
track_field_io_proxy<typename track_field<T>::Type>
track_field<T>
::io() const
{
  element_store<Type>* typed_b = get_element_store<Type>( this->field_handle );
  return track_field_io_proxy<Type>( typed_b->get_io_handler(), this->operator()() );
}

template< typename T >
track_field_io_proxy<typename track_field<T>::Type>
track_field<T>
::io( const oracle_entry_handle_type& row_handle) const
{
  element_store<Type>* typed_b = get_element_store<Type>( this->field_handle );
  return track_field_io_proxy<Type>( typed_b->get_io_handler(), this->operator()( row_handle ) );
}

template< typename T >
track_field_io_proxy<typename track_field<T>::Type>
track_field<T>
::io_fmt( const Type& val ) const
{
  element_store<Type>* typed_b = get_element_store<Type>( this->field_handle );
  return track_field_io_proxy<Type>( typed_b->get_io_handler(), val );
}

template< typename T>
ostream& operator<<( ostream& os, const track_field<T>& f ) {
  os << " (" << f.field_handle << ") "
     << f.name;
  try
  {
    const typename track_field<T>::Type& val = f();
    os << " = " << f.io_fmt( val );
  }
  catch (runtime_error const& )
  {
    os << " (no row set)";
  }
  return os;
}

} // ...track_oracle
} // ...kwiver
