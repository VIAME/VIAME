/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "element_store.h"

#include <tinyxml.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t local_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::vector;

namespace kwiver {
namespace track_oracle {

template< typename T >
element_store<T>
::element_store( const element_descriptor& e )
  : element_store_base( e ), default_value_ptr(0), io_base_ptr( new kwiver_io_base<T>( e.name ))
{}

template< typename T >
element_store<T>
::element_store( const element_descriptor& e,
                 kwiver_io_base<T>* io_base )
  : element_store_base( e ), default_value_ptr(0), io_base_ptr( io_base )
{}

template< typename T >
element_store<T>
::~element_store()
{
  delete default_value_ptr;
  delete io_base_ptr;
}

template< typename T>
void
element_store<T>
::set_io_handler( kwiver_io_base<T>* io_base )
{
  delete this->io_base_ptr;
  this->io_base_ptr = io_base;
}

template< typename T>
kwiver_io_base<T>*
element_store<T>
::get_io_handler() const
{
  return this->io_base_ptr;
}

template< typename T >
bool
element_store<T>
::exists( const track_handle_type& h ) const
{
  return this->exists( h.row );
}

template< typename T >
bool
element_store<T>
::exists( const frame_handle_type& h ) const
{
  return this->exists( h.row );
}

template< typename T >
bool
element_store<T>
::exists( const oracle_entry_handle_type& h ) const
{
  typename map<oracle_entry_handle_type, T>::const_iterator p = this->storage.find( h );
  return (p != this->storage.end() );
}

template< typename T >
vector< bool >
element_store<T>
::exists( const vector< oracle_entry_handle_type>& sorted_hlist ) const
{
  vector< bool > ret( sorted_hlist.size(), false );
  if (sorted_hlist.empty())
  {
    return ret;
  }

  size_t ret_index(0), hlist_size( sorted_hlist.size() );
  typename map<oracle_entry_handle_type, T>::const_iterator p = this->storage.lower_bound( sorted_hlist[0] );
  typename map<oracle_entry_handle_type, T>::const_iterator e = this->storage.end();

  bool keep_going = (p != e) && (ret_index < hlist_size);
  while (keep_going)
  {
    //
    // each time through the loop, we either
    //
    // 1) if p->first < sorted_hlist[ret_index], increment p
    //
    // or
    //
    // 2) if p->first == sorted_hlist[ret_index], set return flag and increment both
    //
    // or
    //
    // 3) if p->first > sorted_hlist[ret_index], increment ret_index

    oracle_entry_handle_type left(p->first), right( sorted_hlist[ ret_index ]);
    if (left < right)
    {
      keep_going = (++p != e);
    }
    else if (left == right)
    {
      ret[ ret_index ] = true;
      keep_going = ((++p != e) && (++ret_index < hlist_size));
    }
    else
    {
      keep_going = (++ret_index < hlist_size);
    }

  }

  return ret;
}

template< typename T>
bool
element_store<T>
::copy_value( const oracle_entry_handle_type& src,
              const oracle_entry_handle_type& dst )
{
  if ( src == INVALID_ROW_HANDLE )
  {
    LOG_ERROR( local_logger, "copy_value for " << this->get_descriptor().name << ": uninitialized source" );
    return false;
  }
  if ( dst == INVALID_ROW_HANDLE )
  {
    LOG_ERROR( local_logger, "copy_value for " << this->get_descriptor().name << ": uninitialized destination" );
    return false;
  }

  typename map<oracle_entry_handle_type, T>::iterator p = this->storage.find( src );
  if ( p == this->storage.end() ) return false;

  this->storage.insert( make_pair( dst, p->second ));
  return true;
}


template< typename T >
bool
element_store<T>
::remove( const track_handle_type& h )
{
  return this->remove( h.row );
}

template< typename T >
bool
element_store<T>
::remove( const frame_handle_type& h )
{
  return this->remove( h.row );
}

template< typename T >
bool
element_store<T>
::remove( const oracle_entry_handle_type& h )
{
  size_t c = this->storage.erase( h );
  return (c != 0);
}

template< typename T >
ostream&
element_store<T>
::emit_as_kwiver( ostream& os, const oracle_entry_handle_type& h, const string& indent ) const
{
  typename map<oracle_entry_handle_type, T>::const_iterator p = this->storage.find( h );
  if ( p != this->storage.end() )
  {
    this->io_base_ptr->write_xml( os, indent, p->second );
  }
  return os;
}

template< typename T >
ostream&
element_store<T>
::emit_as_csv( ostream& os, const oracle_entry_handle_type& h, bool emit_default_if_missing ) const
{
  typename map<oracle_entry_handle_type, T>::const_iterator p = this->storage.find( h );
  if ( p != this->storage.end() )
  {
    this->io_base_ptr->to_csv( os, p->second );
  }
  else
  {
    if ( emit_default_if_missing )
    {
      this->io_base_ptr->to_csv( os, this->get_default_value() );
    }
    else
    {
      vector< string > headers = this->csv_headers();
      for (size_t i=0; i<headers.size()-1; ++i)
      {
        os << ",";
      }
    }
  }
  return os;
}

template< typename T >
bool
element_store<T>
::read_kwiver_xml_to_row( const oracle_entry_handle_type& h, const TiXmlElement* e )
{
  T val;
  if ( ! this->io_base_ptr->read_xml( e, val ))
  {
    LOG_ERROR( local_logger, "Couldn't parse instance of " << this->get_descriptor().name << " at row " << e->Row() );
    return false;
  }
  this->storage[ h ] = val;
  return true;
}

template< typename T >
bool
element_store<T>
::read_csv_to_row( const oracle_entry_handle_type& h, const map< string, string>& header_value_map )
{
  T val;
  if ( ! this->io_base_ptr->from_csv( header_value_map, val ))
  {
    LOG_ERROR( local_logger, "Couldn't parse instance of " << this->get_descriptor().name << " from CSV" );
    return false;
  }
  this->storage[ h ] = val;
  return true;
}

template< typename T >
vector<string>
element_store<T>
::csv_headers() const
{
  return this->io_base_ptr->csv_headers();
}

template< typename T >
const T&
element_store<T>
::get_default_value() const
{
  if ( ! this->default_value_ptr )
  {
    throw runtime_error( "No default value set for "+this->get_descriptor().name );
  }
  return *(this->default_value_ptr);
}

template< typename T >
void
element_store<T>
::set_default_value( const T& val)
{
  delete this->default_value_ptr;
  this->default_value_ptr = new T(val);
}

template< typename T >
void
element_store<T>
::set_to_default_value( const oracle_entry_handle_type& h )
{
  this->storage[ h ] = this->get_default_value();
}

//
// Implementation of emit_as_XML() deferred to xml_output_common_classes.cxx and the
// implementation of complex classes such as CUTIC_descriptor.
//

} // ...track_oracle
} // ...kwiver
