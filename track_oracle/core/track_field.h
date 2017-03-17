/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_FIELD_H
#define INCL_TRACK_FIELD_H

#include <string>
#include <utility>
#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/track_field_base.h>
#include <track_oracle/core/track_field_io_proxy.h>
#include <track_oracle/data_terms/data_term_tmp_utils.h>

// This class inherits from track_field, and supplies the data type
// to allow casting.  It is NOT stateful with regards to the row.

// In v1.5, this is THE primary way to get to data columns (element_stores)
// in the backend.


namespace kwiver {
namespace track_oracle {

class track_field_host;

template<typename T> class track_field;
template<typename T> std::ostream& operator<<( std::ostream& os, const track_field<T>& f);

template< typename T >
class track_field: public track_field_base
{
  friend std::ostream& operator<< <> ( std::ostream& os, const track_field<T>& f );

public:

  // our data type is either identical to T (old-style) or contained
  // within T's typedef (new-style data_term)

  typedef typename data_term_traits< is_data_term< T >::value, T >::Type Type;

  // old-style ctors
  explicit track_field( const std::string& field_name );
  track_field( const std::string& field_name, track_field_host* h );

  // new-style data-term ctors
  track_field();
  explicit track_field( track_field_host* h );

  track_field( const track_field<T>& other );
  track_field<T>& operator=( const track_field<T>& other );
  virtual ~track_field() {}

  // soon: explicit ctors with element_descriptors, rather than field names,
  // to support non-adhoc fields

  Type operator()( const oracle_entry_handle_type& row_handle ) const;
  Type& operator()( const oracle_entry_handle_type& row_handle );

  Type operator()( void ) const;
  Type& operator()( void );

  void remove_at_row( const oracle_entry_handle_type& row );

  oracle_entry_handle_type lookup( const Type& val, domain_handle_type domain );

  // special lookup for track_fields associated with a Frame (i.e. allow
  // a track to search its frames for a value)
  oracle_entry_handle_type lookup( const Type& val, const track_handle_type& src_track );

  bool exists( const oracle_entry_handle_type& row_handle ) const;
  virtual bool exists( void ) const;

  std::pair< bool, Type > get( oracle_entry_handle_type row_handle ) const;

  virtual track_field<T>* clone() const;

  virtual void copy_value( const oracle_entry_handle_type& src,
                           const oracle_entry_handle_type& dst ) const;

  track_field_io_proxy<Type> io() const;
  track_field_io_proxy<Type> io( const oracle_entry_handle_type& row_handle) const;
  track_field_io_proxy<Type> io_fmt( const Type& val ) const;

private:
  field_handle_type lookup_or_create_element_store( const std::string& name );

};

} // ...track_oracle
} // ...kwiver

#endif
