// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_ORACLE_ROW_VIEW_H
#define INCL_TRACK_ORACLE_ROW_VIEW_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <vector>
#include <track_oracle/core/track_field_host.h>
#include <track_oracle/core/track_field_base.h>
#include <track_oracle/core/track_field.h>

namespace kwiver {
namespace track_oracle {

// this does nothing but group track fields into a single group.
// Row state delegated to field host.

class TRACK_ORACLE_EXPORT track_oracle_row_view: public track_field_host
{
  friend TRACK_ORACLE_EXPORT std::ostream& operator<<( std::ostream& os, const track_oracle_row_view& r );
private:
  std::vector< track_field_base* > field_list;
  std::vector< bool > this_owns_ptr;
  void reset();

public:
  track_oracle_row_view();
  track_oracle_row_view( const track_oracle_row_view& other );
  track_oracle_row_view& operator=( const track_oracle_row_view& other );
  virtual ~track_oracle_row_view();

  template< typename T> track_field<T>& add_field( const std::string& name );
  bool add_field( track_field_base& b, bool this_owns = false );

  track_handle_type get_row() const;
  bool set_row( const track_handle_type& new_row );
  void remove_row( const track_handle_type& row );

  void copy_values( const oracle_entry_handle_type& src,
                    const oracle_entry_handle_type& dst ) const;

  const track_oracle_row_view& operator()( const track_handle_type& h ) const;

  bool is_complete() const;
  std::vector< field_handle_type > list_missing_elements( const oracle_entry_handle_type& h ) const;
  bool contains_element( const element_descriptor& e ) const;
  bool contains_element( field_handle_type f ) const;
  std::vector< field_handle_type > list_elements() const;

  track_field_base* clone_field_from_element( const element_descriptor& e ) const;

};

} // ...track_oracle
} // ...kwiver

#endif
