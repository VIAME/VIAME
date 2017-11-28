/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_ORACLE_CORE_H
#define INCL_TRACK_ORACLE_CORE_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <string>
#include <vector>
#include <iostream>
#include <utility>

#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/element_descriptor.h>

namespace kwiver {
namespace track_oracle {

/// Interface to the back-end for handling tracks; clients should use track_base / track_field.
///
/// Tracks and frames are referred to by their oracle_entry_handles.
///
/// In v1.0, a data column field was fully described at the API level by (name, type).
///
/// In v1.5, we move to element_descriptors, which replace "name" in the API.
///

class track_oracle_core_impl;
class element_store_base;

class TRACK_ORACLE_EXPORT track_oracle_core
{
public:

  //
  // introducing new elements (aka inserting new columns)
  //

  static std::vector<field_handle_type> get_all_field_handles();
  static field_handle_type lookup_by_name( const std::string& name );
  static element_descriptor get_element_descriptor( field_handle_type f );
  static const element_store_base* get_element_store_base( field_handle_type f );
  static element_store_base* get_mutable_element_store_base( field_handle_type f );
  template< typename T > static field_handle_type create_element( const element_descriptor& e );

  static bool field_has_row( oracle_entry_handle_type row, field_handle_type field );
  template< typename T > static T& get_field( oracle_entry_handle_type track, field_handle_type field );
  template< typename T > static oracle_entry_handle_type lookup( field_handle_type field, const T& val, domain_handle_type domain );
  static oracle_entry_handle_type get_next_handle();
  template< typename T > static std::pair< bool, T > get( const oracle_entry_handle_type& row, const field_handle_type& field );

  template< typename T > static void remove_field( oracle_entry_handle_type row, field_handle_type field );
  static std::vector< field_handle_type > fields_at_row( oracle_entry_handle_type row );
  static std::vector< std::vector< field_handle_type > > fields_at_rows( const std::vector<oracle_entry_handle_type> rows );

  static handle_list_type get_domain( domain_handle_type domain );
  static domain_handle_type lookup_domain( const std::string& domain_name, bool create_if_not_found );
  static domain_handle_type create_domain();
  static domain_handle_type create_domain( const handle_list_type& handles );
  static bool release_domain( const domain_handle_type& domain );
  static bool is_domain_defined( const domain_handle_type& domain );
  static bool add_to_domain( const handle_list_type& handles, const domain_handle_type& domain );
  static bool add_to_domain( const track_handle_type& track, const domain_handle_type& domain );
  static bool set_domain( const handle_list_type& handles, domain_handle_type domain );

  static track_handle_list_type generic_to_track_handle_list( const handle_list_type& handles );
  static frame_handle_list_type generic_to_frame_handle_list( const handle_list_type& handles );
  static handle_list_type track_to_generic_handle_list( const track_handle_list_type& handles );
  static handle_list_type frame_to_generic_handle_list( const frame_handle_list_type& handles );

  static frame_handle_list_type get_frames( const track_handle_type& t );
  static void set_frames( const track_handle_type& t, const frame_handle_list_type& f );
  static size_t get_n_frames( const track_handle_type& t );

  static bool write_kwiver( std::ostream& os, const track_handle_list_type& tracks );
  static bool write_csv( std::ostream& os, const track_handle_list_type& tracks, bool csv_v1_semantics = false );

  static csv_handler_map_type get_csv_handler_map( const std::vector< std::string >& headers );

  static bool clone_nonsystem_fields( const track_handle_type& src,
                                      const track_handle_type& dst );
  static bool clone_nonsystem_fields( const frame_handle_type& src,
                                      const frame_handle_type& dst );
  static bool clone_nonsystem_fields( const oracle_entry_handle_type& src,
                                      const oracle_entry_handle_type& dst );
private:
  track_oracle_core( const track_oracle_core& );  // no cpctor
  track_oracle_core& operator=( const track_oracle_core& ); // no op=

  static track_oracle_core_impl& get_instance();
  static track_oracle_core_impl* impl;
};

} // ...track_oracle
} // ...kwiver

#endif
