/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_ORACLE_CORE_IMPL_H
#define INCL_TRACK_ORACLE_CORE_IMPL_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <map>
#include <string>
#include <mutex>

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/element_store.h>

namespace  // anon
{

struct csv_element
{
  // the field handle being emitted, or INVALID_FIELD_HANDLE for bookkeeping field
  // such as _track_sequence or _parent_track
  ::kwiver::track_oracle::field_handle_type fh;

  // true if the field defines a default value which should be emitted if not explicitly
  // set on the row (e.g. world_gcs)
  bool emit_default_if_absent;

  // true if the element is defined for the track, false if frame.
  // used to distinguish between the two bookkeeping fields (hack!)
  bool is_track;

  csv_element( ::kwiver::track_oracle::field_handle_type f, bool e, bool t )
    : fh(f), emit_default_if_absent(e), is_track(t)
  {}

  csv_element()
    : fh(::kwiver::track_oracle::INVALID_FIELD_HANDLE), emit_default_if_absent(false), is_track(false)
  {}
};

} // anon

namespace kwiver {
namespace track_oracle {

struct xml_output_helper;

class TRACK_ORACLE_EXPORT track_oracle_core_impl
{
private:
  // element_pool and name_pool are essentially inverses of each other.
  // They get added to only when clients request an element_store which
  // doesn't yet exist.  The only way a request for a column can fail
  // is if you want to re-use a name and it turns out to already exist
  // with a different type.  We can detect that immediately if the request
  // is made with a full element_descriptor, but not if the track_field
  // ctor does a lookup on name only to see if it has to then create
  // with a full element_descriptor.

  std::map< field_handle_type, element_store_base* > element_pool;
  std::map< std::string, field_handle_type > name_pool;
  unsigned field_count;
  unsigned row_count;

  std::map< domain_handle_type, handle_list_type > domain_pool;
  std::map< std::string, domain_handle_type > domain_names;
  domain_handle_type last_domain_allocated;

  track_oracle_core_impl( const track_oracle_core_impl& ); // no cpctor
  track_oracle_core_impl operator=( const track_oracle_core_impl& ); // no op=

  // Given a field (column) index, return the map of that column's data
  // Always create if not found; also return the table's default value
  template< typename T > std::pair< std::map<oracle_entry_handle_type, T>*, T > lookup_table( field_handle_type field );

  // giant mutex for a blunt approach to thread safety
  mutable std::mutex api_lock;

  // throws if the field doesn't exist
  field_handle_type lookup_required_field( const std::string& fn ) const;

  // unlocked versions of API calls which are also called internally
  template< typename T > field_handle_type unlocked_create_element( const element_descriptor& e );
  field_handle_type unlocked_lookup_by_name( const std::string& name ) const;
  template< typename T > T& unlocked_get_field( oracle_entry_handle_type track, field_handle_type field );
  frame_handle_list_type unlocked_get_frames( const track_handle_type& t );
  bool unlocked_field_has_row( oracle_entry_handle_type row, field_handle_type field ) const;
  void emit_pool_as_kwiver( std::ostream& os, const std::string& indent, oracle_entry_handle_type row ) const;
  void emit_pool_as_csv( std::ostream& os,
                         const std::vector< csv_element >& output_order,
                         oracle_entry_handle_type row,
                         size_t sequence_number,
                         bool is_track,
                         bool csv_v1_semantics ) const;
  void get_csv_columns( const oracle_entry_handle_type& row, std::map< field_handle_type, std::vector< std::string> >& m ) const;

  void unlocked_remove_row( oracle_entry_handle_type row );

public:
  friend struct xml_output_helper;

  track_oracle_core_impl();

  template< typename T > field_handle_type create_element( const element_descriptor& e );

  std::vector< field_handle_type > get_all_field_handles() const;
  field_handle_type lookup_by_name( const std::string& name ) const;

  element_descriptor get_element_descriptor( field_handle_type f ) const;
  const element_store_base* get_element_store_base( field_handle_type f ) const;
  element_store_base* get_mutable_element_store_base( field_handle_type f ) const;

  std::map< element_descriptor, field_handle_type > get_fields() const;

  // Given a field handle, does the row (track or frame) have an entry?
  bool field_has_row( oracle_entry_handle_type track, field_handle_type field );

  // Given a {track,frame}/field (row/column) location, return a ref to the value
  // Type T must have a default constructor in case the value doesn't exist, hmm
  // ...this will be tricky to make thread safe.
  template< typename T > T& get_field( oracle_entry_handle_type track, field_handle_type field );

  // return a pair <bool, T>; bool is true if T exists, false otherwise
  template< typename T > std::pair< bool, T > get( oracle_entry_handle_type row, field_handle_type field );

  // Get the handle for the next track or frame
  oracle_entry_handle_type get_next_handle();

  // delete the track (including frames and __frame_list)
  bool remove_track( const track_handle_type& track );

  // delete field at this row
  template< typename T> void remove_field( oracle_entry_handle_type row, field_handle_type field );

  // return fields which have entries at this row
  std::vector< field_handle_type > fields_at_row( oracle_entry_handle_type row ) const;

  // return fields which have entries for these rows (faster than one-at-time calls to above)
  std::vector< std::vector< field_handle_type > > fields_at_rows( const std::vector< oracle_entry_handle_type >& rows ) const;

  // return the row containing the value-- return first if multiple; probably should alert somehow
  template< typename T> oracle_entry_handle_type lookup( field_handle_type field, const T& val, domain_handle_type domain );

  // return the frames associated with the handle without requiring a specific schema
  frame_handle_list_type get_frames( const track_handle_type& t );

  // hammer the list of frames as the associated frames for the track (may leak until a GC is in t_o)
  void set_frames( const track_handle_type& t, const frame_handle_list_type& f);

  // get the number of frames
  size_t get_n_frames( const track_handle_type& t );

  handle_list_type get_domain( domain_handle_type domain );
  domain_handle_type lookup_domain( const std::string& domain_name, bool create_if_not_found );
  domain_handle_type create_domain();
  domain_handle_type create_domain( const handle_list_type& handles );
  bool is_domain_defined( const domain_handle_type& domain );
  bool release_domain( const domain_handle_type& domain );
  bool add_to_domain( const handle_list_type& handles, const domain_handle_type& domain );
  bool add_to_domain( const track_handle_type& track, const domain_handle_type& domain );
  bool set_domain( const handle_list_type& handles, domain_handle_type domain );

  // emit the list of tracks as kwiver XML
  bool write_kwiver( std::ostream& os, const track_handle_list_type& tracks );

  // emit the list of tracks as CSV
  bool write_csv( std::ostream& os, const track_handle_list_type& tracks, bool csv_v1_semantics );

  // given a list of headers, return the handlers
  csv_handler_map_type get_csv_handler_map( const std::vector< std::string >& headers );

  // Copy all fields which are not marked SYSTEM from the src row to the dst row
  bool clone_nonsystem_fields( const oracle_entry_handle_type& src,
                               const oracle_entry_handle_type& dst );
};

} // ...track_oracle
} // ...kwiver

#endif
