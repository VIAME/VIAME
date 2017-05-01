/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef TRACK_ORACLE_BASE_IMPL_H
#define TRACK_ORACLE_BASE_IMPL_H

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <utility>
#include <map>

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/track_oracle_row_view.h>
#include <track_oracle/core/track_oracle_frame_view.h>

namespace kwiver {
namespace track_oracle {

class TRACK_ORACLE_EXPORT track_base_impl
{
  friend TRACK_ORACLE_EXPORT std::ostream& operator<<( std::ostream& os, track_base_impl& track );

protected:
  track_oracle_row_view Track;
  track_oracle_frame_view Frame;


public:
  enum schema_position_type { INVALID, IN_TRACK, IN_FRAME };

  // By default, tracks have no row
  track_base_impl();

  // Delete a frame from this track
  bool remove_frame( const frame_handle_type& row );

  // new track-- move to the next available row
  track_handle_type create();

  // Delete a track (also removes frames)
  void remove_me();

  // new frame
  frame_handle_type create_frame();

  // add frames
  void add_frames( const frame_handle_list_type& frames );

  // for e.g. printing
  const track_oracle_frame_view& frame() const;

  const track_oracle_row_view& track() const;

  bool is_complete( const track_handle_type& row ) const;

  std::vector< field_handle_type > list_missing_elements( const track_handle_type& track ) const;

  bool schema_contains_element( const element_descriptor& e ) const;
  bool schema_contains_element( field_handle_type f ) const;

  std::pair< track_field_base*, schema_position_type > clone_field_from_element( const element_descriptor& e ) const;
  bool add_field_at_position( const std::pair< track_field_base*, schema_position_type >& f );

  std::map< field_handle_type, schema_position_type > list_schema_elements() const;
};

TRACK_ORACLE_EXPORT std::ostream& operator<<( std::ostream& os, track_base_impl& track );

} // ...track_oracle
} // ...kwiver

#endif
