/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_csv.h"

#include <track_oracle/file_formats/track_csv/track_csv.h>
#ifdef KWIVER_ENABLE_TRACK_MGRS
#include <track_oracle/track_scorable_mgrs/scorable_mgrs_data_term.h>
#endif

#include <track_oracle/utils/tokenizers.h>
#include <track_oracle/core/element_store_base.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/core/state_flags.h>
#include <track_oracle/data_terms/data_terms.h>

#include <vul/vul_timer.h>
#include <vul/vul_file.h>
#include <vul/vul_sprintf.h>

#include <cctype>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );


using std::map;
using std::string;
using std::vector;
using std::ostringstream;
using std::ifstream;
using std::istream;
using std::ofstream;
using std::ostream;
using std::runtime_error;

namespace // anon
{
using namespace ::kwiver::track_oracle;

void
parse_csv_row( oracle_entry_handle_type row,
               const map<string, string>& header_value_map,
               const vector< bool >& value_present_flags,
               const csv_handler_map_type& handlers )
{
  for (csv_handler_map_cit i = handlers.begin(); i != handlers.end(); ++i)
  {
    // do we have the values for this element?  (Allow partial
    // instances and let the data type complain)
    size_t value_count = 0;
    const csv_header_index_type& header_indices = i->second;
    for (size_t j=0; j<header_indices.size(); ++j)
    {
      if (value_present_flags[ header_indices[j] ]) ++value_count;
    }

    // skip this element if no values present
    if (value_count == 0) continue;

    element_store_base* b = track_oracle_core::get_mutable_element_store_base( i->first );
    b->read_csv_to_row( row, header_value_map );
  }
}


struct redundant_object_helper_type
{
  // The old CSV reader would synthesize redundant fields for
  // world_location (if world_{x,y} were present) and obj_location (if
  // obj_{x,y} were present).
  //
  // For each object, if the source values are present (world_z is optional)
  // then copy over into local header-value maps and parse directly into the
  // row.

  redundant_object_helper_type()
  {
    field_handle_type world_location_fh = track_oracle_core::lookup_by_name( dt::tracking::world_location::c.name );
    this->world_location_store = track_oracle_core::get_mutable_element_store_base( world_location_fh );
    field_handle_type obj_location_fh = track_oracle_core::lookup_by_name( dt::tracking::obj_location::c.name );
    this->obj_location_store = track_oracle_core::get_mutable_element_store_base( obj_location_fh );
    field_handle_type bounding_box_fh = track_oracle_core::lookup_by_name( dt::tracking::bounding_box::c.name );
    this->bounding_box_store = track_oracle_core::get_mutable_element_store_base( bounding_box_fh );
    field_handle_type track_uuid_fh = track_oracle_core::lookup_by_name( dt::tracking::track_uuid::c.name );
    this->track_uuid_store = track_oracle_core::get_mutable_element_store_base( track_uuid_fh );
  }

  void apply_at_row( oracle_entry_handle_type row,
                     const map< string, string>& header_value_map )
  {

    //
    // world location
    //

    // check for world_{xyz}
    vector< map<string,string>::const_iterator > world_xyz_probes;
    world_xyz_probes.push_back( header_value_map.find( "world_x" ));
    world_xyz_probes.push_back( header_value_map.find( "world_y" ));
    world_xyz_probes.push_back( header_value_map.find( "world_z" ));

    bool have_world_xy =
      ( (world_xyz_probes[0] != header_value_map.end()) &&
        (! world_xyz_probes[0]->second.empty() ) &&
        (world_xyz_probes[1] != header_value_map.end()) &&
        (! world_xyz_probes[1]->second.empty() ));

    if ( have_world_xy )
    {
      map<string, string> local_hv_map;
      local_hv_map[ "world_location_x" ] = world_xyz_probes[0]->second;
      local_hv_map[ "world_location_y" ] = world_xyz_probes[1]->second;
      local_hv_map[ "world_location_z" ] =
        ((world_xyz_probes[2] == header_value_map.end()) || (world_xyz_probes[2]->second.empty()))
        ? "0.0"
        : world_xyz_probes[2]->second;
      this->world_location_store->read_csv_to_row( row, local_hv_map );
    }

    //
    // object location
    //

    vector< map<string,string>::const_iterator  > obj_xy_probes;
    obj_xy_probes.push_back( header_value_map.find( "obj_x" ));
    obj_xy_probes.push_back( header_value_map.find( "obj_y" ));
    bool have_obj_xy =
      ( (obj_xy_probes[0] != header_value_map.end()) &&
        (! obj_xy_probes[0]->second.empty()) &&
        (obj_xy_probes[1] != header_value_map.end()) &&
        (! obj_xy_probes[1]->second.empty()) );

    if ( have_obj_xy )
    {
      map<string, string> local_hv_map;
      local_hv_map[ "obj_location_x" ] = obj_xy_probes[0]->second;
      local_hv_map[ "obj_location_y" ] = obj_xy_probes[1]->second;
      this->obj_location_store->read_csv_to_row( row, local_hv_map );
    }

    //
    // bounding box
    //

    vector< map<string,string>::const_iterator > bbox_probes;
    bbox_probes.push_back( header_value_map.find( "obj_bbox_ul_x" ));
    bbox_probes.push_back( header_value_map.find( "obj_bbox_ul_y" ));
    bbox_probes.push_back( header_value_map.find( "obj_bbox_lr_x" ));
    bbox_probes.push_back( header_value_map.find( "obj_bbox_lr_y" ));
    bool have_bbox = true;
    for (size_t i=0; i<bbox_probes.size(); ++i)
    {
      have_bbox =
        have_bbox
        && (bbox_probes[i] != header_value_map.end())
        && ( ! bbox_probes[i]->second.empty() );
    }
    if ( have_bbox )
    {
      map<string, string> local_hv_map;
      local_hv_map[ "bounding_box_ul_x" ] = bbox_probes[0]->second;
      local_hv_map[ "bounding_box_ul_y" ] = bbox_probes[1]->second;
      local_hv_map[ "bounding_box_lr_x" ] = bbox_probes[2]->second;
      local_hv_map[ "bounding_box_lr_y" ] = bbox_probes[3]->second;
      this->bounding_box_store->read_csv_to_row( row, local_hv_map );
    }

    //
    // unique_id
    //

    map<string, string>::const_iterator unique_id_probe = header_value_map.find( "unique_id" );
    if ( ( unique_id_probe != header_value_map.end() )
         && (! unique_id_probe->second.empty() ))
    {
      map<string, string> local_hv_map;
      local_hv_map[ "track_uuid" ] = unique_id_probe->second;
      this->track_uuid_store->read_csv_to_row( row, local_hv_map );
    }

  }

private:
  element_store_base* world_location_store;
  element_store_base* obj_location_store;
  element_store_base* bounding_box_store;
  element_store_base* track_uuid_store;
};

} // anon


namespace kwiver {
namespace track_oracle {


file_format_csv
::file_format_csv(): file_format_base(TF_CSV, "Generic CSV")
{
  this->globs.push_back("*.csv");
  this->globs.push_back("*.kwcsv");

  // create a "useless" track_field for world_gcs to get it
  // in the element pool, so it can supply its default value
  volatile track_field< dt::tracking::world_gcs > gcs;

#ifdef KWIVER_ENABLE_TRACK_MGRS
  // likewise, create a token MGRS position field so it
  // ends up in the element pool
  volatile track_field< dt::tracking::mgrs_pos > mgrs;
#endif

  // ...and state flags
  volatile track_field< dt::utility::state_flags > state_flags;

  // ...and track_uuid
  volatile track_field< dt::tracking::track_uuid> uuid_field;

  // ...and world_z, to help maintain CSV 1 compatability
  volatile track_field< dt::tracking::world_z> world_z_field;

  // ...and event type and probability... Time to refactor this.
  volatile track_field< dt::events::event_id > ev_id_field;
  volatile track_field< dt::events::event_type > ev_type_field;
  volatile track_field< dt::events::event_probability > ev_prob_field;
  volatile track_field< dt::events::source_track_ids > src_trk_ids_field;

  volatile track_field< dt::detection::detection_id > det_id_field;
}

file_format_csv
::~file_format_csv()
{
}

bool
file_format_csv
::inspect_file(string const& fn) const
{
  ifstream ifs(fn.c_str());
  if ( ! ifs ) return false;

  vector<string> values;
  if (!csv_tokenizer::get_record(ifs, values) && !ifs.eof()) return false;

  csv_handler_map_type m = track_oracle_core::get_csv_handler_map( values );

  // discard the ignored headers
  m.erase( INVALID_FIELD_HANDLE );

  // anybody left?
  return ( ! m.empty() );
}

bool
file_format_csv
::read(string const& fn, track_handle_list_type& tracks) const
{
  ifstream ifs(fn.c_str());
  return ifs && this->internal_stream_read(ifs, vul_file::size( fn ), tracks);
}

bool
file_format_csv
::read(istream& is, track_handle_list_type& tracks) const
{
  return this->internal_stream_read( is, 0, tracks );
}

bool
file_format_csv
::internal_stream_read( istream& is, size_t file_size, track_handle_list_type& tracks ) const
{
  // Read header
  vector<string> headers;
  if (!csv_tokenizer::get_record(is, headers) && !is.eof())
  {
    LOG_ERROR( main_logger,"Failed to read CSV header record");
    return false;
  }

  // Old-style CSVs did not emit the 'track_sequence/parent_track'
  // special headers; instead, 'external_id' was used.
  // Detect new-style / old-style by the presence / absence
  // of the track_sequence header. (frame_sequence may be
  // missing if no frame data is recorded.)

  vector<string>::iterator track_sequence_probe = std::find( headers.begin(), headers.end(), "_track_sequence" );
  vector<string>::iterator external_id_probe = std::find( headers.begin(), headers.end(), "external_id" );

  if (( track_sequence_probe == headers.end() ) &&
      ( external_id_probe == headers.end() ))
  {
    LOG_ERROR( main_logger, "CSV headers do not contain either external_id or _track_sequence; cannot parse" );
    return false;
  }

  bool new_style_flag = ( track_sequence_probe != headers.end() );
  size_t sequence_index =
    new_style_flag
    ? track_sequence_probe - headers.begin()
    : external_id_probe - headers.begin();

  // verify either _parent_track (new-style) or frame_number (old-style)
  // is present
  string frame_tag =
    new_style_flag
    ? "_parent_track"
    : "frame_number";
  vector<string>::iterator frame_probe = find( headers.begin(), headers.end(), frame_tag );
  if ( frame_probe == headers.end() )
  {
    LOG_ERROR( main_logger, "CSV headers do not contain expected frame tag '" << frame_tag << "'; cannot parse" );
    return false;
  }
  size_t frame_index = frame_probe - headers.begin();

  {
    string style = (new_style_flag) ? "new" : "old";
    LOG_INFO( main_logger, "Detected " << style << "-style CSV" );
  }

  //
  // Read records
  //

  csv_handler_map_type m = track_oracle_core::get_csv_handler_map( headers );

  // key is either _track_sequence (new-style) or external-id (old-style)
  map< size_t, track_handle_type > sequence_to_handle_map;
  track_handle_type current_track_handle;


  // various legacy hacks
  redundant_object_helper_type redundant_object_helper;

  // use the schema object mostly as a proxy to create tracks and frames
  track_csv_type track;

  vul_timer timer;
  for (unsigned long long record = 1; is; ++record)
  {
    if (timer.real() > 5 * 1000)
    {
      ostringstream oss;
      oss << "Read " << record << " lines; " << tracks.size() << " tracks";
      if ( file_size > 0)
      {
        oss << vul_sprintf( "; %02.2f%% of file", 100.0*is.tellg()/file_size );
      }
      LOG_INFO( main_logger, oss.str() );
      timer.mark();
    }

    vector< string > values;
    csv_tokenizer::get_record(is, values);

    if (!is && !is.eof())
    {
      LOG_ERROR( main_logger,"I/O error reading CSV at record " << record);
      return false;
    }
    if (values.empty())
    {
      if (is.eof()) break;
      continue;
    }

    if ( values.size() != headers.size() )
    {
      LOG_ERROR( main_logger, "Line " << record << " has " << values.size() << " entries; expected "
                 << headers.size() << "; skipping" );
      continue;
    }

    // If old-style, frame_number is set only on frames.
    // If new-style, _parent_track is set only on frames.

    oracle_entry_handle_type this_row;

    bool row_is_track = ( values[ frame_index ].empty() );
    if (row_is_track)
    {
      unsigned this_id = std::stoul( values[ sequence_index ] );

      // do we need to create a new row?
      if ( sequence_to_handle_map.find( this_id ) == sequence_to_handle_map.end() )
      {
        current_track_handle = track.create();
        sequence_to_handle_map[ this_id ] = current_track_handle;
        tracks.push_back( current_track_handle );
      }
      this_row = current_track_handle.row;
    }
    else
    {
      // old-style: parent_probe should look for external_id
      // new-style: parent_probe should look for _parent_track
      const string& parent_probe_target_string =
        new_style_flag
        ? values[frame_index]
        : values[sequence_index];

      unsigned this_frame_belongs_to = std::stoul( parent_probe_target_string );
      map< size_t, track_handle_type >::const_iterator parent_probe =
        sequence_to_handle_map.find( this_frame_belongs_to );
      if (parent_probe == sequence_to_handle_map.end() )
      {
        // If we get here, it's because we're a 'frame' row referring to a non-existent
        // parent track.  If we're a single line CSV, say
        //
        // latitude,longitude,velocity_x,velocity_y,iso_time,external_id,frame_number
        // 35,-116,14.083146,25.669183,2014-06-25T00:00:00.000000Z,0,0
        //
        // we can recover IF we've embedded the external_id on every frame.  Try that
        // before bailing out.

        if (! new_style_flag) // may be too restrictive?
        {
          track_handle_type new_track_handle = track.create();
          sequence_to_handle_map[ this_frame_belongs_to ] = new_track_handle;
          tracks.push_back( new_track_handle );
          parent_probe = sequence_to_handle_map.find( this_frame_belongs_to );
          track_field< dt::tracking::external_id > id_field;
          id_field( new_track_handle.row ) = this_frame_belongs_to;
        }
        else
        {
          LOG_ERROR( main_logger, "FATAL: record " << record << " belongs to " << this_frame_belongs_to
                     << ", which has no handle" );
          throw runtime_error( "logic error in csv reader" );
        }
      }
      this_row = track( parent_probe->second ).create_frame().row;
    }


    // create this row's header-value map
    map< string, string > header_value_map;
    vector< bool > value_present_flags;
    for ( size_t i=0; i<headers.size(); ++i)
    {
      header_value_map[ headers[i] ] = values[i];
      value_present_flags.push_back( ! values[i].empty() );
    }

    // parse the row
    parse_csv_row( this_row, header_value_map, value_present_flags, m );

    // world location / object_location hack
    redundant_object_helper.apply_at_row( this_row, header_value_map );

  } // ...for each row

  return true;
}

bool
file_format_csv
::write(const string& fn, const track_handle_list_type& tracks) const
{
  ofstream ofs(fn.c_str());
  return ofs && this->write(ofs, tracks);
}

bool
file_format_csv
::write(ostream& os, const track_handle_list_type& tracks) const
{
  return track_oracle_core::write_csv( os, tracks );
}

} // ...track_oracle
} // ...kwiver

