// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#undef TRACK_4676_ENABLED

#include "file_format_manager.h"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <mutex>
#include <vul/vul_string.h>
#include <vul/vul_file.h>

#include <vital/types/timestamp.h>

#include <track_oracle/file_formats/file_format_schema.h>
#include <track_oracle/file_formats/file_format_base.h>

#include <track_oracle/core/state_flags.h>
#include <track_oracle/core/element_store_base.h>

#include <track_oracle/file_formats/track_kw18/file_format_kw18.h>
#ifdef SHAPELIB_ENABLED
#include <track_oracle/file_formats/track_apix/file_format_apix.h>
#endif
#include <track_oracle/file_formats/track_comms_xml/file_format_comms_xml.h>
#include <track_oracle/file_formats/track_kst/file_format_kst.h>
#include <track_oracle/file_formats/track_kwxml/file_format_kwxml.h>
#include <track_oracle/file_formats/track_mitre_xml/file_format_mitre_xml.h>
#include <track_oracle/file_formats/track_xgtf/file_format_xgtf.h>
#include <track_oracle/file_formats/track_vatic/file_format_vatic.h>
#include <track_oracle/file_formats/track_vpd/file_format_vpd_track.h>
#include <track_oracle/file_formats/track_vpd/file_format_vpd_event.h>
#include <track_oracle/file_formats/track_e2at_callout/file_format_e2at_callout.h>
#if KWIVER_ENABLE_KPF
#include <track_oracle/file_formats/track_kpf_geom/file_format_kpf_geom.h>
#endif
#ifdef TRACK_4676_ENABLED
#include <track_oracle/file_formats/track_4676/file_format_4676.h>
#endif
#include <track_oracle/file_formats/track_csv/file_format_csv.h>
#include <track_oracle/file_formats/track_kwiver/file_format_kwiver.h>
#include <track_oracle/core/schema_algorithm.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::ifstream;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::runtime_error;
using std::string;
using std::vector;

::kwiver::track_oracle::file_format_manager_impl* ::kwiver::track_oracle::file_format_manager::impl = 0;

namespace // anon
{
std::mutex instance_lock;
};

namespace kwiver {
namespace track_oracle {

typedef map< file_format_enum, track_base_impl* > schema_map_type;
typedef map< file_format_enum, track_base_impl* >::const_iterator schema_map_cit;

struct file_format_manager_impl
{
  format_map_type formats;             // all the formats we know about
  schema_map_type schemata;          // an instance of each schema

  // giant mutex for a blunt approach to thread safety
  mutable std::mutex api_lock;

  // what formats match the globs?
  vector< file_format_enum > unlocked_globs_match( string fn );
  vector< file_format_enum > globs_match( string fn );

  // what format is this file?  May call inspection routines
  file_format_enum detect_format( const string& fn );

  // get a pointer to the format
  file_format_base* get_format( file_format_enum fmt ) const;

  file_format_manager_impl();
  ~file_format_manager_impl();
};

file_format_manager_impl
::file_format_manager_impl()
{
  std::lock_guard< std::mutex > lock( this->api_lock );

  // register all the formats.
  // when adding a new format, the only things you should need to do
  // are (a) the #include for your format, (b) extend the file_format_enum
  // enumeration, and (c) add a line here inserting an instance of your
  // format into the format map.

  formats[ TF_KW18 ] = new file_format_kw18();
  formats[ TF_XGTF ] = new file_format_xgtf();
  formats[ TF_KWXML ] = new file_format_kwxml();
#ifdef SHAPELIB_ENABLED
  formats[ TF_APIX ] = new file_format_apix();
#endif
  formats[ TF_MITRE_BOX_XML ] = new file_format_mitre_xml();
  formats[ TF_COMMS_XML ] = new file_format_comms_xml();
  formats[ TF_KST ] = new file_format_kst();
  formats[ TF_VATIC ] = new file_format_vatic();
  formats[ TF_VPD_TRACK ] = new file_format_vpd_track();
  formats[ TF_VPD_EVENT ] = new file_format_vpd_event();
  formats[ TF_E2AT_CALLOUT ] = new file_format_e2at_callout();
#ifdef TRACK_4676_ENABLED
  formats[ TF_4676 ] = new file_format_4676();
#endif
  formats[ TF_CSV ] = new file_format_csv();
  formats[ TF_KWIVER ] = new file_format_kwiver();
#ifdef KWIVER_ENABLE_KPF
  formats[ TF_KPF_GEOM ] = new file_format_kpf_geom();
#endif

  // get instances of all the schemas, for introspection
  for (format_map_cit i = formats.begin(); i != formats.end(); ++i)
  {
    schemata[ i->first ] = i->second->schema_instance();
  }
}

vector< file_format_enum >
file_format_manager_impl
::unlocked_globs_match( string fn )
{
  // we downcase here, and typically downcase again in each file_format_base...
  // to do otherwise risks inconsistent behavior between manager and individual
  // file_format_base APIs
  vul_string_downcase( fn );
  vector< file_format_enum > ret;
  for (format_map_cit probe = formats.begin(); probe != formats.end(); ++probe )
  {
    if ( probe->second->filename_matches_globs( fn ))
    {
      ret.push_back( probe->first );
    }
  }
  return ret;
}

vector< file_format_enum >
file_format_manager_impl
::globs_match( string fn )
{
  std::lock_guard< std::mutex > lock( this->api_lock );
  return this->unlocked_globs_match( fn );
}

file_format_enum
file_format_manager_impl
::detect_format( const string& fn )
{
  std::lock_guard< std::mutex > lock( this->api_lock );

  // the assumption here is that multiple formats may match a glob,
  // but only one should pass inspection.  Add a check to verify this.
  // If it ends up that multiple files pass inspection, we may need
  // to add explicit disambiguation logic, such as allowing a filename
  // such as 'foo.txt' to be entered as 'vatic:foo.txt'.
  //

  map< file_format_enum, bool > format_has_been_probed;
  vector< file_format_enum > formats_passing_inspection;
  vector< file_format_enum > glob_matches = this->unlocked_globs_match( fn );

  // if we have glob matches, try those first
  for (size_t i=0; i<glob_matches.size(); ++i)
  {
    format_map_cit probe = formats.find( glob_matches[i] );
    if ( probe == formats.end() ) throw runtime_error( "Logic error: Lost a format???" );
    if ( probe->second->inspect_file( fn ))
    {
      formats_passing_inspection.push_back( glob_matches[i] );
    }
    format_has_been_probed[ glob_matches[i] ] = true;
  }

  // If we have NO formats matching at this point, we go down
  // the rest of them.  (But only if the file exists.)
  if ( formats_passing_inspection.empty() )
  {
    for (format_map_cit probe = formats.begin(); probe != formats.end(); ++probe)
    {
      if ( format_has_been_probed.find( probe->first ) == format_has_been_probed.end() )
      {
        if (probe->second->inspect_file( fn ))
        {
          formats_passing_inspection.push_back( probe->first );
        }
        format_has_been_probed[ probe->first ] = true;
      }
    }
  }

  // How many passed inspection?
  size_t n_passing = formats_passing_inspection.size();

  // If one, good-- return it.
  if ( n_passing == 1 ) return formats_passing_inspection[ 0 ];

  // If none, oh well-- return signalling no match.
  if ( n_passing == 0 ) return TF_INVALID_TYPE;

  // Otherwise, we have a problem.
  ostringstream oss;
  oss << "File '" << fn << "': cannot distinguish between the following " << n_passing << " formats:\n";
  for (size_t i=0; i<n_passing; ++i)
  {
    format_map_cit p = formats.find( formats_passing_inspection[i] );
    if (p == formats.end())
    {
      oss << "... Error!  Lost format " << formats_passing_inspection[i] << "??\n";
    }
    else
    {
      oss << "... " << file_format_type::to_string( p->second->get_format() ) << "\n";
    }
  }
  oss << "Need to implement the file_format_disambiguation scheme (or something similar).\n";

  throw runtime_error( oss.str().c_str() );
}

file_format_base*
file_format_manager_impl
::get_format( file_format_enum fmt ) const
{
  std::lock_guard< std::mutex > lock( this->api_lock );
  format_map_cit probe = formats.find( fmt );
  return
    (probe == formats.end())
    ? 0
    : probe->second;
}

file_format_manager_impl
::~file_format_manager_impl()
{
  std::lock_guard< std::mutex > lock( this->api_lock );
  for (format_map_cit i = formats.begin(); i != formats.end(); ++i)
  {
    delete i->second;
  }
  for (schema_map_cit i = schemata.begin(); i != schemata.end(); ++i)
  {
    delete i->second;
  }
}

file_format_manager_impl&
file_format_manager
::get_instance()
{
  if ( ! file_format_manager::impl )
  {
    std::lock_guard< std::mutex > lock( instance_lock );
    if ( ! file_format_manager::impl )
    {
      file_format_manager::impl = new file_format_manager_impl();
    }
  }
  return *file_format_manager::impl;
}

void
file_format_manager
::initialize()
{
  get_instance();
}

const format_map_type&
file_format_manager
::get_format_map()
{
  return get_instance().formats;
}

file_format_base*
file_format_manager
::get_format( file_format_enum fmt )
{
  return get_instance().get_format( fmt );
 }

file_format_reader_opts_base&
file_format_manager
::options( file_format_enum fmt )
{
  const format_map_type& formats = get_instance().formats;
  format_map_cit probe = formats.find( fmt );
  if (probe == formats.end())
  {
    ostringstream oss;
    oss << "Unimplemented reader or bad format: " << fmt;
    throw runtime_error( oss.str() );
  }
  return probe->second->options();
}

file_format_reader_opts_base&
file_format_manager
::default_options( file_format_enum fmt )
{
  return file_format_manager::options( fmt ).reset();
}

map< file_format_enum, vector< string > >
file_format_manager
::get_all_globs()
{
  map< file_format_enum, vector< string > > ret;
  const format_map_type& formats = get_instance().formats;
  for (format_map_cit probe = formats.begin(); probe != formats.end(); ++probe )
  {
    ret[ probe->first ] = probe->second->format_globs();
  }
  return ret;
}

vector< file_format_enum >
file_format_manager
::globs_match( string fn )
{
  return get_instance().globs_match( fn );
}

file_format_enum
file_format_manager
::detect_format( const string& fn )
{
  return get_instance().detect_format( fn );
}

bool
file_format_manager
::read( const string& fn, track_handle_list_type& tracks )
{
  {
    ifstream is_check( fn.c_str() );
    if ( ! is_check )
    {
      LOG_ERROR( main_logger, "FileFormatManager: File not found: '" << fn << "'" );
      return false;
    }
  }
  if ( vul_file::size( fn ) == 0)
  {
    LOG_WARN( main_logger, "File '" << fn << "' is empty" );
    return true;
  }

  file_format_enum f = get_instance().detect_format( fn );
  if (f == TF_INVALID_TYPE) return false;

  file_format_base* b = get_instance().get_format( f );
  if (! b)
  {
    LOG_ERROR( main_logger, "Logic error: no reader for format " << f << " for '" << fn << "'?");
    return false;
  }

  bool rc = b->read( fn, tracks );
  if (rc)
  {
    file_format_schema_type::record_track_source( tracks, fn, f );
  }
  return rc;
}

bool
file_format_manager
::write( const string& fn,
         const track_handle_list_type& tracks,
         file_format_enum format )
{
  if ( format == TF_INVALID_TYPE )
  {

    // can't call detect_format, because that calls inspection routines,
    // and fn may or may not exist.
    vector< file_format_enum > formats = get_instance().globs_match( fn );
    if ( formats.size() != 1 )
    {
      LOG_ERROR( main_logger, "Can't determine a unique filetype to write '" << fn << "'; found " << formats.size() << " formats" );
      return false;
    }
    format = formats[0];
    LOG_INFO( main_logger, "file_format_manager autodetected format " << file_format_type::to_string( format )
              << " for '" << fn << "'" );
  }

  file_format_base* b = get_instance().get_format( format );
  if ( !b )
  {
    LOG_ERROR( main_logger, "Logic error: no writer for format " << format << " for '" << fn << "'?" );
    return false;
  }

  if ( ! ( b->supported_operations() & FF_WRITE))
  {
    LOG_ERROR( main_logger, "Detected format " << file_format_type::to_string( format ) << " for '"
               << fn << "' does not support writing\n" );
    return false;
  }

  return b->write( fn, tracks );

}

vector< file_format_enum >
file_format_manager
::format_contains_element( const element_descriptor& e )
{
  const schema_map_type& s = get_instance().schemata;
  vector< file_format_enum > formats;
  for (schema_map_cit i = s.begin(); i != s.end(); ++i)
  {
    if (i->second->schema_contains_element( e ))
    {
      formats.push_back( i->first );
    }
  }
  return formats;
}

vector< file_format_enum >
file_format_manager
::format_matches_schema( const track_base_impl& schema)
{
  const schema_map_type& s = get_instance().schemata;
  vector< file_format_enum > formats;
  for (schema_map_cit i = s.begin(); i != s.end(); ++i)
  {
    vector< element_descriptor > missing_fields =
      schema_algorithm::schema_compare( schema, *(i->second) );
    if ( missing_fields.empty() )
    {
      formats.push_back( i->first );
    }
  }
  return formats;
}

pair< track_field_base*, track_base_impl::schema_position_type >
file_format_manager
::clone_field_from_element( const element_descriptor& e )
{

  const schema_map_type& s = get_instance().schemata;
  for (schema_map_cit i = s.begin(); i != s.end(); ++i)
  {
    pair< track_field_base*, track_base_impl::schema_position_type > ret =
      i->second->clone_field_from_element( e );
    if (ret.first) return ret;
  }
  return make_pair( static_cast<track_field_base*>(0), track_base_impl::INVALID );
}

bool
file_format_manager
::write_test_tracks( const string& fn,
                     const csv_handler_map_type& header_map,
                     size_t n_tracks,
                     size_t n_frames_per_track )
{
  // zip through all the formats, building up a union of their schema elements
  map< field_handle_type, track_base_impl::schema_position_type > all_elements;
  const schema_map_type& s = get_instance().schemata;
  for (schema_map_cit i = s.begin(); i != s.end(); ++i)
  {
    map< field_handle_type, track_base_impl::schema_position_type > this_format = i->second->list_schema_elements();
    for (map< field_handle_type, track_base_impl::schema_position_type >::const_iterator j = this_format.begin();
         j != this_format.end();
         ++j)
    {
      // skip system headers
      if (track_oracle_core::get_element_descriptor( j->first ).role == element_descriptor::SYSTEM) continue;

      // make sure the position is valid
      if (j->second == track_base_impl::INVALID)
      {
        LOG_ERROR( main_logger, "Bad schema position for " << track_oracle_core::get_element_descriptor(j->first).name
                   << " in " << file_format_type::to_string( i->first ) );
        return false;
      }
      if (all_elements.find( j->first ) == all_elements.end() )
      {
        all_elements[ j->first ] = j->second;
      }
      else
      {
        if (all_elements[ j->first ] != j->second )
        {
          // prefer track over frame when conflicting
          all_elements[ j->first ] =  track_base_impl::IN_TRACK;
        }
      }

    } // ...for all elements in the format
  } // ...for all formats

  // remove any fields NOT named in the header_map (unless header_map is empty, in which case, keep all)
  if ( ! header_map.empty() )
  {
    LOG_INFO( main_logger, "Before discard: " << all_elements.size() );
    for (map<field_handle_type, track_base_impl::schema_position_type>::iterator i = all_elements.begin();
         i != all_elements.end();
         /* do nothing */)
    {
      if ( header_map.find(i->first) == header_map.end() )
      {
        map<field_handle_type, track_base_impl::schema_position_type>::iterator scoot = i;
        scoot++;
        LOG_INFO( main_logger, "Discarding element " << track_oracle_core::get_element_descriptor( i->first ).name );
        all_elements.erase( i->first );
        i = scoot;
      }
      else
      {
        LOG_INFO( main_logger, "Keeping element " << track_oracle_core::get_element_descriptor( i->first ).name );
        ++i;
      }
    }
    LOG_INFO( main_logger, "After discard: " << all_elements.size() );
  }

  // create track fields for the items we'll set explicitly
  track_field< dt::tracking::external_id > external_id;
  track_field< dt::utility::state_flags > state_flags;
  track_field< dt::tracking::frame_number > frame_number;
  track_field< dt::tracking::timestamp_usecs > timestamp_usecs;
  track_field< dt::tracking::time_stamp > time_stamp;

  // assert track/frame positions for some particular fields
  all_elements[ track_oracle_core::lookup_by_name( dt::tracking::external_id::c.name ) ] = track_base_impl::IN_TRACK;
  all_elements[ track_oracle_core::lookup_by_name( dt::utility::state_flags::c.name ) ] = track_base_impl::IN_FRAME;
  all_elements[ track_oracle_core::lookup_by_name( dt::tracking::frame_number::c.name) ] = track_base_impl::IN_FRAME;
  all_elements[ track_oracle_core::lookup_by_name( dt::tracking::timestamp_usecs::c.name) ] = track_base_impl::IN_FRAME;
  all_elements[ track_oracle_core::lookup_by_name( dt::tracking::time_stamp::c.name) ] = track_base_impl::IN_FRAME;

  // split 'em up
  vector< field_handle_type > track_elements, frame_elements;
  for ( map<field_handle_type, track_base_impl::schema_position_type>::const_iterator i = all_elements.begin();
        i != all_elements.end();
        ++i )
  {
    if ( i->second == track_base_impl::IN_TRACK )
    {
      track_elements.push_back( i->first );
    }
    else
    {
      frame_elements.push_back( i->first );
    }
  }

  LOG_INFO( main_logger, "Test tracks contain " << track_elements.size() << " track-level elements and "
            << frame_elements.size() << " frame elements" );

  // use track_csv as our proxy schema for convenience
  track_handle_list_type tracks;
  track_csv_type trk;

  unsigned fn_counter = 1000;
  for (size_t track_i = 0; track_i<n_tracks; ++track_i)
  {
    track_handle_type t = trk.create();
    tracks.push_back( t );

    external_id( t.row ) = track_i + 10;

    map< field_handle_type, bool> field_has_been_set;
    field_has_been_set[ external_id.get_field_handle() ] = true;

    for (size_t i=0; i<track_elements.size(); ++i)
    {
      field_handle_type fh = track_elements[ i ];
      if ( ! field_has_been_set[ fh ] )
      {
        track_oracle_core::get_mutable_element_store_base( fh )->set_to_default_value( t.row );
        field_has_been_set[ fh ] = true;
      }
    }

    field_has_been_set.clear();
    for (size_t frame_i=0; frame_i<n_frames_per_track; ++frame_i)
    {
      frame_handle_type f =  trk( t ).create_frame();

      frame_number( f.row ) = fn_counter++;
      unsigned long long ts_usecs = (frame_number(f.row) * 1000 * 1000) + (500 * 1000) + 500;
      timestamp_usecs( f.row ) = ts_usecs;
      time_stamp( f.row ) = vital::timestamp( ts_usecs / 1.0e6,  frame_number(f.row) );

      ostringstream track_oss, frame_oss;
      track_oss << "track_" << external_id( t.row );
      frame_oss << "frame_" << frame_number( f.row );
      string oddeven = (frame_i % 2 == 0) ? "even" : "odd";
      state_flags( f.row ).set_flag( "track", track_oss.str() );
      state_flags( f.row ).set_flag( "frame", frame_oss.str() );
      state_flags( f.row ).set_flag( oddeven );
      if (oddeven == string("even") )
      {
        state_flags( f.row ).set_flag( "present_on_even" );
        state_flags( f.row ).clear_flag( "present_on_odd" );
      }
      else
      {
        state_flags( f.row ).clear_flag( "present_on_even" );
        state_flags( f.row ).set_flag( "present_on_odd" );
      }

      field_has_been_set[ frame_number.get_field_handle() ] = true;
      field_has_been_set[ timestamp_usecs.get_field_handle() ] = true;
      field_has_been_set[ time_stamp.get_field_handle() ] = true;
      field_has_been_set[ state_flags.get_field_handle() ] = true;

      for (size_t i=0; i<frame_elements.size(); ++i)
      {
        field_handle_type fh = frame_elements[ i  ];
        if ( ! field_has_been_set[ fh ] )
        {
          track_oracle_core::get_mutable_element_store_base( fh )->set_to_default_value( t.row );
          field_has_been_set[ fh ] = true;
        }
      }
    } // ... for all the frames
  } // ...for all the tracks

  return file_format_manager::write( fn, tracks, TF_INVALID_TYPE );
}

} // ...track_oracle
} // ...kwiver
