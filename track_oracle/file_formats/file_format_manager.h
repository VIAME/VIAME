/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_FILE_FORMAT_MANAGER_H
#define INCL_FILE_FORMAT_MANAGER_H

///
/// This class is the typical client interface to all
/// the file formats the system knows about.
///
/// ****
/// **** IN ORDER TO ADD A FILE FORMAT:
/// ****
/// **** 1) write its schema (derived from track_base)
/// **** 2) write its file_format object (derived from file_format_base)
/// **** 3) add it to the file_format_enum enumeration (file_format_type.h)
/// **** 4) add it to the formats[] map in file_format_manager_impl's constructor
/// ****
/// **** That should be it; all the rest should follow automatically.
/// ****
///
///
/// Use cases:
///
/// 1) You're the GUI and want to know what globs
/// to put in the file dialog for both KW18 and XGTF files:
///
/// [...]
/// std::vector< std::string > globs;
/// globs.push_back( file_format_manager::get_format( TF_KW18 )->format_globs() );
/// globs.push_back( file_format_manager::get_format( TF_XGTF )->format_globs() );
/// [...]
///
///
/// 2) Somebody gave you a filename; you just want to load
/// the freaking tracks and get on with your life:
///
/// [...]
/// track_handle_list_type the_tracks;
/// bool okay = file_format_manager::read( filename, the_tracks );
/// [...]
///
///
/// 3) Deep in your code, you suddenly want to know which file
/// was the origin for a track handle (this is actually the use
/// case for the file_format_schema...)
///
/// [...]
/// track_handle_type the_handle = ...
/// [...]
/// file_format_schema_type ffs;
/// std::string source_filename =
///   file_format_schema_type::source_id_to_filename( ffs( the_handle ).source_file_id() );
/// [...]
///
///
/// 4) You know you're loading an XGTF file, and want to set the promote-to-PVMoving
/// flag in the activity classifier
///
/// [...]
/// track_handle_list_type xgtf_tracks;
/// file_format_xgtf xgtf_reader;
/// xgtf_reader.options().set_promote_pvmoving( this->promote_pvmoving() );
/// bool okay = xgtf_reader.read( filename, xgtf_tracks );
/// [...]
///
///

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_oracle_file_formats_export.h>

#include <vector>
#include <string>
#include <map>
#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/track_base_impl.h>
#include <track_oracle/file_formats/file_format_type.h>

namespace kwiver {
namespace track_oracle {

struct file_format_manager_impl;
struct file_format_reader_opts_base;
class file_format_base;
class track_field_base;
struct element_descriptor;

typedef std::map< file_format_enum, file_format_base* > format_map_type;
typedef std::map< file_format_enum, file_format_base* >::const_iterator format_map_cit;

class TRACK_ORACLE_FILE_FORMATS_EXPORT file_format_manager
{
public:

  // intialize the internal maps; call this if your workflow queries
  // track_oracle before loading any files
  static void initialize();

  // get a ref to the map of formats
  static const format_map_type& get_format_map();

  // get a pointer to the file_format_base for the format
  static file_format_base* get_format( file_format_enum fmt );

  // give the user a mutable ref to a format's options
  // (throw if bad format or no reader implemented)
  static file_format_reader_opts_base& options( file_format_enum fmt );

  // give the user a mutable ref to a format's options, but call reset() first
  // (throw if bad format or no reader implemented)
  static file_format_reader_opts_base& default_options( file_format_enum fmt );

  // return a map of the globs
  static std::map< file_format_enum, std::vector< std::string > > get_all_globs();

  // what formats match the globs?
  static std::vector< file_format_enum > globs_match( std::string fn );

  // what format is this file? (inspect if multiple formats match at glob level)
  static file_format_enum detect_format( const std::string& fn );

  // read the tracks from the file, using the current state of the options
  // stored in each of the file_format_base derived objects
  static bool read( const std::string& fn, track_handle_list_type& tracks );

  // write the tracks to the file.  If explicit_format is not TF_INVALID_TYPE,
  // use that format regardless of filename; if it is TF_INVALID_TYPE, deduce the
  // format from the filename (failing unless we get exactly one matching format.)
  static bool write( const std::string& fn,
                     const track_handle_list_type& tracks,
                     file_format_enum explicit_format = TF_INVALID_TYPE );

  // format introspection: what formats contain this element?
  static std::vector< file_format_enum > format_contains_element( const element_descriptor& e);

  // format introspection: what formats match this schema?
  static std::vector< file_format_enum > format_matches_schema( const track_base_impl& s);

  // clone (dynamically allocate) a field from an existing format matching the element descriptor
  // (return null if no match.)  Use to construct new schemata by slicing and dicing
  // existing formats.
  static std::pair< track_field_base*, track_base_impl::schema_position_type >
  clone_field_from_element( const element_descriptor& e );

  // Write a file of synthesized (boring) tracks containing all the data fields,
  // mostly to test I/O.  If header_map is non-empty, only those fields are
  // used (plus a few we always write out.)
  static bool write_test_tracks( const std::string& fn,
                                 const csv_handler_map_type& header_map,
                                 size_t n_tracks = 5,
                                 size_t n_frames_per_track = 10 );

private:

  static file_format_manager_impl& get_instance();
  static file_format_manager_impl* impl;
};

} // ...track_oracle
} // ...kwiver

#endif
