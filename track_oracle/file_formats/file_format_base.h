// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_BASE_H
#define INCL_FILE_FORMAT_BASE_H

///
/// This class enables reasoning about file formats: is
/// file X an instance of a particular format, what's the
/// set of canonical file globs for a format, etc.
///
/// The relationship between file formats and schemas:
///
/// - You should be able to deal with the schema inside your
/// code without needing to #include anything dealing with the
/// file format.
///
/// - Similarly, you shouldn't need the schema #includes (or anything
/// else about track_oracle) to find out about track formats.
///
/// This argues for separate classes for schemas and formats.
///
/// BUT
///
/// - realistically, it's nice to have the kw18 file I/O bundled
/// with the schema.  All the readers have pretty much the same API.
/// Hmm.
///
/// If we had an ABC for file formats, that could be inherited by
/// the schemas.  But there's no reason for schemas to be that
/// tightly coupled to formats.
///
/// Instead, file formats are *factories* for schemas.  Hmm!
/// Expand the reader classes, put them under an abstract parent
/// class, call that parent class the file_format_base, and
/// spin a lot of the functionality in track_reader out to those
/// classes.  Hah!  I like it.
///

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_oracle_format_base_export.h>

#include <string>
#include <vector>
#include <map>
#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/element_descriptor.h>
#include <track_oracle/core/track_base_impl.h>
#include <track_oracle/file_formats/file_format_type.h>

namespace kwiver {
namespace track_oracle {

class track_base_impl;

enum file_format_operation
{
  FF_READ_FILE      = 0x01,
  FF_READ_STREAM    = 0x02,
  FF_READ           = 0x03,
  FF_WRITE_FILE     = 0x10,
  FF_WRITE_STREAM   = 0x20,
  FF_WRITE          = 0x30,
};

struct TRACK_ORACLE_FORMAT_BASE_EXPORT file_format_reader_opts_base
{
  // empty base class for format-specific reader options
  virtual file_format_reader_opts_base& reset() { return *this; } // restore options to a known state

  // make op= virtual so we can assign into the manager's map like this:
  // file_format_manager::reader_options( TF_APIX ) = apix_reader_opts().set_verbose( true );
  // ... and not have the derived class sliced to the base
  virtual file_format_reader_opts_base& operator=( const file_format_reader_opts_base& /*other*/) { return *this;}
  virtual ~file_format_reader_opts_base() {}
};

class TRACK_ORACLE_FORMAT_BASE_EXPORT file_format_base
{
public:
  file_format_base( file_format_enum format_type, const std::string& description );
  virtual ~file_format_base();

  // a description of the format (defaults to name)
  std::string format_description() const;

  // set the description
  void set_format_description( const std::string& d );

  // get the format type
  file_format_enum get_format() const;

  // get supported file format operations
  virtual int supported_operations() const = 0;

  // list of filename globs typically matching this file format
  std::vector< std::string > format_globs() const;

  // does the filename match any of our globs?
  bool filename_matches_globs( std::string fn ) const;

  // Inspect the file and return true if it is of this format
  // Whether or not empty files return true is format-dependent,
  // typically if the reader accepts empty files
  virtual bool inspect_file( const std::string& fn ) const = 0;

  // return our option object; clients should mutate before calling read
  virtual file_format_reader_opts_base& options();

  // read tracks from the file using current state of options
  virtual bool read( const std::string& fn,
                     track_handle_list_type& tracks ) const = 0;

  // read tracks from a stream (not implemented for all formats)
  virtual bool read( std::istream& is,
                     track_handle_list_type& tracks ) const;

  // write tracks to the file (not implemented for all formats)
  virtual bool write( const std::string& fn,
                      const track_handle_list_type& tracks ) const;

  // write tracks to a stream (not implemented for all formats)
  virtual bool write( std::ostream& os,
                      const track_handle_list_type& tracks ) const;

  // read tracks from the file, check that each track contains the
  // fields listed in the required_fields schema.  List missing fields
  // in the missing_fields vector.  Return (true, false) if the read
  // (succeeds, fails); if all required fields are present, then
  // missing_fields will be empty.  (In other words, returning true
  // does NOT mean all the fields are present; it means the read
  // succeeded.)

  bool read( const std::string& fn,
             track_handle_list_type& tracks,
             const track_base_impl& required_fields,
             std::vector< element_descriptor >& missing_fields ) const;

  bool read( std::istream& is,
             track_handle_list_type& tracks,
             const track_base_impl& required_fields,
             std::vector< element_descriptor >& missing_fields ) const;

  // format introspection: what fields does this format define?
  std::map< field_handle_type, track_base_impl::schema_position_type > enumerate_schema() const;

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const = 0;

protected:
  file_format_enum type;
  std::string description;
  std::vector< std::string > globs;
  file_format_reader_opts_base null_opts;

};

} // ...track_oracle
} // ...kwiver

#endif
