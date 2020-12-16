// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_KW18_H
#define INCL_FILE_FORMAT_KW18_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kw18/track_kw18_export.h>

#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/track_kw18/track_kw18.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_KW18_EXPORT kw18_reader_opts: public file_format_reader_opts_base
{
  bool kw19_hack;  // if true, set per-frame "relevancy" from the 19th column of a kw18

  kw18_reader_opts& set_kw19_hack( bool b ) { this->kw19_hack = b; return *this; }

  virtual kw18_reader_opts& reset() { set_kw19_hack( false ); return *this; }
  kw18_reader_opts() { reset(); }
  kw18_reader_opts& operator=( const file_format_reader_opts_base& rhs );
};

class TRACK_KW18_EXPORT file_format_kw18: public file_format_base
{
public:
  file_format_kw18(): file_format_base( TF_KW18, "Kitware generic tracks" )
  {
    this->globs.push_back( "*.kw18" );
    this->globs.push_back( "*.kw19" );
    this->globs.push_back( "*.kw20" );
    current_filename = "(none)";
  }
  virtual ~file_format_kw18() {}

  virtual int supported_operations() const { return FF_READ | FF_WRITE; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_kw18_type(); }

  // return the options for the kw18 format
  kw18_reader_opts& options() { return this->opts; }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file( const std::string& fn ) const;

  // read tracks from the file
  virtual bool read( const std::string& fn,
                     track_handle_list_type& tracks ) const;

  // read tracks from a stream
  virtual bool read( std::istream& is,
                     track_handle_list_type& tracks ) const;

  // write tracks to a file
  virtual bool write( const std::string& fn,
                      const track_handle_list_type& tracks ) const;

  // write tracks to a file
  virtual bool write( std::ostream& os,
                      const track_handle_list_type& tracks ) const;

private:
  mutable std::string current_filename;
  bool internal_stream_read( std::istream& is, size_t file_size, track_handle_list_type& tracks) const;
  kw18_reader_opts opts;

};

} // ...track_oracle
} // ...kwiver

#endif
