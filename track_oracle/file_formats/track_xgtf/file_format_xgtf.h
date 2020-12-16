// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_XGTF_H
#define INCL_FILE_FORMAT_XGTF_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_xgtf/track_xgtf_export.h>

#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/track_xgtf/track_xgtf.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_XGTF_EXPORT xgtf_reader_opts: public file_format_reader_opts_base
{
  bool promote_pvmoving; // if set, some activities will be converted to {p,v}moving

  xgtf_reader_opts& set_promote_pvmoving( bool p ) { this->promote_pvmoving = p; return *this; }
  xgtf_reader_opts& operator=( const file_format_reader_opts_base& rhs_base );
  virtual xgtf_reader_opts& reset() { set_promote_pvmoving( false ); return *this; }
  xgtf_reader_opts() { reset(); }
};

class TRACK_XGTF_EXPORT file_format_xgtf: public file_format_base
{
public:
  file_format_xgtf(): file_format_base( TF_XGTF, "ViPER ground truth (using VIRAT schemas)" )
  {
    this->opts.reset();
    this->globs.push_back( "*.xgtf" );
  }
  virtual ~file_format_xgtf() {}

  virtual int supported_operations() const { return FF_READ_FILE; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_xgtf_type(); }

  xgtf_reader_opts& options() { return this->opts; }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file( const std::string& fn ) const;

  using file_format_base::read;

  // read tracks from the file
  virtual bool read( const std::string& fn,
                     track_handle_list_type& tracks ) const;

private:
  xgtf_reader_opts opts;

};

} // ...track_oracle
} // ...kwiver

#endif
