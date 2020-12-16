// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_APIX_H
#define INCL_FILE_FORMAT_APIX_H

#include <vital/vital_config.h>
#include <track_oracle/track_apix/track_apix_export.h>

#include <track_oracle/file_format_base.h>
#include <track_oracle/track_apix/track_apix.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_APIX_EXPORT apix_reader_opts: public file_format_reader_opts_base
{
  bool verbose;  // if true, reader will compare MGRS and lat/lon fields in shapefile

  apix_reader_opts& set_verbose( bool v ) { this->verbose = v; return *this; }
  apix_reader_opts& operator=( const file_format_reader_opts_base& rhs );
  virtual apix_reader_opts& reset() { set_verbose( false ); return *this; }
  apix_reader_opts() { reset(); }
};

class TRACK_APIX_EXPORT file_format_apix: public file_format_base
{
public:
  file_format_apix(): file_format_base( TF_APIX, "APIX-compatible shapefile" )
  {
    this->opts.reset();
    this->globs.push_back( "*.dbf" );
  }
  virtual ~file_format_apix() {}

  virtual int supported_operations() const { return FF_READ_FILE; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_apix_type(); }

  apix_reader_opts& options() { return this->opts; }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file( const std::string& fn ) const;

  using file_format_base::read;
  // read tracks from the file
  virtual bool read( const std::string& fn,
                     track_handle_list_type& tracks ) const;

private:
  apix_reader_opts opts;

};

} // ...track_oracle
} // ...kwiver

#endif
