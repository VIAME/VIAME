// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_VPD_TRACK_H
#define INCL_FILE_FORMAT_VPD_TRACK_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_vpd/track_vpd_export.h>

#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/track_vpd/track_vpd_track.h>

namespace kwiver {
namespace track_oracle {

class TRACK_VPD_EXPORT file_format_vpd_track: public file_format_base
{
public:
  file_format_vpd_track(): file_format_base( TF_VPD_TRACK, "VIRAT Public Data 2.0 object track" )
  {
    this->globs.push_back( "*.viratdata.objects.txt" );
  }
  virtual ~file_format_vpd_track() {}

  virtual int supported_operations() const { return FF_READ; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_vpd_track_type(); }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file( const std::string& fn ) const;

  // read tracks from the file
  virtual bool read( const std::string& fn,
                     track_handle_list_type& tracks ) const;

  // read tracks from a stream
  virtual bool read( std::istream& is,
                     track_handle_list_type& tracks ) const;

};

} // ...track_oracle
} // ...kwiver

#endif
