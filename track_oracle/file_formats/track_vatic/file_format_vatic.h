// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_VATIC_H
#define INCL_FILE_FORMAT_VATIC_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_vatic/track_vatic_export.h>

#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/track_vatic/track_vatic.h>

namespace kwiver {
namespace track_oracle {

class TRACK_VATIC_EXPORT file_format_vatic: public file_format_base
{
public:
  file_format_vatic(): file_format_base( TF_VATIC, "VATIC ground truth" )
  {
    this->globs.push_back( "*.txt" );
  }
  virtual ~file_format_vatic() {}

  virtual int supported_operations() const { return FF_READ; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_vatic_type(); }

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
