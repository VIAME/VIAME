// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_KWIVER_H
#define INCL_FILE_FORMAT_KWIVER_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kwiver/track_kwiver_export.h>

#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/track_kwiver/track_kwiver.h>

namespace kwiver {
namespace track_oracle {

class TRACK_KWIVER_EXPORT file_format_kwiver: public file_format_base
{
public:
  file_format_kwiver(): file_format_base( TF_KWIVER, "Kitware kwiver file" )
  {
    this->globs.push_back( "*.kwxml" );
    this->globs.push_back( "*.kwiver" );
    this->globs.push_back( "*.xml" );
  }
  virtual ~file_format_kwiver() {}

  virtual int supported_operations() const { return FF_READ_FILE | FF_WRITE; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_kwiver_type(); }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file( const std::string& fn ) const;

  using file_format_base::read;

  // read tracks from the file
  virtual bool read( const std::string& fn,
                     track_handle_list_type& tracks ) const;

  // write tracks to the file
  virtual bool write( const std::string& fn,
                      const track_handle_list_type& tracks ) const;

  // write tracks to a stream
  virtual bool write( std::ostream& os,
                      const track_handle_list_type& tracks ) const;

};

} // ...track_oracle
} // ...kwiver

#endif
