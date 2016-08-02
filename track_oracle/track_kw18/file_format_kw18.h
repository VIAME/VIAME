/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_FILE_FORMAT_KW18_H
#define INCL_FILE_FORMAT_KW18_H

#include <vital/vital_config.h>
#include <track_oracle/track_kw18/track_kw18_export.h>

#include <track_oracle/file_format_base.h>
#include <track_oracle/track_kw18/track_kw18.h>

namespace kwiver {
namespace track_oracle {

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

};

} // ...track_oracle
} // ...kwiver

#endif
