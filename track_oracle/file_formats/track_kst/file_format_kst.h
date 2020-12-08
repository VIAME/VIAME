// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_KST_H
#define INCL_FILE_FORMAT_KST_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kst/track_kst_export.h>

#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/track_kst/track_kst.h>

namespace kwiver {
namespace track_oracle {

class TRACK_KST_EXPORT file_format_kst: public file_format_base
{
public:
  file_format_kst(): file_format_base( TF_KST, "VisGUI query results" )
  {
    this->globs.push_back( "*.vqr" );
  }
  virtual ~file_format_kst() {}

  virtual int supported_operations() const { return FF_READ; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_kst_type(); }

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
