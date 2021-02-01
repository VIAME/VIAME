// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_KWXML_H
#define INCL_FILE_FORMAT_KWXML_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kwxml/track_kwxml_export.h>

#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/track_kwxml/track_kwxml.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_KWXML_EXPORT kwxml_reader_opts: public file_format_reader_opts_base
{
  std::string track_style_filter; // if set, only load tracks whose <trackStyle> matches this

  kwxml_reader_opts& set_track_style_filter( const std::string& s )
  {
    this->track_style_filter = s;
    return *this;
  }
  kwxml_reader_opts& operator=( const file_format_reader_opts_base& rhs_base );
  virtual kwxml_reader_opts& reset() { set_track_style_filter( "" ); return *this; }
  kwxml_reader_opts(){ reset(); }
};

class TRACK_KWXML_EXPORT file_format_kwxml: public file_format_base
{
public:
  file_format_kwxml(): file_format_base( TF_KWXML, "Kitware descriptor XML" )
  {
    this->globs.push_back( "*.xml" );
    this->globs.push_back( "*.kwxml" );
  }
  virtual ~file_format_kwxml() {}

  virtual int supported_operations() const { return FF_READ_FILE; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_kwxml_type(); }

  kwxml_reader_opts& options() { return this->opts; }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file( const std::string& fn ) const;

  using file_format_base::read;
  // read tracks from the file
  virtual bool read( const std::string& fn,
                     track_handle_list_type& tracks ) const;

private:
  kwxml_reader_opts opts;

};

} // ...track_oracle
} // ...kwiver

#endif
