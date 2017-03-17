/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_FILE_FORMAT_COMMS_XML_H
#define INCL_FILE_FORMAT_COMMS_XML_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_comms_xml/track_comms_xml_export.h>

#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/track_comms_xml/track_comms_xml.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_COMMS_XML_EXPORT comms_xml_reader_opts: public file_format_reader_opts_base
{
  // When reading comms files, which contain tracks from different queries, set
  // this to instruct the reader to only load tracks from the specific query.
  // Leave blank to load all tracks.
  std::string comms_qid;

  comms_xml_reader_opts& set_comms_qid( const std::string& s ) { this->comms_qid = s; return *this; }
  comms_xml_reader_opts& operator=( const file_format_reader_opts_base& rhs_base );
  virtual comms_xml_reader_opts& reset() { set_comms_qid( "" ); return *this; }
  comms_xml_reader_opts() { reset(); }
};


class TRACK_COMMS_XML_EXPORT file_format_comms_xml: public file_format_base
{
public:
  file_format_comms_xml(): file_format_base( TF_COMMS_XML, "MITRE VIRAT test harness comms XML" )
  {
    this->globs.push_back( "*comms-*.xml" );
  }
  virtual ~file_format_comms_xml() {}

  virtual int supported_operations() const { return FF_READ_FILE; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_comms_xml_type(); }

  comms_xml_reader_opts& options() { return this->opts; }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file( const std::string& fn ) const;

  using file_format_base::read;
  // read tracks from the file
  virtual bool read( const std::string& fn,
                     track_handle_list_type& tracks ) const;

private:
  comms_xml_reader_opts opts;

};

} // ...track_oracle
} // ...kwiver

#endif
