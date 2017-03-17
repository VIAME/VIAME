/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_FILE_FORMAT_MITRE_BOX_XML_H
#define INCL_FILE_FORMAT_MITRE_BOX_XML_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_mitre_xml/track_mitre_xml_export.h>

#include <track_oracle/file_formats/file_format_base.h>
#include <track_oracle/file_formats/track_mitre_xml/track_mitre_xml.h>

namespace kwiver {
namespace track_oracle {

class TRACK_MITRE_XML_EXPORT file_format_mitre_xml: public file_format_base
{
public:
  file_format_mitre_xml(): file_format_base( TF_MITRE_BOX_XML, "MITRE VIRAT query tracks" )
  {
    this->globs.push_back( "*.xml" );
  }
  virtual ~file_format_mitre_xml() {}

  virtual int supported_operations() const { return FF_READ_FILE; }

  // return a dynamically-allocated instance of the schema
  virtual track_base_impl* schema_instance() const { return new track_mitre_xml_type(); }

  // Inspect the file and return true if it is of this format
  virtual bool inspect_file( const std::string& fn ) const;

  using file_format_base::read;
  // read tracks from the file
  virtual bool read( const std::string& fn,
                     track_handle_list_type& tracks ) const;

};

} // ...track_oracle
} // ...kwiver

#endif
