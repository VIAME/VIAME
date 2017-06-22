/*ckwg +5
 * Copyright 2011-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_COMMS_XML_H
#define INCL_TRACK_COMMS_XML_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_comms_xml/track_comms_xml_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <vgl/vgl_box_2d.h>

/*
This is a reader for the XML tracks comms supplies as queries ("comms-*.xml")
for VIRAT.
*/

namespace kwiver {
namespace track_oracle {

struct TRACK_COMMS_XML_EXPORT track_comms_xml_type: public track_base< track_comms_xml_type >
{
  // track level data
  track_field< std::string >& track_source;
  track_field< double >& probability;
  track_field< std::string >& query_id;

  // frame level data
  track_field< vgl_box_2d<double> >& bounding_box;
  track_field< unsigned long long >& timestamp;

  track_comms_xml_type():
    track_source( Track.add_field< std::string >( "track_source" )),
    probability( Track.add_field< double > ( "activity_probability" )),
    query_id( Track.add_field< std::string > ( "query_id" )),
    bounding_box( Frame.add_field< vgl_box_2d<double> >( "bounding_box" )),
    timestamp( Frame.add_field< unsigned long long > ("timestamp_usecs" ))
  {
  }
};

} // ...track_oracle
} // ...kwiver

#endif
