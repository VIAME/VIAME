/*ckwg +5
 * Copyright 2011-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_MITRE_XML_H
#define INCL_TRACK_MITRE_XML_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_mitre_xml/track_mitre_xml_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_point_2d.h>

/*
This is a reader for the XML tracks MITRE supplies as queries ("*_BOX.xml")
for VIRAT.
*/

namespace kwiver {
namespace track_oracle {

struct TRACK_MITRE_XML_EXPORT track_mitre_xml_type: public track_base< track_mitre_xml_type >
{
  // frame level data
  track_field< vgl_box_2d<double> >& bounding_box;
  track_field< unsigned >& frame_number;

  track_mitre_xml_type():
    bounding_box( Frame.add_field< vgl_box_2d<double> >( "bounding_box" )),
    frame_number( Frame.add_field< unsigned > ("frame_number" ))
  {
  }
};

} // ...track_oracle
} // ...kwiver

#endif
