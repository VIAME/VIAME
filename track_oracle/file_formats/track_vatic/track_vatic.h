/*ckwg +5
 * Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_VATIC_H
#define INCL_TRACK_VATIC_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_vatic/track_vatic_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_point_2d.h>
#include <set>

namespace kwiver {
namespace track_oracle {

struct TRACK_VATIC_EXPORT track_vatic_type: public track_base< track_vatic_type >
{
  // track level data
  track_field< unsigned int >& external_id;

  // frame level data
  track_field< vgl_box_2d<double> >& bounding_box;
  track_field< unsigned >& frame_number;
  track_field< bool >& lost;
  track_field< bool >& occluded;
  track_field< bool >& generated;
  track_field< std::string >& label;
  track_field< std::set< std::string > >& attributes;

  track_vatic_type():
    external_id( Track.add_field< unsigned int >( "external_id" )),
    bounding_box( Frame.add_field< vgl_box_2d<double> >( "bounding_box" )),
    frame_number( Frame.add_field< unsigned > ("frame_number" )),
    lost( Frame.add_field< bool > ("lost" )),
    occluded( Frame.add_field< bool > ("occluded" )),
    generated( Frame.add_field< bool > ("generated" )),
    label( Frame.add_field< std::string > ("label" )),
    attributes( Frame.add_field< std::set< std::string > > ("vatic_attributes" ))
  {
  }
};

} // ...track_oracle
} // ...kwiver

#endif
