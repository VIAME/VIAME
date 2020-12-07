// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_VPD_H
#define INCL_TRACK_VPD_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_vpd/track_vpd_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <string>
#include <utility>
#include <vgl/vgl_box_2d.h>

namespace kwiver {
namespace track_oracle {

/// This is the track_oracle schema for the VIRAT Public Data 2.0 object
/// tracks.

struct TRACK_VPD_EXPORT track_vpd_track_type: public track_base< track_vpd_track_type >
{
  track_field< unsigned >& object_id;
  track_field< unsigned >& object_type;
  track_field< unsigned >& frame_number;
  track_field< vgl_box_2d< double > >& bounding_box;

  track_vpd_track_type():
    object_id( Track.add_field< unsigned >( "object_id" )),
    object_type( Track.add_field< unsigned >( "object_type" )),
    frame_number( Frame.add_field< unsigned >( "frame_number" )),
    bounding_box( Frame.add_field< vgl_box_2d< double > >( "bounding_box" ))
  {}

  static std::string object_type_to_str( unsigned t );
  static unsigned str_to_object_type( const std::string& s );
};

} // ...track_oracle
} // ...kwiver

#endif
