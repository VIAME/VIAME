/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_KW_18_H
#define INCL_TRACK_KW_18_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_kw18/track_kw18_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/data_terms/data_terms.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_KW18_EXPORT track_kw18_type: public track_base< track_kw18_type >
{

  track_field< dt::tracking::external_id > external_id;

  track_field< dt::tracking::fg_mask_area > fg_mask_area;
  track_field< dt::tracking::track_location > track_location;
  track_field< dt::tracking::obj_x > obj_x;
  track_field< dt::tracking::obj_y > obj_y;
  track_field< dt::tracking::obj_location> obj_location;
  track_field< dt::tracking::velocity_x > velocity_x;
  track_field< dt::tracking::velocity_y > velocity_y;
  track_field< dt::tracking::bounding_box > bounding_box;
  track_field< dt::tracking::world_x > world_x;
  track_field< dt::tracking::world_y > world_y;
  track_field< dt::tracking::world_z > world_z;
  track_field< dt::tracking::world_location > world_location;
  track_field< dt::tracking::timestamp_usecs > timestamp_usecs;
  track_field< dt::tracking::frame_number > frame_number;

  track_kw18_type()
  {
    Track.add_field( external_id );
    Frame.add_field( fg_mask_area );
    Frame.add_field( track_location );
    Frame.add_field( obj_x );
    Frame.add_field( obj_y );
    Frame.add_field( obj_location );
    Frame.add_field( velocity_x );
    Frame.add_field( velocity_y );
    Frame.add_field( bounding_box );
    Frame.add_field( world_x );
    Frame.add_field( world_y );
    Frame.add_field( world_z );
    Frame.add_field( world_location );
    Frame.add_field( timestamp_usecs );
    Frame.add_field( frame_number );
  }

};

} // ...track_oracle
} // ...kwiver

#endif
