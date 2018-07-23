/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_4676_H
#define INCL_TRACK_4676_H

#include <track_oracle/track_base.h>
#include <track_oracle/track_field.h>

#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_point_3d.h>

#include <vital/types/uid.h>

namespace vidtk
{

typedef viat::uid uuid_t;

struct track_4676_type: public track_base< track_4676_type >
{

  // Track specific
  track_field< unsigned >& external_id;
  track_field< uuid_t >& unique_id;
  track_field< std::string >& augmented_annotation;

  // Frame specific
  track_field< unsigned long long >& timestamp_usecs;
  track_field< vgl_point_2d< double > >& obj_location;
  track_field< vgl_point_3d< double > >& world_location;
  track_field< double >& obj_x;
  track_field< double >& obj_y;
  track_field< double >& world_x;
  track_field< double >& world_y;
  track_field< double >& world_z;

  track_4676_type():
    external_id(Track.add_field< unsigned >("external_id")),
    unique_id(Track.add_field< uuid_t >("unique_id")),
    augmented_annotation(Track.add_field< std::string >("augmented_annotation")),

    timestamp_usecs(Frame.add_field< unsigned long long >("timestamp_usecs")),
    obj_location(Frame.add_field< vgl_point_2d< double > >("obj_location")),
    world_location(Frame.add_field< vgl_point_3d< double > >("world_location")),
    obj_x(Frame.add_field< double >("obj_x")),
    obj_y(Frame.add_field< double >("obj_y")),
    world_x(Frame.add_field< double >("world_x")),
    world_y(Frame.add_field< double >("world_y")),
    world_z(Frame.add_field< double >("world_z"))
  {
  }
};

} // \namespace vidtk

#endif
