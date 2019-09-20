/*ckwg +5
 * Copyright 2014-2018 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_DATA_TERMS_H
#define INCL_DATA_TERMS_H

///
/// A data term represents the unique semantic concept contained in a
/// track_oracle column.  If multiple file formats use the same data
/// term, say an image bounding box, they thereby assert that the box
/// means the same thing in both formats.
///
/// Since the goal of track_oracle is to enable dynamic composition of
/// data structures using data terms, it's important that the data
/// term be semantically self-contained.  If there are two ways to
/// interpret the image box in a file format, i.e. coordinates relative
/// to the image vs. relative to the AOI, these need to be expressed
/// via two separate but complete data terms ("image absolute box" and
/// "aoi relative box"), rather than two dependent terms ("box" and
/// "coordinate system"), because a downstream user might use "box" but
/// know nothing about "coordinate system".
///
/// A data term has two attributes:
///
/// - the name (must be unique)
///
/// - the C++ data type
///
/// Although data terms are declared in C++ namespaces, those namespaces
/// are not necessarily reflected in the data term's name member.
///
/// The intent is that a data term type may be used to create a data_field.
/// Thus no instance of the data term is required, thus the name must be
/// statically defined.  (Which is tedious.)
///
///

#include <iostream>
#include <track_oracle/data_terms/data_terms_common.h>

#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_point_3d.h>

#include <vital/types/timestamp.h>
#include <vital/types/uid.h>

class TiXmlElement;

namespace kwiver {
namespace track_oracle {

namespace dt {

/// The data term declaration macros are:
///
/// DECL_DT: defines a data term with type-standard stream and XML I/O.
///
/// DECL_DT_RW_STRCSV: Custom stream I/O, custom CSV I/O, default XML I/O (using the stream I/O.)
///
/// DECL_DT_W_STR: Custom stream output; everything else default.  (For example,
/// setting the precision on the output of a double or float.)
///
/// DECL_DT_RW_STR: Custom stream input/output; everything else default.
///
/// DECL_DT_RW_STRXMLCSV: Custom stream, CSV, and XML I/O. (For example, if you
/// want to use custom XML attributes.)
///

namespace detection {
  DECL_DT( detection_id, unsigned long long, "detection ID; unique within a session but not a UUID" );
}

namespace tracking {

  DECL_DT( external_id, unsigned long long, "track ID; unique within a session but not a UUID" );
  DECL_DT( timestamp_usecs, unsigned long long, "timestamp of a frame, in usecs; epoch is data-dependent" );
  DECL_DT( frame_number, unsigned, "frame number; relationship to any downsampling is unspecified" );
  DECL_DT( fg_mask_area, double, "area of foreground mask; in pixels?" );
  DECL_DT_RW_STRCSV_DEFAULT( track_location, vgl_point_2d<double>, vgl_point_2d<double>(0,0), "track location; in frame-relative pixels?" );
  DECL_DT( obj_x, double, "estimate of object's x-axis location" );
  DECL_DT( obj_y, double, "estimate of object's y-axis location" );
  DECL_DT_RW_STRCSV( obj_location, vgl_point_2d<double>, "obj_{x,y} as a structure" );
  DECL_DT( velocity_x, double, "speed of object along x-axis; usually m/s" );
  DECL_DT( velocity_y, double, "speed of object along y-axis; usually m/s" );
  DECL_DT_RW_STRXMLCSV( bounding_box, vgl_box_2d<double>, "bounding box of tracked object; pixels" );
  DECL_DT_W_STR( world_x, double, "world X location (UTM or lat/lon)" );
  DECL_DT_W_STR( world_y, double, "world Y location (UTM or lat/lon)" );
  DECL_DT_W_STR( world_z, double, "world Z location (altitude?)" );
  DECL_DT_DEFAULT( world_gcs, int, 4326, "GCS (4326 == WGS84_LATLON); only written w/ world_x && world_y are written" );
  DECL_DT_RW_STRCSV( world_location, vgl_point_3d<double>, "world {x,y,z} as a structure" );
  DECL_DT_W_STR( latitude,  double, "latitude, -90 to 90" );
  DECL_DT_W_STR( longitude, double, "longitude, -180 to 180" );
  DECL_DT_RW_STRXMLCSV( time_stamp, vital::timestamp, "timestamp (carries both time and framenumber); epoch is data-dependent" );
  DECL_DT_RW_STR( track_uuid, vital::uid, "UUID associated with the track" );
  DECL_DT( track_style, std::string, "track_style, typically indicating the source (tracker, detector, etc.)" );

} // ...tracking

namespace events {
  DECL_DT( event_id, unsigned long long, "event ID; unique within a session but not a UUID" );
  DECL_DT_RW_STRXMLCSV( event_type, int, "event type: currently always in the VIRAT domain" );
  DECL_DT( event_probability, double, "event probability" );
  DECL_DT_RW_STR( source_track_ids, std::vector<unsigned>, "Track IDs contributing to the event" );
  DECL_DT_RW_STR( actor_track_rows, track_handle_list_type, "Track handles participating in the event" );

  DECL_DT( kpf_activity_domain, int, "KPF activity domain" );
  DECL_DT( kpf_activity_start, unsigned, "KPF activity start (frame number)" );
  DECL_DT( kpf_activity_stop, unsigned, "KPF activity stop (frame number)" );

} // ...events

namespace virat {
  DECL_DT_RW_STRXML( descriptor_classifier, std::vector<double>, "40-column vector of VIRAT activity probabilities" );

} // ...virat

} // ...dt

} // ...track_oracle
} // ...kwiver

#endif
