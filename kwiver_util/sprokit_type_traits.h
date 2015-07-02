/*ckwg +5
 * Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _VITAL_TYPES_VITAL_H_
#define _VITAL_TYPES_VITAL_H_

#include <vital/geo_lat_lon.h>
#include <vital/timestamp.h>
#include <vital/homography_f2f.h>
#include <vital/image_container.h>
#include <vital/trait_utils.h>
#include <vital/types.h>


// ================================================================
//
// Create type traits for common pipeline types.
// ( type-trait-name, "canonical_type_name", concrete-type )
//
create_type_trait( timestamp, "kwiver:timestamp", kwiver::vital::timestamp );
create_type_trait( gsd, "kwiver:gsd", kwiver::vital::gsd_t );
create_type_trait( corner_points, "corner_points", kwiver::vital::geo_polygon_t );
create_type_trait( image, "kwiver:image_container", kwiver::vital::image_container_sptr ); // polymorphic type must pass by reference
create_type_trait( homography, "kwiver:s2r_homography", kwiver::vital::f2f_homography );
create_type_trait( image_file_name, "kwiver:image_file_name", kwiver::vital::path_t );
create_type_trait( video_file_name, "kwiver:video_file_name", kwiver::vital::path_t );


// ================================================================
//
// Create port traits for common port types.
// ( port-name, type-trait-name, "port-description" )
//
create_port_trait( timestamp, timestamp, "Timestamp for input image." );
create_port_trait( corner_points, corner_points, "Four corner points for image in lat/lon units, ordering ul ur lr ll." );
create_port_trait( gsd, gsd, "GSD for image in meters per pixel." );
create_port_trait( image, image, "Single frame image." );
create_port_trait( src_to_ref_homography, homography, "Source image to ref image homography." );
create_port_trait( image_file_name, image_file_name, "Name of an image file. Usually a single frame of a video." );
create_port_trait( video_file_name, video_file_name, "Name of video file." );

#endif /* _KWIVER_TYPES_KWIVER_H_ */
