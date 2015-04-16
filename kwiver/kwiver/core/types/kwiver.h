/*ckwg +5
 * Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_TYPES_KWIVER_H_
#define _KWIVER_TYPES_KWIVER_H_

#include <vector>

#include <core/geo_lat_lon.h>
#include <core/timestamp.h>
#include <core/homography.h>
#include <core/image_container.h>
#include <core/trait_utils.h>

#include <sprokit/pipeline_util/path.h>


namespace kwiver
{

/*! \file KWIVER specific types.
 *
 * This file contains the canonical type names for KWIVER types used
 * in the sprokit pipeline.
 */

// -- concrete types --
typedef double gsd_t;

/// \todo establish and document proper semantics for a polygon.
/// E.G. generally starts in upper left, proceeds around clockwise.
/// Is a closed figure, last point is connected back to first point.
/// Could wrap in a class to provide data abstraction.
typedef std::vector < kwiver::geo_lat_lon > geo_polygon_t;

/// \todo make a better corner points class that uses data abstraction
/// to provide proper semantics.

/**
 * \brief Geo polygon input operator.
 *
 * This operator converts a string to a geo polygon object. The
 * format of the string is "ul_lat ul_lon ur_lat ur_lon lr_lat lr_lon ll_lat ll_lon"
 *
 * @param str Stream to read from
 * @param obj Object to receive values
 *
 * @return
 */
std::istream& operator>> ( std::istream& str, geo_polygon_t& obj );


// ================================================================
//
// Create type traits for common pipeline types.
// ( type-trait-name, "canonical_type_name", concrete-type )
//
create_type_trait( timestamp, "kwiver:timestamp", kwiver::timestamp );
create_type_trait( gsd, "kwiver:gsd", gsd_t );
create_type_trait( corner_points, "corner_points", kwiver::geo_polygon_t );
create_type_trait( image, "kwiver:image_container", kwiver::image_container_sptr ); // polymorphic type must pass by reference
create_type_trait( homography, "kwiver:s2r_homography", kwiver::f2f_homography );
create_type_trait( image_file_name, "kwiver:image_file_name", sprokit::path_t );
create_type_trait( video_file_name, "kwiver:video_file_name", sprokit::path_t );


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

} // end namespace kwiver
#endif /* _KWIVER_TYPES_KWIVER_H_ */
