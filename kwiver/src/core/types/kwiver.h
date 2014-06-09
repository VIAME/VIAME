/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_TYPES_KWIVER_H_
#define _KWIVER_TYPES_KWIVER_H_

#include <vector>
#include <geo_lat_lon.h>
#include <timestamp.h>

#include <maptk/core/homography.h>
#include <maptk/core/image_container.h>

#include <trait_utils.h>


namespace kwiver
{

/*! \file KWIVER specific types.
 *
 * This file contains the canonical type names for KWIVER types used
 * in the sprokit pipeline.
 */

// -- concrete types --
typedef double gsd_t;
typedef std::vector < kwiver::geo_lat_lon > geo_polygon_t;


/**
 * \brief Corner points input operator.
 *
 * This operator converts a string to a corner points object. The
 * format of the string is "ul_lat ul_lon ur_lat ur_lon lr_lat lr_lon ll_lat ll_lon"
 *
 *
 *
 * @param str Stream to read from
 * @param obj Object to receive values
 *
 * @return
 */
std::istream& operator>> ( std::istream& str, geo_polygon_t& obj );


// ================================================================
//
// Create type traits for common pipeline tipes.
// ( type-trait-name, concrete-type )
//
create_type_trait( timestamp, kwiver::timestamp );
create_type_trait( gsd, gsd_t );
create_type_trait( corner_points, kwiver::geo_polygon_t );
create_type_trait( image, maptk::image_container_sptr );
create_type_trait( homography, maptk::f2f_homography );

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


} // end namespace
#endif /* _KWIVER_TYPES_KWIVER_H_ */
