/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_TYPES_KWIVER_H_
#define _KWIVER_TYPES_KWIVER_H_

#include <sprokit/pipeline/process.h>

#include <vector>
#include <geo_lat_lon.h>

namespace kwiver
{

/*! \file KWIVER specific types.
 *
 * This file contains the canonical type names for KWIVER types used
 * in the sprokit pipeline.
 */

static sprokit::process::type_t const kwiver_corner_points( "kwiver_corner_points_ul_ur_lr_ll" );
static sprokit::process::type_t const kwiver_gsd( "kwiver_meters_per_pixel" );
static sprokit::process::type_t const kwiver_timestamp( "kwiver_timestamp" );

// -- concrete types --
typedef double gsd_t;
typedef std::vector < kwiver::geo_lat_lon > corner_points_t;

/**
 * \brief Corner points input operator.
 *
 * This operator converts a string to a corner points object. The
 * format of the string is "ul_lat ul_lon ur_lat ur_lon lr_lat lr_lon ll_lat ll_lon"
 *
 * @param str Stream to read from
 * @param obj Object to receive values
 *
 * @return
 */
std::istream& operator>> ( std::istream& str, corner_points_t& obj );

} // end namespace
#endif /* _KWIVER_TYPES_KWIVER_H_ */
