/*ckwg +29
 * Copyright 2013-2014 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * @brief Shared type declarations for the kwiver module.
 *
 * This file contains the canonical type names for KWIVER types used
 * in the sprokit pipeline.
 */

#ifndef KWIVER_CORE_TYPES_H
#define KWIVER_CORE_TYPES_H

#include <kwiver/geo_lat_lon.h>

#include <string>
#include <vector>
#include <stdint.h>

#include <boost/filesystem/path.hpp>

namespace kwiver
{

/// The type to be used for file and directory paths
typedef boost::filesystem::path path_t;

/// The type of a landmark ID number
typedef unsigned int landmark_id_t;

/// The type of a track ID number
typedef int64_t track_id_t;

/// The type of a frame number or camera ID
typedef int64_t frame_id_t;


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
 * This operator is needed to read polygons from the config.
 *
 * @param str Stream to read from
 * @param obj Object to receive values
 *
 * @return
 */
std::istream& operator>> ( std::istream& str, geo_polygon_t& obj );

}

#endif // KWIVER_CORE_TYPES_H
