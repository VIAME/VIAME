/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \file
 * \brief File description here.
 */

#ifndef VITAL_C_ALGO_GEO_MAP_H_
#define VITAL_C_ALGO_GEO_MAP_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/algorithm.h>
#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/vital_c_export.h>


DECLARE_COMMON_ALGO_API( geo_map )


/// Convert UTM coordinate into latitude and longitude.
/**
 * \param[in]   algo        geo_map algorithm instance
 * \param[in]   easting     The easting (X) UTM coordinate in meters.
 * \param[in]   northing    The northing (Y) UTM coordinate in meters.
 * \param[in]   zone        The zone of the UTM coordinate (1-60).
 * \param[in]   north_hemi  True if the UTM northing coordinate is in respect
 *                          to the northern hemisphere.
 * \param[out]  lat         Output latitude (Y) in decimal degrees.
 * \param[out]  lon         Output longiture (X) in decimal degrees.
 * \param[in]   eh          Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_algorithm_geo_map_utm_to_latlon( vital_algorithm_t *algo,
                                       double easting, double northing,
                                       int zone, bool north_hemi,
                                       double *lat, double *lon,
                                       vital_error_handle_t *eh );


/// Convert latitude and longitude into UTM coordinates.
/**
 * \param[in]   algo        geo_map algorithm instance
 * \param[in]   lat         The latitude (Y) coordinate in decimal degrees.
 * \param[in]   lon         The longitude (X) coordinate in decimal degrees.
 * \param[out]  easting     Output easting (X) coordinate in meters.
 * \param[out]  northing    Output northing (Y) coordinate in meters.
 * \param[out]  zone        Zone of the output UTM coordinate.
 * \param[out]  north_hemi  True if the output UTM coordinate northing is in
 *                          respect to the northern hemisphere. False if not.
 * \param       setzone     If a valid UTM zone and not -1, use the given zone
 *                          instead of the computed zone from the given lat/lon
 *                          coordinate.
 * \param[in]   eh          Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_algorithm_geo_map_latlon_to_utm( vital_algorithm_t *algo,
                                       double lat, double lon,
                                       double *easting, double *northing,
                                       int *zone, bool *north_hemi,
                                       int setzone,
                                       vital_error_handle_t *eh );


/// Return the standard zone number for a given latitude and longitude
/**
 * This is a simplified implementation that ignores the exceptions to the
 * standard UTM zone rules (e.g. around Norway, etc.).
 *
 * \param algo geo_map algorithm instance
 * \param lat latitude in decimal degrees.
 * \param lon longitude in decimal degrees.
 * \param eh Vital error handle instance
 * \returns integer zone value
 */
int
vital_algorithm_geo_map_latlon_zone( vital_algorithm_t *algo,
                                     double lat, double lon,
                                     vital_error_handle_t *eh );


#ifdef __cplusplus
}
#endif

#endif //VITAL_C_ALGO_GEO_MAP_H_
