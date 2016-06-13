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

#include "geo_map.h"

#include <vital/algo/geo_map.h>

#include <vital/bindings/c/helpers/algorithm.h>


DEFINE_COMMON_ALGO_API( geo_map )


using namespace kwiver;


/// Convert UTM coordinate into latitude and longitude.
void
vital_algorithm_geo_map_utm_to_latlon( vital_algorithm_t *algo,
                                       double easting, double northing,
                                       int zone, bool north_hemi,
                                       double *lat, double *lon,
                                       vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "utm_to_latlon", eh,
    auto a_sptr = vital_c::ALGORITHM_geo_map_SPTR_CACHE.get( algo );
    a_sptr->utm_to_latlon( easting, northing, zone, north_hemi, *lat, *lon );
  );
}

/// Convert latitude and longitude into UTM coordinates.
void
vital_algorithm_geo_map_latlon_to_utm( vital_algorithm_t *algo,
               double lat, double lon,
               double *easting, double *northing, int *zone, bool *north_hemi,
               int setzone,
               vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "latlon_to_utm", eh,
    auto a_sptr = vital_c::ALGORITHM_geo_map_SPTR_CACHE.get( algo );
    a_sptr->latlon_to_utm( lat, lon, *easting, *northing, *zone, *north_hemi,
                           setzone );
  );
}

/// Return the standard zone number for a given latitude and longitude
int
vital_algorithm_geo_map_latlon_zone( vital_algorithm_t *algo,
                                     double lat, double lon,
                                     vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "latlon_to_utm", eh,
    auto a_sptr = vital_c::ALGORITHM_geo_map_SPTR_CACHE.get( algo );
    return a_sptr->latlon_zone( lat, lon );
  );
  return -1;
}
