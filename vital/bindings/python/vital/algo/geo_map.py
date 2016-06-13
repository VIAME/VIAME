"""
ckwg +31
Copyright 2016 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

vital::algo::geo_map interface

"""
import ctypes

from vital.algo import VitalAlgorithm


class GeoMap (VitalAlgorithm):

    TYPE_NAME = "geo_map"

    def utm_to_latlon(self, easting, northing, zone, north_hemi):
        """
        Convert UTM coordinate into latitude and longitude

        :param easting: The easting (X) UTM coordinate in meters
        :param northing: The northing (Y) UTM coordinate in meters
        :param zone: The zone of the UTM coordinate (1-60).
        :param north_hemi: True if the UTM northing coordinate is in respect to
            the northern hemisphere.

        :return: converted latitude and longitude coordinates
        :rtype: (float, float)

        """
        lat = ctypes.c_double()
        lon = ctypes.c_double()
        self._call_cfunc(
            'vital_algorithm_geo_map_utm_to_latlon',
            [self.C_TYPE_PTR, ctypes.c_double, ctypes.c_double, ctypes.c_int,
             ctypes.c_bool, ctypes.POINTER(ctypes.c_double),
             ctypes.POINTER(ctypes.c_double)],
            [self, easting, northing, zone, north_hemi, ctypes.byref(lat),
             ctypes.byref(lon)]
        )
        return lat.value, lon.value

    def latlon_to_utm(self, lat, lon, set_zone=None):
        """
        Return the standard zone number for a given latitude and longitude

        :param lat: The latitude (Y) coordinate in decimal degrees
        :type lat: float

        :param lon: The longitude (X) coordinate in decimal degrees
        :type lon: float

        :param set_zone: If a valid UTM zone and not -1, use the given zone
            instead of the computed zone from the given lat/lon coordinate.
        :type set_zone: None | int

        :return: Converted easting, northing, zone and north_hemi values
        :rtype: (float, float, int, bool)

        """
        if set_zone is None:
            set_zone = -1
        easting = ctypes.c_double()
        northing = ctypes.c_double()
        zone = ctypes.c_int()
        north_hemi = ctypes.c_bool()

        self._call_cfunc(
            "vital_algorithm_geo_map_latlon_to_utm",
            [self.C_TYPE_PTR, ctypes.c_double, ctypes.c_double,
             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
             ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_bool),
             ctypes.c_int],
            [self, lat, lon, ctypes.byref(easting), ctypes.byref(northing),
             ctypes.byref(zone), ctypes.byref(north_hemi), set_zone]
        )

        return easting.value, northing.value, zone.value, north_hemi.value

    def latlon_zone(self, lat, lon):
        """
        Return the standard zone number for a given latitude and longitude

        This is a simplified implementation that ignores the exceptions to the
        standard UTM zone rules (e.g. around Norway, etc.).

        :param lat: latitude in decimal degrees
        :type lat: float

        :param lon: longitude in decimal degrees
        :type lon: float

        :return: integer zone value
        :rtype: int

        """
        return self._call_cfunc(
            'vital_algorithm_geo_map_latlon_zone',
            [self.C_TYPE_PTR, ctypes.c_double, ctypes.c_double],
            [self, lat, lon],
            ctypes.c_int
        )
