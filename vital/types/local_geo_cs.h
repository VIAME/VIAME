/*ckwg +29
 * Copyright 2013-2019 by Kitware, Inc.
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
 * \brief core local_geo_cs interface
 */

#ifndef VITAL_LOCAL_GEO_CS_H_
#define VITAL_LOCAL_GEO_CS_H_

#include <vital/vital_export.h>
#include <vital/vital_config.h>

#include <vital/types/geo_point.h>
#include <vital/vital_types.h>
#include <vital/vital_config.h>

namespace kwiver {
namespace vital {


/// Represents a local geo coordinate system origin expressed in UTM
class VITAL_EXPORT local_geo_cs
{
public:
  /// Constructor
  local_geo_cs();

  /// Set the geographic coordinate origin
  /**
   * Internally converts this coordinate to WGS84 UTM
   */
  void set_origin(const vital::geo_point& origin);

  /// Access the geographic coordinate of the origin
  const vital::geo_point& origin() const { return geo_origin_; }

private:
  /// The local coordinates origin
  vital::geo_point geo_origin_;

};


/// Read a local_geo_cs from a text file
/**
 * The file format is the geographic origin in latitude (deg), longitude (deg),
 * and altitude (m) in space delimited ASCII value.  These values are read
 * into an existing local_geo_cs.
 *
 * \param [in,out] lgcs      The local geographic coordinate system that is
 *                           updated with the origin in the file.
 * \param [in]     file_path The path to the file to read.
 */
VITAL_EXPORT
void
read_local_geo_cs_from_file(local_geo_cs& lgcs,
                            vital::path_t const& file_path);


/// Write a local_geo_cs to a text file
/**
 * The file format is the geographic origin in latitude (deg), longitude (deg),
 * and altitude (m) in space delimited ASCII value.  These values are written
 * from an existing local_geo_cs.
 *
 * \param [in] lgcs      The local geographic coordinate system to write.
 * \param [in] file_path The path to the file to write.
 */
VITAL_EXPORT
void
write_local_geo_cs_to_file(local_geo_cs const& lgcs,
                           vital::path_t const& file_path);


} // end namespace vital
} // end namespace kwiver


#endif // VITAL_LOCAL_GEO_CS_H_
