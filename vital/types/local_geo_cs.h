// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
