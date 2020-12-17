// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the interface for the geo MGRS coordinate.
 */

#ifndef KWIVER_VITAL_GEO_MGRS_H_
#define KWIVER_VITAL_GEO_MGRS_H_

#include <vital/vital_export.h>

#include <ostream>
#include <string>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * \brief Geographic point in MGRS.
 *
 * This class represents a geographic location in MGRS coordinates. MGRS is
 * defined in Chapter 3 of:
 *
 * - J. W. Hager, L. L. Fry, S. S. Jacks, D. R. Hill,
 *   <a href="http://earth-info.nga.mil/GandG/publications/tm8358.1/pdf/TM8358_1.pdf">
 *   Datums, Ellipsoids, Grids, and Grid Reference Systems</a>,
 *   Defense Mapping Agency, Technical Manual TM8358.1 (1990).
 */
class VITAL_EXPORT geo_MGRS
{
public:
  geo_MGRS();
  geo_MGRS(std::string const& coord);
  ~geo_MGRS();

  /// default constructed coordinate
  bool is_empty() const;
  bool is_valid() const;

  geo_MGRS & set_coord( std::string const& coord);

  std::string const& coord() const;

  bool operator==( const geo_MGRS& rhs ) const;
  bool operator!=( const geo_MGRS& rhs ) const;
  geo_MGRS operator=( const geo_MGRS& mu );

private:
  std::string mgrs_coord_;

}; // end class geo_MGRS

VITAL_EXPORT std::ostream & operator<< (std::ostream & str, const kwiver::vital::geo_MGRS & obj);

} } // end namespace

#endif
