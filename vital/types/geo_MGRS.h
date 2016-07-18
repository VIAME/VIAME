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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief This file contains the interface for the geo MGRS coordinate
 */

#ifndef KWIVER_VITAL_GEO_MGRS_H_
#define KWIVER_VITAL_GEO_MGRS_H_

#include <ostream>
#include <string>

/**
 * \page MGRS Convert between UTM/UPS and MGRS
 *
 * MGRS is defined in Chapter 3 of
 * - J. W. Hager, L. L. Fry, S. S. Jacks, D. R. Hill,
 *   <a href="http://earth-info.nga.mil/GandG/publications/tm8358.1/pdf/TM8358_1.pdf">

 *   Datums, Ellipsoids, Grids, and Grid Reference Systems</a>,
 *   Defense Mapping Agency, Technical Manual TM8358.1 (1990).
 *
 * This implementation has the following properties:
 * - The conversions are closed, i.e., output from Forward is legal input for
 *   Reverse and vice versa.  Conversion in both directions preserve the
 *   UTM/UPS selection and the UTM zone.
 * - Forward followed by Reverse and vice versa is approximately the
 *   identity.  (This is affected in predictable ways by errors in
 *   determining the latitude band and by loss of precision in the MGRS
 *   coordinates.)
 * - All MGRS coordinates truncate to legal 100 km blocks.  All MGRS
 *   coordinates with a legal 100 km block prefix are legal (even though the
 *   latitude band letter may now belong to a neighboring band).
 * - The range of UTM/UPS coordinates allowed for conversion to MGRS
 *   coordinates is the maximum consistent with staying within the letter
 *   ranges of the MGRS scheme.
 * - All the transformations are implemented as static methods in the MGRS
 *   class.
 *
 * The <a href="http://www.nga.mil">NGA</a> software package
 * <a href="http://earth-info.nga.mil/GandG/geotrans/index.html">geotrans</a>
 * also provides conversions to and from MGRS.  Version 3.0 (and earlier)
 * suffers from some drawbacks:
 * - Inconsistent rules are used to determine the whether a particular MGRS
 *   coordinate is legal.  A more systematic approach is taken here.
 * - The underlying projections are not very accurately implemented.
 *
 **********************************************************************/

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * \brief Geographic point in MGRS
 *
 * This class represents a geographic location in MGRS coordinates.
 */
class geo_MGRS
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


std::ostream & operator<< (std::ostream & str, const kwiver::vital::geo_MGRS & obj);

} } // end namespace

#endif // KWIVER_VITAL_GEO_MGRS_H_
