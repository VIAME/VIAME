/*ckwg +29
 * Copyright 2016, 2019 by Kitware, Inc.
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
