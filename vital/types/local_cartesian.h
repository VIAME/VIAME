/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
 * \brief This file contains the interface to a local geographic offset coordinate system.
 */

#ifndef KWIVER_VITAL_LOCAL_CARTESIAN_H_
#define KWIVER_VITAL_LOCAL_CARTESIAN_H_

#include <vital/types/point.h>
#include <vital/types/geo_point.h>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
/// \brief Local Cartesian Conversion Utility. */
class VITAL_EXPORT local_cartesian
{
public:

  local_cartesian( geo_point const& origin, double orientation = 0 );
  virtual ~local_cartesian();

  /// \brief Set origin of the cartesian system as a geo_point
  /**
   * Set local origin parameters as inputs and sets the corresponding state variables.
   * NOTE : If the origin changes, this method needs to be called again to recompute
   *        variables needed in the conversion math.
   *
   * \param [in] geo_point       : Geograpical origin of the cartesian system
   * \param [in] orientation     : Orientation angle of the local cartesian coordinate system,
   *                               in radians along the Z axis, normal to the earth surface
   */
  void set_origin( geo_point const& origin, double orientation=0 );

  /// \brief Get the origin of the coordinate system
  /**
   * \returns a geo_point origin of the coordinate system
   */
  geo_point get_origin() const;

  /// \brief Get the orientation of the coordinate system
  /**
   * \returns an orientation of the coordinate system along the Z axis
   */
  double get_orientation() const;

  /// \brief Calculate the geo_point of the provided cartesian coordinates.
  /**
   * \param [in]     cartesian_coordinate : The location to convert in cartesian system
   * \param [in,out] location             : The geo_point set with the computed WGS84 lon/lat/alt (degrees, meters)
   */
  void convert_from_cartesian( vector_3d const& cartesian_coordinate, geo_point& location ) const;

  /// \brief Calculate the location of the provided geo location in the local coordinate system.
  /**
   * \param [in]     location : The location to convert in cartesian system
   * \param [in,out] cartesian_coordinate : The location in the local coordinate system, in meters
   */
  void convert_to_cartesian( geo_point const& location, vector_3d& cartesian_coordinate ) const;

private:
  class geotrans;
  geotrans* geotrans_;
  geo_point origin_;
  double orientation_;
};

} } // end namespace

#endif /* KWIVER_VITAL_LOCAL_CARTESIAN_H_ */
