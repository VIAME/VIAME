// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
