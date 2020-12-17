// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the implementation of a local geographical offset coordinate system.
 */

#include <vital/math_constants.h>
#include <vital/types/local_cartesian.h>
#include <vital/types/geodesy.h>

#include <iomanip>
#include <stdexcept>

namespace kwiver {
namespace vital {

/**
 * @Brief Local Cartesian Conversion Math Utility
 *
 * Base on the NGA GeoTrans library
 * https://earth-info.nga.mil/GandG/update/index.php?action=home
 *
 * This class is a cleaned up version of the LocalCartesian and Geocentric classes provided in GeoTrans
 * This allows a user to define a local cartesian coordinate system with any origin (expressed in WGS84)
 */
class local_cartesian::geotrans
{
public:
  geotrans( geo_point const& origin, double orientation )
  {
    // Using WGS84 ellipsoid
    semiMajorAxis = 6378137.0;
    inv_f = 298.257223563;
    flattening = 1 / inv_f;

    Geocent_e2 = 2 * flattening - flattening * flattening;
    Geocent_ep2 = (1 / (1 - Geocent_e2)) - 1;
    Cos_67p5 = 0.38268343236508977; // cosine of 67.5 degrees

    set_origin(origin, orientation);
  }
   ~geotrans() = default;

  /**
   * @brief Set origin of the cartesian system as a geo_point
   *
   * Set local origin parameters as inputs and sets the corresponding state variables.
   * NOTE : If the origin changes, this method needs to be called again to recompute
   *        variables needed in the conversion math.
   *
   * @param [in] geo_point       : Geograpical origin of the cartesian system
   * @param [in] orientation     : Orientation angle of the local cartesian coordinate system,
   *                               in radians along the Z axis, normal to the earth surface
   */
  void set_origin( geo_point const& origin, double orientation )
  {
    if (origin.is_empty())
    {
      throw std::runtime_error("Origin geo_point is empty");
    }

    double N0;
    double val;

    auto loc = origin.location(kwiver::vital::SRID::lat_lon_WGS84);

    LocalCart_Origin_Lat = loc[1] * deg_to_rad;
    LocalCart_Origin_Long = loc[0] * deg_to_rad;
    if (LocalCart_Origin_Long > pi)
      LocalCart_Origin_Long -= two_pi;
    LocalCart_Origin_Height = loc[2];
    if (orientation > pi)
    {
      orientation -= two_pi;
    }
    LocalCart_Orientation = orientation;

    Sin_LocalCart_Origin_Lat = sin(LocalCart_Origin_Lat);
    Cos_LocalCart_Origin_Lat = cos(LocalCart_Origin_Lat);
    Sin_LocalCart_Origin_Lon = sin(LocalCart_Origin_Long);
    Cos_LocalCart_Origin_Lon = cos(LocalCart_Origin_Long);
    Sin_LocalCart_Orientation = sin(LocalCart_Orientation);
    Cos_LocalCart_Orientation = cos(LocalCart_Orientation);

    Sin_Lat_Sin_Orient = Sin_LocalCart_Origin_Lat * Sin_LocalCart_Orientation;
    Sin_Lat_Cos_Orient = Sin_LocalCart_Origin_Lat * Cos_LocalCart_Orientation;

    N0 = semiMajorAxis / sqrt(1 - Geocent_e2 * Sin_LocalCart_Origin_Lat * Sin_LocalCart_Origin_Lat);

    val = (N0 + LocalCart_Origin_Height) * Cos_LocalCart_Origin_Lat;
    u0 = val * Cos_LocalCart_Origin_Lon;
    v0 = val * Sin_LocalCart_Origin_Lon;
    w0 = ((N0 * (1 - Geocent_e2)) + LocalCart_Origin_Height) * Sin_LocalCart_Origin_Lat;
  }

  /**
   * @brief Converts geodetic coordinates to local cartesian coordinates
   *
   * The function convertFromGeodetic converts geodetic coordinates
   * (latitude, longitude, and height) to local cartesian coordinates (X, Y, Z),
   * according to the WGS84 ellipsoid and local origin parameters.
   *
   * @param [in]     geodetic_coordinate : WGS84 longitude/latitude in degrees and hight in meters
   * @param [in,out] cartesian_coordinate : The location in the local coordinate system, in meters
   */
  void convert_from_geodetic( vector_3d const& geodetic_coordinate, vector_3d& cartesian_coordinate ) const
  {
    double longitude = geodetic_coordinate.x() * deg_to_rad;
    double latitude = geodetic_coordinate.y() * deg_to_rad;
    double height = geodetic_coordinate.z();

    double Rn;            /*  Earth radius at location  */
    double Sin_Lat;       /*  sin(Latitude)  */
    double Sin2_Lat;      /*  Square of sin(Latitude)  */
    double Cos_Lat;       /*  cos(Latitude)  */

    if (longitude > pi)
    {
      longitude -= (2 * pi);
    }
    Sin_Lat = sin(latitude);
    Cos_Lat = cos(latitude);
    Sin2_Lat = Sin_Lat * Sin_Lat;
    Rn = semiMajorAxis / (sqrt(1.0e0 - Geocent_e2 * Sin2_Lat));
    double X = (Rn + height) * Cos_Lat * cos(longitude);
    double Y = (Rn + height) * Cos_Lat * sin(longitude);
    double Z = ((Rn * (1 - Geocent_e2)) + height) * Sin_Lat;

    vector_3d geocentric_coordinate;
    geocentric_coordinate << X, Y, Z;

    convert_from_geocentric(geocentric_coordinate, cartesian_coordinate);
  }

  /**
   * @brief Converts local cartesian coordinates to geodetic coordinates
   *
   * The function convertToGeodetic converts local cartesian
   * coordinates (X, Y, Z) to geodetic coordinates (latitude, longitude,
   * and height), according to the WGS84 ellipsoid and local origin parameters.
   *
   * @param [in]     cartesian_coordinate : The location in the local coordinate system, in meters
   * @param [in,out] geodetic_coordinate  : WGS84 longitude/latitude in degrees and hight, in meters
   */
  void convert_to_geodetic( vector_3d const& cartesian_coordinate, vector_3d& geodetic_coordinate ) const
  {
    vector_3d geocentric_coordinate;
    convert_to_geocentric(cartesian_coordinate, geocentric_coordinate);

    double X = geocentric_coordinate.x();
    double Y = geocentric_coordinate.y();
    double Z = geocentric_coordinate.z();
    double latitude, longitude, height;

    // Only copied the new geocentric-to-geodetic algorithm
    // Removed the legacy geocentric-to-geodetic algorithm
    double equatorial_radius = semiMajorAxis;
    double eccentricity_squared = Geocent_e2;

    double rho, c, s, e1, e2a;

    e1 = 1.0 - eccentricity_squared;
    e2a = eccentricity_squared * equatorial_radius;

    rho = sqrt(X * X + Y * Y);

    if (Z == 0.0)
    {
      c = 1.0;
      s = 0.0;
      latitude = 0.0;
    }
    else
    {
      double  ct, new_ct, zabs;
      double  f, new_f, df_dct, e2;

      zabs = fabs(Z);

      new_ct = rho / zabs;
      new_f = std::numeric_limits<double>::max();

      do
      {
        ct = new_ct;
        f = new_f;

        e2 = sqrt(e1 + ct * ct);

        new_f = rho - zabs * ct - e2a * ct / e2;

        if (new_f == 0.0) break;

        df_dct = -zabs - (e2a*e1) / (e2*e2*e2);

        new_ct = ct - new_f / df_dct;

        if (new_ct < 0.0) new_ct = 0.0;
      } while (fabs(new_f) < fabs(f));

      s = 1.0 / sqrt(1.0 + ct * ct);
      c = ct * s;

      if (Z < 0.0)
      {
        s = -s;
        latitude = -atan(1.0 / ct);
      }
      else
      {
        latitude = atan(1.0 / ct);
      }
    }

    longitude = atan2(Y, X);

    height = rho * c + Z * s - equatorial_radius * sqrt(1.0 - eccentricity_squared * s*s);

    geodetic_coordinate << (longitude*rad_to_deg), (latitude*rad_to_deg), height;
  }

  /**
   * @brief Converts geocentric coordinates to cartesian coordinates
   *
   * The function convertFromGeocentric converts geocentric
   * coordinates according to the WGS84 ellipsoid and local origin parameters.
   *
   * @param [in]     geocentric_coordinate : The geocentric location, in meters
   * @param [in,out] cartesian_coordinate  : Calculated local cartesian coordinate, in meters
   */
  void convert_from_geocentric( vector_3d const& geocentric_coordinate, vector_3d& cartesian_coordinate ) const
  {
    double X, Y, Z;
    double u_MINUS_u0, v_MINUS_v0, w_MINUS_w0;

    double U = geocentric_coordinate.x();
    double V = geocentric_coordinate.y();
    double W = geocentric_coordinate.z();

    u_MINUS_u0 = U - u0;
    v_MINUS_v0 = V - v0;
    w_MINUS_w0 = W - w0;

    if (LocalCart_Orientation == 0.0)
    {
      double cos_lon_u_MINUS_u0 = Cos_LocalCart_Origin_Lon * u_MINUS_u0;
      double sin_lon_v_MINUS_v0 = Sin_LocalCart_Origin_Lon * v_MINUS_v0;

      X = -Sin_LocalCart_Origin_Lon * u_MINUS_u0 + Cos_LocalCart_Origin_Lon * v_MINUS_v0;
      Y = -Sin_LocalCart_Origin_Lat * cos_lon_u_MINUS_u0 + -Sin_LocalCart_Origin_Lat * sin_lon_v_MINUS_v0 + Cos_LocalCart_Origin_Lat * w_MINUS_w0;
      Z = Cos_LocalCart_Origin_Lat * cos_lon_u_MINUS_u0 + Cos_LocalCart_Origin_Lat * sin_lon_v_MINUS_v0 + Sin_LocalCart_Origin_Lat * w_MINUS_w0;
    }
    else
    {
      double cos_lat_w_MINUS_w0 = Cos_LocalCart_Origin_Lat * w_MINUS_w0;

      X = (-Cos_LocalCart_Orientation * Sin_LocalCart_Origin_Lon + Sin_Lat_Sin_Orient * Cos_LocalCart_Origin_Lon) * u_MINUS_u0 +
        (Cos_LocalCart_Orientation * Cos_LocalCart_Origin_Lon + Sin_Lat_Sin_Orient * Sin_LocalCart_Origin_Lon) * v_MINUS_v0 +
        (-Sin_LocalCart_Orientation * cos_lat_w_MINUS_w0);

      Y = (-Sin_LocalCart_Orientation * Sin_LocalCart_Origin_Lon - Sin_Lat_Cos_Orient * Cos_LocalCart_Origin_Lon) * u_MINUS_u0 +
        (Sin_LocalCart_Orientation * Cos_LocalCart_Origin_Lon - Sin_Lat_Cos_Orient * Sin_LocalCart_Origin_Lon) * v_MINUS_v0 +
        (Cos_LocalCart_Orientation * cos_lat_w_MINUS_w0);

      Z = (Cos_LocalCart_Origin_Lat * Cos_LocalCart_Origin_Lon) * u_MINUS_u0 +
        (Cos_LocalCart_Origin_Lat * Sin_LocalCart_Origin_Lon) * v_MINUS_v0 +
        Sin_LocalCart_Origin_Lat * w_MINUS_w0;
    }

    cartesian_coordinate << X, Y, Z;
  }

  /**
   * @brief Converts cartesian coordinates to geocentric coordinates
   *
   * The function Convert_Local_Cartesian_To_Geocentric converts local cartesian
   * coordinates (x, y, z) to geocentric coordinates (X, Y, Z) according to the
   * current ellipsoid and local origin parameters.
   *
   * @param [in,out] cartesian_coordinate  : Local cartesian coordinate, in meters
   * @param [in]     geocentric_coordinate : The geocentric location, in meters
   */
  void convert_to_geocentric( vector_3d const& cartesian_coordinate, vector_3d& geocentric_coordinate ) const
  {
    double U, V, W;

    double X = cartesian_coordinate.x();
    double Y = cartesian_coordinate.y();
    double Z = cartesian_coordinate.z();

    if (LocalCart_Orientation == 0.0)
    {
      double sin_lat_y = Sin_LocalCart_Origin_Lat * Y;
      double cos_lat_z = Cos_LocalCart_Origin_Lat * Z;

      U = -Sin_LocalCart_Origin_Lon * X - sin_lat_y * Cos_LocalCart_Origin_Lon + cos_lat_z * Cos_LocalCart_Origin_Lon + u0;
      V = Cos_LocalCart_Origin_Lon * X - sin_lat_y * Sin_LocalCart_Origin_Lon + cos_lat_z * Sin_LocalCart_Origin_Lon + v0;
      W = Cos_LocalCart_Origin_Lat * Y + Sin_LocalCart_Origin_Lat * Z + w0;
    }
    else
    {
      double rotated_x, rotated_y;
      double rotated_y_sin_lat, z_cos_lat;

      rotated_x = Cos_LocalCart_Orientation * X + Sin_LocalCart_Orientation * Y;
      rotated_y = -Sin_LocalCart_Orientation * X + Cos_LocalCart_Orientation * Y;

      rotated_y_sin_lat = rotated_y * Sin_LocalCart_Origin_Lat;
      z_cos_lat = Z * Cos_LocalCart_Origin_Lat;

      U = -Sin_LocalCart_Origin_Lon * rotated_x - Cos_LocalCart_Origin_Lon * rotated_y_sin_lat + Cos_LocalCart_Origin_Lon * z_cos_lat + u0;
      V = Cos_LocalCart_Origin_Lon * rotated_x - Sin_LocalCart_Origin_Lon * rotated_y_sin_lat + Sin_LocalCart_Origin_Lon * z_cos_lat + v0;
      W = Cos_LocalCart_Origin_Lat * rotated_y + Sin_LocalCart_Origin_Lat * Z + w0;
    }

    geocentric_coordinate << U, V, W;
  }
private:
  // Ellipsoid Parameters
// All are set to a WGS 84 ellipsoid
  double semiMajorAxis;
  double flattening;
  double inv_f;
  double Geocent_e2;
  double Geocent_ep2;
  double Cos_67p5;

  double es2;                       /* Eccentricity (0.08181919084262188000) squared */
  double u0;                        /* Geocentric origin coordinates in */
  double v0;                        /* terms of Local Cartesian origin  */
  double w0;                        /* parameters                       */

  /* Local Cartesian Projection Parameters */
  double LocalCart_Origin_Lat;      /* Latitude of origin in radians     */
  double LocalCart_Origin_Long;     /* Longitude of origin in radians    */
  double LocalCart_Origin_Height;   /* Height of origin in meters        */
  double LocalCart_Orientation;     /* Orientation of Y axis in radians  */

  double Sin_LocalCart_Origin_Lat;  /* sin(LocalCart_Origin_Lat)         */
  double Cos_LocalCart_Origin_Lat;  /* cos(LocalCart_Origin_Lat)         */
  double Sin_LocalCart_Origin_Lon;  /* sin(LocalCart_Origin_Lon)         */
  double Cos_LocalCart_Origin_Lon;  /* cos(LocalCart_Origin_Lon)         */
  double Sin_LocalCart_Orientation; /* sin(LocalCart_Orientation)        */
  double Cos_LocalCart_Orientation; /* cos(LocalCart_Orientation)        */

  double Sin_Lat_Sin_Orient; /* sin(LocalCart_Origin_Lat) * sin(LocalCart_Orientation) */
  double Sin_Lat_Cos_Orient; /* sin(LocalCart_Origin_Lat) * cos(LocalCart_Orientation) */
  double Cos_Lat_Cos_Orient; /* cos(LocalCart_Origin_Lat) * cos(LocalCart_Orientation) */
  double Cos_Lat_Sin_Orient; /* cos(LocalCart_Origin_Lat) * sin(LocalCart_Orientation) */
};

local_cartesian::local_cartesian( geo_point const& origin, double orientation )
{
  origin_ = origin;
  orientation_ = orientation;
  geotrans_ = new geotrans(origin, orientation);
}

void local_cartesian::set_origin( geo_point const& origin, double orientation )
{
  origin_ = origin;
  orientation_ = orientation;
  geotrans_->set_origin(origin, orientation);
}

local_cartesian::~local_cartesian()
{
  delete geotrans_;
}

geo_point local_cartesian::get_origin() const
{
  return origin_;
}

double local_cartesian::get_orientation() const
{
  return orientation_;
}

void local_cartesian::convert_from_cartesian( vector_3d const& cartesian_coordinate, geo_point& location ) const
{
  vector_3d geodetic;
  geotrans_->convert_to_geodetic(cartesian_coordinate, geodetic);
  location.set_location(geodetic, kwiver::vital::SRID::lat_lon_WGS84);
}

void local_cartesian::convert_to_cartesian( geo_point const& location, vector_3d& cartesian_coordinate ) const
{
  geotrans_->convert_from_geodetic(location.location(kwiver::vital::SRID::lat_lon_WGS84), cartesian_coordinate);
}

} } // end namespace
