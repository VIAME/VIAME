// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_TYPES_LOCAL_TANGENT_SPACE_H_
#define VITAL_TYPES_LOCAL_TANGENT_SPACE_H_

#include <viame/core_types/geo_point.h>
#include <viame/core_types/rotation.h>
#include <viame/core_types/vital_types_export.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Local coordinate system tangent to the earth.
///
/// This class comes with the following guarantees:
///   1. The coordinate system is always cartesian; distances do not curve
///      around the Earth.
///   2. The coordinate system is always right-handed.
///   3. The coordinate system is always expressed in meters.
///   4. If the origin is not at the center of the earth, The X-Y plane is
///      tangent to the earth's surface at the origin and the Z axis points up.
///   5. If the origin is not along the polar axis, the X axis points east and
///      the Y axis points north.
///
/// These guarantees only hold if the ``local_tangent_space`` has been
/// initialized with a non-empty origin point.
struct VITAL_TYPES_EXPORT local_tangent_space
{
public:
  local_tangent_space();

  /// \param origin The point at which local coordinates will be zero.
  explicit local_tangent_space( geo_point const& origin );

  /// Return \c true if this object holds a valid tangent space.
  bool valid() const;

  /// Return the origin of the coordinate system.
  geo_point const& origin() const;

  /// Convert \p global_point to the local coordinate system.
  ///
  /// \param global_point Point in geographic coordinates to convert.
  vector_3d to_local( geo_point const& global_point ) const;

  /// Convert \p global_rotation to the local coordinate system.
  ///
  /// \param global_rotation
  ///   Point in geographic coordinates to convert, in the reference system
  ///   defined by \p global_point.crs(). If that reference system is geodetic,
  ///   East-North-Up axes are assumed.
  /// \param global_point Point where the rotation is centered.
  rotation_d to_local(
    rotation_d const& global_rotation, geo_point const& global_point ) const;

  /// Convert \p local_point to geographic coordinates.
  ///
  /// \param local_point Point in local coordinates to convert.
  geo_point to_global( vector_3d const& local_point ) const;

  /// Convert \p global_rotation to the local coordinate system.
  ///
  /// \param local_rotation
  ///   Point in local coordinates to convert. Will be converted the reference
  ///   system defined by \p global_point.crs(). If that reference system is
  ///   geodetic, East-North-Up axes will be used.
  /// \param global_point Point where the rotation is centered.
  rotation_d to_global(
    rotation_d const& local_rotation, geo_point const& global_point ) const;

private:
  geo_point m_origin;
  matrix_3x3d m_axes;
};

// ----------------------------------------------------------------------------
VITAL_TYPES_EXPORT
local_tangent_space
read_local_tangent_space_from_file( std::string const& filepath );

// ----------------------------------------------------------------------------
VITAL_TYPES_EXPORT
void
write_local_tangent_space_to_file(
  local_tangent_space const& local_space, std::string const& filepath );

} // namespace vital

} // namespace kwiver

#endif
