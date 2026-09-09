// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Abstract polygon interface

#ifndef VITAL_TYPES_POLYGON_H
#define VITAL_TYPES_POLYGON_H

#include <viame/core_types/vector.h>
#include <viame/core_types/vital_types_export.h>
#include <viame/algorithm_framework/vital_config.h>

#include <initializer_list>
#include <memory>
#include <optional>
#include <vector>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// @brief Abstract base polygon class.
///
/// This class represents a polygon with a limited number of
/// attributes. The concrete implementation of the polygon is delegated
/// to a concrete derived class. There may be more than one possible
/// implementation. These implementations should provide a way to
/// access the implementation specific methods because they usually
/// provide additional attributes and operations. These derived classes
/// should supply conversion methods to and from the basic (core)
/// implementation.
///
/// This class behaviour is considered the specification for all
/// derived classes.

class VITAL_TYPES_EXPORT polygon
{
public:
  typedef kwiver::vital::vector_2d point_t;

  polygon();
  polygon( const std::vector< point_t >& dat );
  polygon( std::initializer_list< point_t > dat );
  ~polygon();

  /// @brief Add point to end of polygon.
  ///
  /// This method adds a point to the end of the list of points that
  /// define the polygon.
  ///
  /// @param x The X coordinate
  /// @param y The Y coordinate
  void push_back( double x, double y );

  /// @brief Add point to end of polygon.
  ///
  /// This method adds a point to the end of the list of points that
  /// define the polygon.
  ///
  /// @param pt The point to add to polygon.
  void push_back( const point_t& pt );

  /// @brief Get number of vertices in polygon.
  ///
  /// This method returns the number of vertices or points in this
  /// polygon.
  ///
  ///  @return Number of vertices/points.
  size_t num_vertices() const;

  /// @brief Get list of vertices.
  ///
  /// This method returns the list of points that make up the polygon.
  ///
  /// @return List of vertices.
  std::vector< point_t > get_vertices() const;

  /// @brief Does this polygon contain the point.
  ///
  /// This method determines if the specified point is within the
  /// polygon or not. Vertex points and points in the boundary are
  /// considered within the polygon.
  ///
  /// @param x The X coordinate
  /// @param y The Y coordinate
  ///
  /// @return \b true if the point is within the polygon.
  bool contains( double x, double y );

  /// @brief Does this polygon contain the point.
  ///
  /// This method determines if the specified point is within the
  /// polygon or not. Vertex points and points in the boundary are
  /// considered within the polygon.
  ///
  /// @param pt The point to test.
  ///
  /// @return \b true if the point is within the polygon.
  bool contains( const point_t& pt );

  /// @brief Get Nth vertex in polygon.
  ///
  /// This method returns the requested vertex point. If the index is
  /// beyond the bounds of this polygon, an exception is thrown.
  ///
  /// @param idx The vertex index, from 0 to num_vertices()-1
  ///
  /// @return The point at the specified vertex.
  ///
  /// @throws std::out_of_range exception
  point_t at( size_t idx ) const;

  /// Calculate area of polygon.
  ///
  /// Polygon is assumed to be simple (not self-intersecting). Counter-clockwise
  /// polygons will produce positive area; clockwise polygons negative area.
  double area() const;

  /// Compute the intersection of two simple convex polygons.
  ///
  /// Both inputs are assumed to be simple (not self-intersecting), convex, and
  /// counter-clockwise. If the inputs meet at a single point or edge, that
  /// point or edge will be returned as the "polygon" of intersection.
  ///
  /// \warning
  ///   Any returned polygon is not necessarily simple, so combining more than
  ///   two polygons with this function should be done with care.
  ///
  /// \param a First polygon.
  /// \param b Second polygon.
  ///
  /// \return
  ///   Polygon representing the intersection of \p a and \p b, or
  ///   \c std::nullopt if they do not intersect.
  static std::optional< polygon >
  convex_intersection( polygon const& a, polygon const& b );

  /// Compute the union of two simple convex polygons.
  ///
  /// Both inputs are assumed to be simple (not self-intersecting), convex, and
  /// counter-clockwise. If the inputs meet only at a single point or edge, a
  /// single combined polygon will still be returned.
  ///
  /// \warning
  ///   Any returned polygon is not necessarily convex, so combining more than
  ///   two polygons with this function should be done with care.
  ///
  /// \param a First polygon.
  /// \param b Second polygon.
  ///
  /// \return
  ///   Polygon representing the union of \p a and \p b, or \c std::nullopt if
  ///   they do not intersect, in which case the union is simply \p a and \p b
  ///   considered separately.
  static std::optional< polygon >
  convex_union( polygon const& a, polygon const& b );

private:
  std::vector< point_t > m_polygon;
}; // end class polygon

// Types for managing polygons
typedef std::shared_ptr< polygon > polygon_sptr;
typedef std::vector< polygon_sptr >  polygon_sptr_list;

VITAL_TYPES_EXPORT
bool operator==( polygon const& lhs, polygon const& rhs );

VITAL_TYPES_EXPORT
bool operator!=( polygon const& lhs, polygon const& rhs );

} // namespace vital

}   // end namespace

#endif // VITAL_TYPES_POLYGON_H
