// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_TYPES_BOUNDING_BOX_H
#define KWIVER_VITAL_TYPES_BOUNDING_BOX_H

#include <vital/types/vector.h>
#include <Eigen/Geometry>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Coordinate aligned bounding box.
 *
 * This class represents a coordinate aligned box. The coordinate
 * system places the origin in the upper left.
 *
 * A bounding box must be constructed with the correct geometry. Once
 * created, the geometry can not be altered.
 */
template < typename T >
class bounding_box
{
public:
  typedef Eigen::Matrix< T, 2, 1 > vector_type;

  /**
   * @brief Create box from two corner points.
   *
   * @param upper_left Upper left corner of box.
   * @param lower_right Lower right corner of box.
   */
  bounding_box( vector_type const& upper_left,
                vector_type const& lower_right )
    : m_bbox( upper_left, lower_right )
  { }

  /**
   * @brief Create box from point and dimensions.
   *
   * @param upper_left Upper left corner point
   * @param width Width of box.
   * @param height Height of box.
   */
  bounding_box( vector_type const& upper_left,
                T width, T height )
  {
    vector_type lr( upper_left );
    lr.x() += width;
    lr.y() += height;
    m_bbox =  Eigen::AlignedBox< T, 2 >( upper_left, lr );
  }

  /**
   * @brief Create a box from four coordinates.
   *
   * @param xmin Minimum x coordinate
   * @param ymin Minimum y coordinate
   * @param xmax Maximum x coordinate
   * @param ymax Maximum y coordinate
   */
  bounding_box( T xmin, T ymin, T xmax, T ymax )
  {
    vector_type ul( xmin, ymin );
    vector_type lr( xmax, ymax );
    m_bbox =  Eigen::AlignedBox< T, 2 >( ul, lr );
  }

  /**
   * @brief Create default (invalid) box.
   */
  bounding_box()
  {
    // NOTE: Any initial state logic here
    // Should be reproduced in the reset method
  }

  /**
   * @brief Check to see if the two corner points are valid.
   *
   * @return true if the box is valid
   */
  bool is_valid() const
  {
    return !m_bbox.isEmpty();
  }

  /**
   * @brief Reset the bounding box to an initial invalid state.
   */
  void reset()
  {
    m_bbox.setEmpty();
  }

  /**
   * @brief Get center coordinate of box.
   *
   * @return Center coordinate of box.
   */
  vector_type center() const { return m_bbox.center(); }

  /**
   * @brief Get upper left coordinate of box.
   *
   * @return Upper left coordinate of box.
   */
  vector_type upper_left() const { return m_bbox.min(); }

  /**
   * @brief Get lower right coordinate of box.
   *
   * @return Lower right coordinate of box.
   */
  vector_type lower_right() const { return m_bbox.max(); }

  T min_x() const { return this->upper_left()[0]; }
  T min_y() const { return this->upper_left()[1]; }
  T max_x() const { return this->lower_right()[0]; }
  T max_y() const { return this->lower_right()[1]; }

  /**
   * @brief Get width of box.
   *
   * @return Width of box.
   */
  T width() const { return m_bbox.sizes()[0]; }

  /**
   * @brief Get height of box.
   *
   * @return Height of box.
   */
  T height() const { return m_bbox.sizes()[1]; }

  /**
   * @brief Get area of box.
   *
   * @return Area of box.
   */
  double area() const { return m_bbox.volume(); }

  /**
  * @brief Check if point inside box.
  *
  * @return true if point is inside box
  */
  bool contains(vector_type const& pt) const { return m_bbox.contains(pt); }

protected:
  /*
   * @brief Obscure accessors for underlying data.
   *
   *
   * @return Underlying data type.
   */
  Eigen::AlignedBox< T, 2 >& get_eabb()  { return m_bbox; }
  Eigen::AlignedBox< T, 2 > get_eabb() const  { return m_bbox; }

private:
  // Note that this class is implemented using Eigen types.
  // There is no guarantee of this in the future.
  bounding_box( Eigen::AlignedBox< T, 2 > const& b )
    : m_bbox( b )
  { }

  /*
   * These operations are friends to allow them access to the
   * underlying data type. They need access to the private data in
   * order to use the methods on that type, an implementation
   * convenience.
   */
  template < typename T1 >
  friend bounding_box<T1> & translate( bounding_box<T1>& bbox,
                              typename bounding_box<T1>::vector_type const& pt );

  template < typename T1 >
  friend bounding_box<T1> scale( bounding_box<T1> const& bbox,
                                 double scale_factor );

  template<typename T2>
  friend bounding_box<T2> intersection( bounding_box<T2> const& one,
                                        bounding_box<T2> const& other );

  Eigen::AlignedBox< T, 2 > m_bbox;
};

/**
 * @brief Equality operator for bounding box
 *
 * @param lhs The box to check against
 * @param rhs The other box to check against
 *
 * @return \b true if boxes are identical
 */
template <typename T>
bool operator== ( bounding_box<T> const& lhs, bounding_box<T> const& rhs )
{
  if ( ( &lhs == &rhs ) ||
       ( lhs.upper_left() == rhs.upper_left()  &&
         lhs.lower_right() == rhs.lower_right() )
    )
  {
    return true;
  }

  return false;
}

/**
 * @brief Inequality operator for bounding box
 *
 * @param lhs The box to check against
 * @param rhs The other box to check against
 *
 * @return \b true if boxes are different
 */
template <typename T>
bool operator!= ( bounding_box<T> const& lhs, bounding_box<T> const& rhs )
{
  return !(lhs == rhs);
}

// Define for common types.
typedef bounding_box< int > bounding_box_i;
typedef bounding_box< double > bounding_box_d;

/**
 * @brief Translate a box by (x,y) offset.
 *
 * This operator translates a bounding_box by the specified
 * amount. The box being translated is modified.
 *
 * @param[in,out] bbox Box to translate
 * @param[in] pt X and Y offsets to use for translation
 *
 * @return The specified parameter box, updated with the new
 * coordinates, is returned.
 */
template < typename T >
bounding_box<T> & translate( bounding_box<T>& bbox,
                             typename bounding_box<T>::vector_type const& pt )
{
  bbox.get_eabb().translate( pt );
  return bbox;
}

/**
 * @brief Scale a box by some scale factor.
 *
 * This operator scales bounding_box by the specified factor. A new
 * box is returned that has been scaled from the input box.
 *
 * @param bbox Box to scale
 * @param scale_factor Scale factor to use
 *
 * @return A new bounding box that has been scaled.
 */
template < typename T >
bounding_box<T> scale( bounding_box<T> const& bbox,
                       double scale_factor )
{
  return bounding_box<T>(
    (bbox.upper_left().template cast<double>() * scale_factor).template cast<T>(),
    (bbox.lower_right().template cast<double>() * scale_factor).template cast<T>() );
}

/**
 * @brief Determine intersection of two boxes.
 *
 * This operator calculates the intersection of two
 * bounding_boxes. The rectangular intersection is returned as a new
 * bounding box.
 *
 * @param one The first bounding box for finding intersection
 * @param other The other bounding box for finding intersection.
 *
 * @return A new bounding_box specifying the intersection if the two
 * parameters.
 */
template<typename T>
bounding_box<T> intersection( bounding_box<T> const& one,
                              bounding_box<T> const& other )
{
  return bounding_box<T>( one.get_eabb().intersection( other.get_eabb() ) );
}

} }   // end namespace kwiver

#endif /* KWIVER_VITAL_TYPES_BOUNDING_BOX_H */
