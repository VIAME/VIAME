// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the interface to a simple point.
 *
 * This point class pairs a eigen vector with a covariance matrix.
 */

#ifndef KWIVER_VITAL_POINT_H_
#define KWIVER_VITAL_POINT_H_

#include <vital/types/covariance.h>
#include <vital/types/vector.h>

#include <vital/vital_config.h>
#include <vital/vital_export.h>

#include <memory>

namespace kwiver {
namespace vital {

template < unsigned N, typename T >
class VITAL_EXPORT point
{
public:
  using vector_type = Eigen::Matrix< T, N, 1 >;
  using covariance_type = covariance_< N, float >;
  using data_type = T;

  point() {}
  explicit point( vector_type const& v, covariance_type const& c = {} )
    : m_value{ v }, m_covariance{ c } {}

  template < unsigned K = N,
             typename std::enable_if< K == 2, bool >::type = true >
  point( T x, T y ) : m_value{ x, y } {}

  template < unsigned K = N,
             typename std::enable_if< K == 3, bool >::type = true >
  point( T x, T y, T z ) : m_value{ x, y, z } {}

  template < unsigned K = N,
             typename std::enable_if< K == 4, bool >::type = true >
  point( T x, T y, T z, T w ) : m_value{ x, y, z, w } {}

  virtual ~point() = default;

  vector_type value() const { return m_value; }
  void set_value( vector_type v ) { m_value = v; }

  covariance_type covariance() const { return m_covariance; }
  void set_covariance( covariance_type const& v ) { m_covariance = v; }

  /// Equality operator
  bool operator==( point< N, T > const& other ) const
  {
    return this->value() == other.value();
  }

  /// Inequality operator
  bool operator!=( point< N, T > const& other ) const
  {
    return ! operator==(other);
  }

  T operator[]( size_t i ) { return m_value[i]; }
  const T operator[]( size_t i ) const { return m_value[i]; }

protected:
  vector_type m_value = vector_type::Zero();
  covariance_type m_covariance;
};

// Aliases for common types.
using point_2i = point< 2, int >;
using point_2d = point< 2, double >;
using point_2f = point< 2, float >;
using point_3d = point< 3, double >;
using point_3f = point< 3, float >;
using point_4d = point< 4, double >;
using point_4f = point< 4, float >;

VITAL_EXPORT ::std::ostream& operator<<( ::std::ostream&, point_2i const& );
VITAL_EXPORT ::std::ostream& operator<<( ::std::ostream&, point_2d const& );
VITAL_EXPORT ::std::ostream& operator<<( ::std::ostream&, point_2f const& );
VITAL_EXPORT ::std::ostream& operator<<( ::std::ostream&, point_3d const& );
VITAL_EXPORT ::std::ostream& operator<<( ::std::ostream&, point_3f const& );
VITAL_EXPORT ::std::ostream& operator<<( ::std::ostream&, point_4d const& );
VITAL_EXPORT ::std::ostream& operator<<( ::std::ostream&, point_4f const& );

} // namespace vital
} // namespace kwiver

#endif
