// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_UTIL_RATIONAL_H
#define VITAL_UTIL_RATIONAL_H

#include <numeric>
#include <ostream>
#include <stdexcept>

namespace kwiver {

namespace vital {

/**
 * \class rational rational.h <vital/util/rational.h>
 *
 * \brief A simple rational number class.
 *
 * This provides a minimal implementation of rational numbers to replace
 * boost::rational. The rational number is always kept in normalized form
 * (lowest terms with positive denominator).
 */
template < typename IntType >
class rational
{
public:
  typedef IntType int_type;

  /// Default constructor - creates 0/1
  rational()
    : m_numerator( 0 ),
      m_denominator( 1 )
  {}

  /// Constructor from integer - creates n/1
  rational( IntType n )
    : m_numerator( n ),
      m_denominator( 1 )
  {}

  /// Constructor from numerator and denominator
  rational( IntType n, IntType d )
    : m_numerator( n ),
      m_denominator( d )
  {
    if( d == 0 )
    {
      throw std::domain_error( "rational: zero denominator" );
    }
    normalize();
  }

  /// Get numerator
  IntType
  numerator() const { return m_numerator; }

  /// Get denominator
  IntType
  denominator() const { return m_denominator; }

  /// Comparison operators
  bool
  operator==( rational const& rhs ) const
  {
    return m_numerator == rhs.m_numerator && m_denominator == rhs.m_denominator;
  }

  bool
  operator!=( rational const& rhs ) const
  {
    return !( *this == rhs );
  }

  bool
  operator<( rational const& rhs ) const
  {
    // a/b < c/d  <=>  a*d < c*b (when b,d > 0, which is ensured by normalize)
    return m_numerator * rhs.m_denominator < rhs.m_numerator * m_denominator;
  }

  bool
  operator>( rational const& rhs ) const
  {
    return rhs < *this;
  }

  bool
  operator<=( rational const& rhs ) const
  {
    return !( rhs < *this );
  }

  bool
  operator>=( rational const& rhs ) const
  {
    return !( *this < rhs );
  }

  /// Arithmetic operators
  rational
  operator+( rational const& rhs ) const
  {
    IntType num = m_numerator * rhs.m_denominator + rhs.m_numerator *
                  m_denominator;
    IntType den = m_denominator * rhs.m_denominator;
    return rational( num, den );
  }

  rational
  operator-( rational const& rhs ) const
  {
    IntType num = m_numerator * rhs.m_denominator - rhs.m_numerator *
                  m_denominator;
    IntType den = m_denominator * rhs.m_denominator;
    return rational( num, den );
  }

  rational
  operator*( rational const& rhs ) const
  {
    return rational(
      m_numerator * rhs.m_numerator,
      m_denominator * rhs.m_denominator );
  }

  rational
  operator/( rational const& rhs ) const
  {
    if( rhs.m_numerator == 0 )
    {
      throw std::domain_error( "rational: division by zero" );
    }
    return rational(
      m_numerator * rhs.m_denominator,
      m_denominator * rhs.m_numerator );
  }

  /// Compound assignment operators
  rational&
  operator+=( rational const& rhs )
  {
    *this = *this + rhs;
    return *this;
  }

  rational&
  operator-=( rational const& rhs )
  {
    *this = *this - rhs;
    return *this;
  }

  rational&
  operator*=( rational const& rhs )
  {
    *this = *this * rhs;
    return *this;
  }

  rational&
  operator/=( rational const& rhs )
  {
    *this = *this / rhs;
    return *this;
  }

  /// Conversion to bool (true if non-zero)
  explicit
  operator bool() const
  {
    return m_numerator != 0;
  }

private:
  void
  normalize()
  {
    // Ensure denominator is positive
    if( m_denominator < 0 )
    {
      m_numerator = -m_numerator;
      m_denominator = -m_denominator;
    }

    // Reduce to lowest terms
    IntType g = std::gcd(
      m_numerator < 0 ? -m_numerator : m_numerator,
      m_denominator );
    if( g > 1 )
    {
      m_numerator /= g;
      m_denominator /= g;
    }
  }

  IntType m_numerator;
  IntType m_denominator;
};

/// Compute LCM of denominators for a set of rationals
template < typename IntType >
IntType
rational_lcm( rational< IntType > const& a, rational< IntType > const& b )
{
  return std::lcm( a.denominator(), b.denominator() );
}

/// Stream output operator
template < typename IntType >
std::ostream&
operator<<( std::ostream& os, rational< IntType > const& r )
{
  os << r.numerator();
  if( r.denominator() != 1 )
  {
    os << "/" << r.denominator();
  }
  return os;
}

/// Multiplication with integer on the left: int * rational
template < typename IntType >
rational< IntType > operator*( IntType lhs, rational< IntType > const& rhs )
{
  return rational< IntType >( lhs ) * rhs;
}

} // namespace vital

} // namespace kwiver

#endif // VITAL_UTIL_RATIONAL_H
