/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief An axis-aligned box, for `bounding_box`
///
/// The member names are Eigen's `AlignedBox` -- `min()`, `max()`, `sizes()`,
/// `center()`, `volume()`, `contains()`, `extend()`, `translate()`,
/// `isEmpty()`, `setEmpty()` -- so that `core_types/bounding_box` changes its
/// include and nothing else.
///
/// Empty is the inverted box, min above max, which is what Eigen uses and what
/// makes `extend()` on a fresh box do the right thing.

#ifndef VIAME_CORE_TYPES_MATH_ALIGNED_BOX_H_
#define VIAME_CORE_TYPES_MATH_ALIGNED_BOX_H_

#include "vector.h"

#include <limits>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
template < typename T, unsigned N >
class aligned_box
{
public:
  using point_type = vector_< N, T >;

  /// The empty box.
  aligned_box() { setEmpty(); }

  aligned_box( point_type const& low, point_type const& high )
    : min_( low ), max_( high ) {}

  template < typename U >
  explicit aligned_box( aligned_box< U, N > const& other )
  {
    for( unsigned i = 0; i < N; ++i )
    {
      min_[ i ] = static_cast< T >( other.min()[ i ] );
      max_[ i ] = static_cast< T >( other.max()[ i ] );
    }
  }

  point_type& min()             { return min_; }
  point_type const& min() const { return min_; }
  point_type& max()             { return max_; }
  point_type const& max() const { return max_; }

  point_type sizes() const { return max_ - min_; }

  point_type center() const { return ( min_ + max_ ) / T( 2 ); }

  T volume() const
  {
    point_type const s = sizes();
    T out = T( 1 );
    for( unsigned i = 0; i < N; ++i ) { out *= s[ i ]; }
    return out;
  }

  bool isEmpty() const
  {
    for( unsigned i = 0; i < N; ++i ) { if( min_[ i ] > max_[ i ] ) { return true; } }
    return false;
  }

  void setEmpty()
  {
    for( unsigned i = 0; i < N; ++i )
    {
      min_[ i ] = std::numeric_limits< T >::max();
      max_[ i ] = std::numeric_limits< T >::lowest();
    }
  }

  bool contains( point_type const& p ) const
  {
    for( unsigned i = 0; i < N; ++i )
    {
      if( p[ i ] < min_[ i ] || p[ i ] > max_[ i ] ) { return false; }
    }
    return true;
  }

  bool contains( aligned_box const& b ) const
  { return contains( b.min() ) && contains( b.max() ); }

  bool intersects( aligned_box const& b ) const
  {
    for( unsigned i = 0; i < N; ++i )
    {
      if( b.max()[ i ] < min_[ i ] || b.min()[ i ] > max_[ i ] ) { return false; }
    }
    return true;
  }

  aligned_box intersection( aligned_box const& b ) const
  {
    aligned_box out;
    for( unsigned i = 0; i < N; ++i )
    {
      out.min_[ i ] = min_[ i ] > b.min()[ i ] ? min_[ i ] : b.min()[ i ];
      out.max_[ i ] = max_[ i ] < b.max()[ i ] ? max_[ i ] : b.max()[ i ];
    }
    return out;
  }

  aligned_box& extend( point_type const& p )
  {
    for( unsigned i = 0; i < N; ++i )
    {
      if( p[ i ] < min_[ i ] ) { min_[ i ] = p[ i ]; }
      if( p[ i ] > max_[ i ] ) { max_[ i ] = p[ i ]; }
    }
    return *this;
  }

  aligned_box& extend( aligned_box const& b )
  {
    if( b.isEmpty() ) { return *this; }
    extend( b.min() );
    extend( b.max() );
    return *this;
  }

  aligned_box& translate( point_type const& t )
  {
    min_ += t;
    max_ += t;
    return *this;
  }

  bool operator==( aligned_box const& o ) const
  { return min_ == o.min_ && max_ == o.max_; }

  bool operator!=( aligned_box const& o ) const { return !( *this == o ); }

private:
  point_type min_;
  point_type max_;
};

} // namespace vital

} // namespace kwiver

#endif
