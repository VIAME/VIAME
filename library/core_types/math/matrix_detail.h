/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Determinant and inverse for the sizes VIAME uses
///
/// Two, three and four are closed forms -- cofactor expansion for the
/// determinant, the adjugate over it for the inverse -- because at those
/// sizes that is both faster and more accurate than elimination, and because
/// the closed forms are what Eigen uses too, so the last bit agrees. Anything
/// larger goes through Gauss-Jordan with partial pivoting.

#ifndef VIAME_CORE_TYPES_MATH_MATRIX_DETAIL_H_
#define VIAME_CORE_TYPES_MATH_MATRIX_DETAIL_H_

namespace kwiver {

namespace vital {

namespace math_detail {

template < unsigned N, typename T >
T determinant_square( matrix_< N, N, T > const& m );

template < typename T >
inline T determinant_square( matrix_< 1, 1, T > const& m )
{ return m( 0, 0 ); }

template < typename T >
inline T determinant_square( matrix_< 2, 2, T > const& m )
{ return m( 0, 0 ) * m( 1, 1 ) - m( 0, 1 ) * m( 1, 0 ); }

template < typename T >
inline T determinant_square( matrix_< 3, 3, T > const& m )
{
  return m( 0, 0 ) * ( m( 1, 1 ) * m( 2, 2 ) - m( 1, 2 ) * m( 2, 1 ) ) -
         m( 0, 1 ) * ( m( 1, 0 ) * m( 2, 2 ) - m( 1, 2 ) * m( 2, 0 ) ) +
         m( 0, 2 ) * ( m( 1, 0 ) * m( 2, 1 ) - m( 1, 1 ) * m( 2, 0 ) );
}

template < typename T >
inline T determinant_square( matrix_< 4, 4, T > const& m )
{
  // The six 2x2 minors of the bottom two rows, each used twice.
  T const s0 = m( 2, 0 ) * m( 3, 1 ) - m( 2, 1 ) * m( 3, 0 );
  T const s1 = m( 2, 0 ) * m( 3, 2 ) - m( 2, 2 ) * m( 3, 0 );
  T const s2 = m( 2, 0 ) * m( 3, 3 ) - m( 2, 3 ) * m( 3, 0 );
  T const s3 = m( 2, 1 ) * m( 3, 2 ) - m( 2, 2 ) * m( 3, 1 );
  T const s4 = m( 2, 1 ) * m( 3, 3 ) - m( 2, 3 ) * m( 3, 1 );
  T const s5 = m( 2, 2 ) * m( 3, 3 ) - m( 2, 3 ) * m( 3, 2 );

  return m( 0, 0 ) * ( m( 1, 1 ) * s5 - m( 1, 2 ) * s4 + m( 1, 3 ) * s3 ) -
         m( 0, 1 ) * ( m( 1, 0 ) * s5 - m( 1, 2 ) * s2 + m( 1, 3 ) * s1 ) +
         m( 0, 2 ) * ( m( 1, 0 ) * s4 - m( 1, 1 ) * s2 + m( 1, 3 ) * s0 ) -
         m( 0, 3 ) * ( m( 1, 0 ) * s3 - m( 1, 1 ) * s1 + m( 1, 2 ) * s0 );
}

// ----------------------------------------------------------------------------
template < unsigned N, typename T >
matrix_< N, N, T > inverse_square( matrix_< N, N, T > const& m );

template < typename T >
inline matrix_< 1, 1, T > inverse_square( matrix_< 1, 1, T > const& m )
{
  matrix_< 1, 1, T > out;
  out( 0, 0 ) = T( 1 ) / m( 0, 0 );
  return out;
}

template < typename T >
inline matrix_< 2, 2, T > inverse_square( matrix_< 2, 2, T > const& m )
{
  T const det = determinant_square( m );
  matrix_< 2, 2, T > out;
  out( 0, 0 ) =  m( 1, 1 ) / det;
  out( 0, 1 ) = -m( 0, 1 ) / det;
  out( 1, 0 ) = -m( 1, 0 ) / det;
  out( 1, 1 ) =  m( 0, 0 ) / det;
  return out;
}

template < typename T >
inline matrix_< 3, 3, T > inverse_square( matrix_< 3, 3, T > const& m )
{
  matrix_< 3, 3, T > adj;
  adj( 0, 0 ) = m( 1, 1 ) * m( 2, 2 ) - m( 1, 2 ) * m( 2, 1 );
  adj( 0, 1 ) = m( 0, 2 ) * m( 2, 1 ) - m( 0, 1 ) * m( 2, 2 );
  adj( 0, 2 ) = m( 0, 1 ) * m( 1, 2 ) - m( 0, 2 ) * m( 1, 1 );
  adj( 1, 0 ) = m( 1, 2 ) * m( 2, 0 ) - m( 1, 0 ) * m( 2, 2 );
  adj( 1, 1 ) = m( 0, 0 ) * m( 2, 2 ) - m( 0, 2 ) * m( 2, 0 );
  adj( 1, 2 ) = m( 0, 2 ) * m( 1, 0 ) - m( 0, 0 ) * m( 1, 2 );
  adj( 2, 0 ) = m( 1, 0 ) * m( 2, 1 ) - m( 1, 1 ) * m( 2, 0 );
  adj( 2, 1 ) = m( 0, 1 ) * m( 2, 0 ) - m( 0, 0 ) * m( 2, 1 );
  adj( 2, 2 ) = m( 0, 0 ) * m( 1, 1 ) - m( 0, 1 ) * m( 1, 0 );

  T const det = m( 0, 0 ) * adj( 0, 0 ) + m( 0, 1 ) * adj( 1, 0 ) +
                m( 0, 2 ) * adj( 2, 0 );
  return adj / det;
}

template < typename T >
inline matrix_< 4, 4, T > inverse_square( matrix_< 4, 4, T > const& m )
{
  // The 2x2 minors of the top two rows and of the bottom two, which the
  // adjugate's sixteen cofactors are built from.
  T const s0 = m( 0, 0 ) * m( 1, 1 ) - m( 0, 1 ) * m( 1, 0 );
  T const s1 = m( 0, 0 ) * m( 1, 2 ) - m( 0, 2 ) * m( 1, 0 );
  T const s2 = m( 0, 0 ) * m( 1, 3 ) - m( 0, 3 ) * m( 1, 0 );
  T const s3 = m( 0, 1 ) * m( 1, 2 ) - m( 0, 2 ) * m( 1, 1 );
  T const s4 = m( 0, 1 ) * m( 1, 3 ) - m( 0, 3 ) * m( 1, 1 );
  T const s5 = m( 0, 2 ) * m( 1, 3 ) - m( 0, 3 ) * m( 1, 2 );

  T const c5 = m( 2, 2 ) * m( 3, 3 ) - m( 2, 3 ) * m( 3, 2 );
  T const c4 = m( 2, 1 ) * m( 3, 3 ) - m( 2, 3 ) * m( 3, 1 );
  T const c3 = m( 2, 1 ) * m( 3, 2 ) - m( 2, 2 ) * m( 3, 1 );
  T const c2 = m( 2, 0 ) * m( 3, 3 ) - m( 2, 3 ) * m( 3, 0 );
  T const c1 = m( 2, 0 ) * m( 3, 2 ) - m( 2, 2 ) * m( 3, 0 );
  T const c0 = m( 2, 0 ) * m( 3, 1 ) - m( 2, 1 ) * m( 3, 0 );

  T const det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
  T const inv = T( 1 ) / det;

  matrix_< 4, 4, T > out;
  out( 0, 0 ) = (  m( 1, 1 ) * c5 - m( 1, 2 ) * c4 + m( 1, 3 ) * c3 ) * inv;
  out( 0, 1 ) = ( -m( 0, 1 ) * c5 + m( 0, 2 ) * c4 - m( 0, 3 ) * c3 ) * inv;
  out( 0, 2 ) = (  m( 3, 1 ) * s5 - m( 3, 2 ) * s4 + m( 3, 3 ) * s3 ) * inv;
  out( 0, 3 ) = ( -m( 2, 1 ) * s5 + m( 2, 2 ) * s4 - m( 2, 3 ) * s3 ) * inv;

  out( 1, 0 ) = ( -m( 1, 0 ) * c5 + m( 1, 2 ) * c2 - m( 1, 3 ) * c1 ) * inv;
  out( 1, 1 ) = (  m( 0, 0 ) * c5 - m( 0, 2 ) * c2 + m( 0, 3 ) * c1 ) * inv;
  out( 1, 2 ) = ( -m( 3, 0 ) * s5 + m( 3, 2 ) * s2 - m( 3, 3 ) * s1 ) * inv;
  out( 1, 3 ) = (  m( 2, 0 ) * s5 - m( 2, 2 ) * s2 + m( 2, 3 ) * s1 ) * inv;

  out( 2, 0 ) = (  m( 1, 0 ) * c4 - m( 1, 1 ) * c2 + m( 1, 3 ) * c0 ) * inv;
  out( 2, 1 ) = ( -m( 0, 0 ) * c4 + m( 0, 1 ) * c2 - m( 0, 3 ) * c0 ) * inv;
  out( 2, 2 ) = (  m( 3, 0 ) * s4 - m( 3, 1 ) * s2 + m( 3, 3 ) * s0 ) * inv;
  out( 2, 3 ) = ( -m( 2, 0 ) * s4 + m( 2, 1 ) * s2 - m( 2, 3 ) * s0 ) * inv;

  out( 3, 0 ) = ( -m( 1, 0 ) * c3 + m( 1, 1 ) * c1 - m( 1, 2 ) * c0 ) * inv;
  out( 3, 1 ) = (  m( 0, 0 ) * c3 - m( 0, 1 ) * c1 + m( 0, 2 ) * c0 ) * inv;
  out( 3, 2 ) = ( -m( 3, 0 ) * s3 + m( 3, 1 ) * s1 - m( 3, 2 ) * s0 ) * inv;
  out( 3, 3 ) = (  m( 2, 0 ) * s3 - m( 2, 1 ) * s1 + m( 2, 2 ) * s0 ) * inv;
  return out;
}

} // namespace math_detail

// ----------------------------------------------------------------------------
template < unsigned R, unsigned C, typename T >
T matrix_< R, C, T >::determinant() const
{
  static_assert( R == C, "a determinant needs a square matrix" );
  return math_detail::determinant_square( *this );
}

template < unsigned R, unsigned C, typename T >
matrix_< R, C, T > matrix_< R, C, T >::inverse() const
{
  static_assert( R == C, "an inverse needs a square matrix" );
  return math_detail::inverse_square( *this );
}

} // namespace vital

} // namespace kwiver

#endif
