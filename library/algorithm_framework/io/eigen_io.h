// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Stream input and serialization for the math types
///
/// The reading operator throws rather than setting failbit, which is what the
/// camera and landmark readers rely on to report a malformed file, and it is
/// why this cannot simply be the `operator>>` in `core_types/math`.
///
/// Phase 6 replaced Eigen with `core_types/math`; the file keeps its name
/// until phase 8 renames the tree, so that a hundred includes do not move for
/// nothing.

#ifndef VITAL_EIGEN_IO_H_
#define VITAL_EIGEN_IO_H_

#include <cstring>
#include <iostream>

#include <viame/core_types/math/dynamic_vector.h>
#include <viame/core_types/math/matrix.h>
#include <viame/core_types/math/vector.h>

#include <viame/algorithm_framework/exceptions/io.h>

namespace kwiver {

namespace vital {

/// Input stream operator for a fixed-size matrix
///
/// \throws vital::invalid_data
///    when the data being read is not in the valid form or format, such as a
///    character where a double should be
template < unsigned R, unsigned C, typename T >
std::istream&
operator>>( std::istream& s, matrix_< R, C, T >& m )
{
  for( unsigned i = 0; i < R; ++i )
  {
    for( unsigned j = 0; j < C; ++j )
    {
      if( !( s >> std::skipws >> m( i, j ) ) )
      {
        VITAL_THROW(
          kwiver::vital::invalid_data, "Encountered a non-numeric value while "
                                       "parsing a matrix" );
      }
    }
  }
  return s;
}

/// Input stream operator for a fixed-size vector
template < unsigned N, typename T >
std::istream&
operator>>( std::istream& s, vector_< N, T >& v )
{
  for( unsigned i = 0; i < N; ++i )
  {
    if( !( s >> std::skipws >> v[ i ] ) )
    {
      VITAL_THROW(
        kwiver::vital::invalid_data, "Encountered a non-numeric value while "
                                     "parsing a vector" );
    }
  }
  return s;
}

/// Input stream operator for a vector whose length the stream decides
///
/// Reads to the end of the stream, which is what the camera reader wants: the
/// distortion coefficients are the last thing in the file and there are as
/// many as the model has.
template < typename T >
std::istream&
operator>>( std::istream& s, dynamic_vector< T >& v )
{
  std::vector< T > values;
  T value;
  while( s >> std::skipws >> value ) { values.push_back( value ); }

  if( s.bad() )
  {
    VITAL_THROW(
      kwiver::vital::invalid_data, "Encountered a non-numeric value while "
                                   "parsing a vector" );
  }

  s.clear( s.rdstate() & ~std::ios::failbit );
  v = dynamic_vector< T >::Map( values.data(), values.size() );
  return s;
}

/// Serialization of a fixed-size matrix
template < class Archive, unsigned R, unsigned C, typename T >
void
serialize( Archive& archive, matrix_< R, C, T >& m )
{
  for( unsigned i = 0; i < R; ++i )
  {
    for( unsigned j = 0; j < C; ++j ) { archive( m( i, j ) ); }
  }
}

/// Serialization of a fixed-size vector
template < class Archive, unsigned N, typename T >
void
serialize( Archive& archive, vector_< N, T >& v )
{
  for( unsigned i = 0; i < N; ++i ) { archive( v[ i ] ); }
}

} // namespace vital

} // namespace kwiver

#endif // VITAL_EIGEN_IO_H_
