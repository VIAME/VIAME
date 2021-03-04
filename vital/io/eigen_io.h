// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Missing istream operator for Eigen fixed sized matrices
 */

#ifndef VITAL_EIGEN_IO_H_
#define VITAL_EIGEN_IO_H_

#include <iostream>
#include <cstring>

#include <Eigen/Core>

#include <vital/exceptions/io.h>

namespace Eigen {

/// input stream operator for an Eigen matrix
/**
 * \throws vital::invalid_data
 *    Throws an invalid data exception when the data being read is either not
 *    in the valid form or format, e.g. read a character where a double should
 *    be..
 *
 * \param s an input stream
 * \param m a matrix to stream into
 */
template < typename T, int M, int N >
std::istream&
operator>>( std::istream& s, Matrix< T, M, N >& m )
{
  for ( int i = 0; i < M; ++i )
  {
    for ( int j = 0; j < N; ++j )
    {
      if ( ! ( s >> std::skipws >> m( i, j ) ) )
      {
        VITAL_THROW( kwiver::vital::invalid_data, "Encountered a non-numeric value while "
                                    "parsing an Eigen::Matrix" );
      }
    }
  }
  return s;
}

/// Serialization of fixed Eigen matrices
template < typename Archive, typename T, int M, int N, int O, int MM, int NN >
void serialize(Archive & archive, Matrix< T, M, N, O, MM, NN >& m)
{
  for ( int i = 0; i < M; ++i )
  {
    for ( int j = 0; j < N; ++j )
    {
      archive( m( i, j ) );
    }
  }
}

} // end namespace Eigen

#endif // VITAL_EIGEN_IO_H_
