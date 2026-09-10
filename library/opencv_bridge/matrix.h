// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Conversion between cv::Mat and vital's fixed-size matrices
///
/// This replaces the two functions OpenCV's `opencv2/core/eigen.hpp` used to
/// supply, `cv::cv2eigen` and `cv::eigen2cv`. That header is unusable once
/// Eigen is gone -- it is guarded on `EIGEN_WORLD_VERSION` and #errors out
/// when Eigen's headers have not been included first -- and only the fixed
/// size, double precision cases were ever called here.

#ifndef VIAME_OPENCV_BRIDGE_MATRIX_H
#define VIAME_OPENCV_BRIDGE_MATRIX_H

#include <opencv2/core.hpp>
#include <viame/core_types/matrix.h>
#include <viame/core_types/vector.h>

#include <sstream>
#include <stdexcept>

namespace kwiver {

namespace arrows {

namespace ocv {

/// @brief Copy a cv::Mat into a fixed-size vital matrix
///
/// The source must have exactly \c R rows and \c C columns and a single
/// channel; unlike `cv::cv2eigen`, which asserts, a mismatch throws so that
/// a release build reports it rather than reading out of bounds. Any of
/// OpenCV's real element types is accepted and converted to \c T.
///
/// @param src cv::Mat to convert
/// @param dst matrix to fill
template < unsigned R, unsigned C, typename T >
void
mat_to_matrix( cv::Mat const& src, kwiver::vital::matrix_< R, C, T >& dst )
{
  if( static_cast< unsigned >( src.rows ) != R ||
      static_cast< unsigned >( src.cols ) != C ||
      src.channels() != 1 )
  {
    std::ostringstream ss;
    ss << "mat_to_matrix: expected a " << R << "x" << C
       << " single channel matrix, got " << src.rows << "x" << src.cols
       << " with " << src.channels() << " channel(s)";
    throw std::runtime_error( ss.str() );
  }

  cv::Mat tmp;
  if( src.depth() == cv::DataType< T >::depth )
  {
    tmp = src;
  }
  else
  {
    src.convertTo( tmp, cv::DataType< T >::depth );
  }

  for( unsigned r = 0; r < R; ++r )
  {
    for( unsigned c = 0; c < C; ++c )
    {
      dst( r, c ) = tmp.at< T >( static_cast< int >( r ),
                                 static_cast< int >( c ) );
    }
  }
}

/// @brief Copy a fixed-size vital matrix into a cv::Mat
///
/// The destination is (re)allocated to \c R x \c C of \c T.
///
/// @param src matrix to convert
/// @param dst cv::Mat to fill
template < unsigned R, unsigned C, typename T >
void
matrix_to_mat( kwiver::vital::matrix_< R, C, T > const& src, cv::Mat& dst )
{
  dst.create( static_cast< int >( R ), static_cast< int >( C ),
              cv::DataType< T >::type );
  for( unsigned r = 0; r < R; ++r )
  {
    for( unsigned c = 0; c < C; ++c )
    {
      dst.at< T >( static_cast< int >( r ),
                   static_cast< int >( c ) ) = src( r, c );
    }
  }
}

/// @brief Copy a cv::Mat into a fixed-size vital vector
///
/// The source must be \c N x 1 or 1 x \c N and single channel.
///
/// @param src cv::Mat to convert
/// @param dst vector to fill
template < unsigned N, typename T >
void
mat_to_vector( cv::Mat const& src, kwiver::vital::vector_< N, T >& dst )
{
  bool const column = ( static_cast< unsigned >( src.rows ) == N &&
                        src.cols == 1 );
  bool const row = ( src.rows == 1 &&
                     static_cast< unsigned >( src.cols ) == N );
  if( ( !column && !row ) || src.channels() != 1 )
  {
    std::ostringstream ss;
    ss << "mat_to_vector: expected a single channel " << N
       << " element vector, got " << src.rows << "x" << src.cols
       << " with " << src.channels() << " channel(s)";
    throw std::runtime_error( ss.str() );
  }

  cv::Mat tmp;
  if( src.depth() == cv::DataType< T >::depth )
  {
    tmp = src;
  }
  else
  {
    src.convertTo( tmp, cv::DataType< T >::depth );
  }

  for( unsigned i = 0; i < N; ++i )
  {
    dst[ i ] = column
               ? tmp.at< T >( static_cast< int >( i ), 0 )
               : tmp.at< T >( 0, static_cast< int >( i ) );
  }
}

/// @brief Copy a fixed-size vital vector into a cv::Mat
///
/// The destination is (re)allocated to \c N x 1 of \c T, which is the shape
/// `cv::eigen2cv` produced for a column vector.
///
/// @param src vector to convert
/// @param dst cv::Mat to fill
template < unsigned N, typename T >
void
vector_to_mat( kwiver::vital::vector_< N, T > const& src, cv::Mat& dst )
{
  dst.create( static_cast< int >( N ), 1, cv::DataType< T >::type );
  for( unsigned i = 0; i < N; ++i )
  {
    dst.at< T >( static_cast< int >( i ), 0 ) = src[ i ];
  }
}

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
