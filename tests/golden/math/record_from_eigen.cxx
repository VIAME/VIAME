/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Records what Eigen answers, so that the replacement can be checked
///        after Eigen is gone
///
/// Run once, by hand, while phase 6 still has Eigen on the include path:
///
///     g++ -std=c++17 -O2 -o record record_from_eigen.cxx -I<eigen3>
///     ./record > eigen.json
///
/// It is not built by CMake. Eigen is the thing being removed, so a target
/// that needs it would have to be deleted in the same change that makes the
/// recording worth having, and then nobody could ever regenerate the file to
/// see how it was made.
///
/// Every number is printed with `%.17g`, which round-trips a double exactly.
/// The inputs come from a fixed seed so that a regenerated file differs only
/// where the answer differs.

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

std::mt19937 rng( 20260910 );
std::uniform_real_distribution< double > uni( -2.0, 2.0 );

void emit( double x ) { std::printf( "%.17g", x ); }

void emit_flat( Eigen::MatrixXd const& m )
{
  // Row-major, which is how a human reads a matrix and how the test reads it
  // back.
  std::printf( "[" );
  for( int r = 0; r < m.rows(); ++r )
  {
    for( int c = 0; c < m.cols(); ++c )
    {
      if( r || c ) { std::printf( ", " ); }
      emit( m( r, c ) );
    }
  }
  std::printf( "]" );
}

void emit_vec( std::vector< double > const& v )
{
  std::printf( "[" );
  for( size_t i = 0; i < v.size(); ++i )
  {
    if( i ) { std::printf( ", " ); }
    emit( v[ i ] );
  }
  std::printf( "]" );
}

Eigen::MatrixXd random_matrix( int rows, int cols )
{
  Eigen::MatrixXd m( rows, cols );
  for( int r = 0; r < rows; ++r )
  {
    for( int c = 0; c < cols; ++c ) { m( r, c ) = uni( rng ); }
  }
  return m;
}

// ----------------------------------------------------------------------------
void record_square( int n, int count, bool& first )
{
  for( int i = 0; i < count; ++i )
  {
    Eigen::MatrixXd m = random_matrix( n, n );
    // Away from singular, so that the inverse is a fair comparison.
    for( int d = 0; d < n; ++d ) { m( d, d ) += 4.0; }

    if( !first ) { std::printf( ",\n" ); }
    first = false;
    std::printf( "    { \"n\": %d, \"a\": ", n );
    emit_flat( m );
    std::printf( ", \"det\": " );
    emit( m.determinant() );
    std::printf( ", \"inverse\": " );
    emit_flat( m.inverse() );
    std::printf( " }" );
  }
}

void record_svd( int rows, int cols, int count, bool& first )
{
  for( int i = 0; i < count; ++i )
  {
    Eigen::MatrixXd const a = random_matrix( rows, cols );
    Eigen::JacobiSVD< Eigen::MatrixXd > svd(
      a, Eigen::ComputeThinU | Eigen::ComputeFullV );

    std::vector< double > s( cols );
    for( int j = 0; j < cols; ++j ) { s[ j ] = svd.singularValues()( j ); }

    std::vector< double > nv( cols );
    for( int j = 0; j < cols; ++j ) { nv[ j ] = svd.matrixV()( j, cols - 1 ); }

    if( !first ) { std::printf( ",\n" ); }
    first = false;
    std::printf( "    { \"rows\": %d, \"cols\": %d, \"a\": ", rows, cols );
    emit_flat( a );
    std::printf( ", \"singular_values\": " );
    emit_vec( s );
    std::printf( ", \"null_vector\": " );
    emit_vec( nv );
    std::printf( " }" );
  }
}

void record_symmetric_eigen( int count, bool& first )
{
  for( int i = 0; i < count; ++i )
  {
    Eigen::Matrix3d m;
    for( int c = 0; c < 3; ++c )
    {
      for( int r = c; r < 3; ++r ) { m( r, c ) = m( c, r ) = uni( rng ); }
    }
    Eigen::SelfAdjointEigenSolver< Eigen::Matrix3d > es( m );

    std::vector< double > values( 3 );
    for( int j = 0; j < 3; ++j ) { values[ j ] = es.eigenvalues()( j ); }

    if( !first ) { std::printf( ",\n" ); }
    first = false;
    std::printf( "    { \"a\": " );
    emit_flat( m );
    std::printf( ", \"eigenvalues\": " );
    emit_vec( values );
    std::printf( " }" );
  }
}

void record_cholesky( int count, bool& first )
{
  for( int i = 0; i < count; ++i )
  {
    Eigen::Matrix3d const raw = random_matrix( 3, 3 );
    Eigen::Matrix3d a = raw * raw.transpose();
    for( int d = 0; d < 3; ++d ) { a( d, d ) += 1.0; }

    Eigen::Matrix3d const l = a.llt().matrixL();

    if( !first ) { std::printf( ",\n" ); }
    first = false;
    std::printf( "    { \"a\": " );
    emit_flat( a );
    std::printf( ", \"l\": " );
    emit_flat( l );
    std::printf( " }" );
  }
}

void record_least_squares( int rows, int cols, int count, bool& first )
{
  for( int i = 0; i < count; ++i )
  {
    Eigen::MatrixXd const a = random_matrix( rows, cols );
    Eigen::VectorXd b( rows );
    for( int r = 0; r < rows; ++r ) { b( r ) = uni( rng ); }

    Eigen::VectorXd const x =
      a.jacobiSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve( b );

    std::vector< double > bv( rows ), xv( cols );
    for( int r = 0; r < rows; ++r ) { bv[ r ] = b( r ); }
    for( int c = 0; c < cols; ++c ) { xv[ c ] = x( c ); }

    if( !first ) { std::printf( ",\n" ); }
    first = false;
    std::printf( "    { \"rows\": %d, \"cols\": %d, \"a\": ", rows, cols );
    emit_flat( a );
    std::printf( ", \"b\": " );
    emit_vec( bv );
    std::printf( ", \"x\": " );
    emit_vec( xv );
    std::printf( " }" );
  }
}

void record_rotation( int count, bool& first )
{
  for( int i = 0; i < count; ++i )
  {
    Eigen::Vector3d axis( uni( rng ), uni( rng ), uni( rng ) );
    if( axis.norm() < 1e-6 ) { axis = Eigen::Vector3d( 1, 0, 0 ); }
    double const angle = uni( rng );

    Eigen::Quaterniond const q( Eigen::AngleAxisd( angle, axis.normalized() ) );
    Eigen::Vector3d const v( uni( rng ), uni( rng ), uni( rng ) );
    Eigen::Vector3d const rotated = q * v;

    Eigen::Vector3d axis2( uni( rng ), uni( rng ), uni( rng ) );
    double const angle2 = uni( rng );
    Eigen::Quaterniond const q2(
      Eigen::AngleAxisd( angle2, axis2.normalized() ) );
    Eigen::Quaterniond const composed = q * q2;

    if( !first ) { std::printf( ",\n" ); }
    first = false;
    std::printf( "    { \"axis\": [%.17g, %.17g, %.17g], \"angle\": %.17g",
                 axis( 0 ), axis( 1 ), axis( 2 ), angle );
    std::printf( ", \"quaternion_xyzw\": [%.17g, %.17g, %.17g, %.17g]",
                 q.x(), q.y(), q.z(), q.w() );
    std::printf( ", \"matrix\": " );
    emit_flat( q.toRotationMatrix() );
    std::printf( ", \"vector\": [%.17g, %.17g, %.17g]", v( 0 ), v( 1 ), v( 2 ) );
    std::printf( ", \"rotated\": [%.17g, %.17g, %.17g]",
                 rotated( 0 ), rotated( 1 ), rotated( 2 ) );
    std::printf( ", \"axis2\": [%.17g, %.17g, %.17g], \"angle2\": %.17g",
                 axis2( 0 ), axis2( 1 ), axis2( 2 ), angle2 );
    std::printf( ", \"composed_xyzw\": [%.17g, %.17g, %.17g, %.17g]",
                 composed.x(), composed.y(), composed.z(), composed.w() );
    std::printf( ", \"angular_distance\": %.17g", q.angularDistance( q2 ) );
    std::printf( " }" );
  }
}

} // namespace

int main()
{
  std::printf( "{\n" );
  std::printf( "  \"note\": \"Recorded from Eigen by "
               "tests/golden/math/record_from_eigen.cxx, seed 20260910. "
               "Row-major.\",\n" );

  bool first = true;
  std::printf( "  \"square\": [\n" );
  record_square( 2, 40, first );
  record_square( 3, 40, first );
  record_square( 4, 40, first );
  std::printf( "\n  ],\n" );

  first = true;
  std::printf( "  \"svd\": [\n" );
  record_svd( 3, 3, 20, first );
  record_svd( 4, 4, 20, first );
  record_svd( 6, 4, 20, first );
  record_svd( 12, 4, 20, first );
  std::printf( "\n  ],\n" );

  first = true;
  std::printf( "  \"symmetric_eigen\": [\n" );
  record_symmetric_eigen( 40, first );
  std::printf( "\n  ],\n" );

  first = true;
  std::printf( "  \"cholesky\": [\n" );
  record_cholesky( 40, first );
  std::printf( "\n  ],\n" );

  first = true;
  std::printf( "  \"least_squares\": [\n" );
  record_least_squares( 8, 3, 20, first );
  record_least_squares( 20, 5, 20, first );
  std::printf( "\n  ],\n" );

  first = true;
  std::printf( "  \"rotation\": [\n" );
  record_rotation( 60, first );
  std::printf( "\n  ]\n" );

  std::printf( "}\n" );
  return 0;
}
