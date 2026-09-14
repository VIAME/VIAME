/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief `core_types/math` against the values Eigen gave for the same inputs
///
/// `tests/golden/math/eigen.json` was recorded by
/// `tests/golden/math/record_from_eigen.cxx` while phase 6 still had Eigen,
/// which is the only moment those numbers can be obtained. Checking against
/// the file rather than against Eigen is what lets this test outlive the
/// dependency it is replacing.
///
/// The identities -- `A * inverse(A) == I`, `U diag(s) V^T == A`,
/// `A v == lambda v` -- are checked as well, because a recording only says
/// the two implementations agree, not that either is right.

#include <viame/core_types/math/decomp.h>
#include <viame/core_types/math/dynamic_matrix.h>
#include <viame/core_types/math/matrix.h>
#include <viame/core_types/math/quaternion.h>
#include <viame/core_types/math/vector.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
/// Just enough JSON for the recording: objects of numbers and flat arrays of
/// numbers. Pulling in a parser for one file of one shape is not worth the
/// dependency.
class golden
{
public:
  explicit golden( std::string const& path )
  {
    std::ifstream in( path );
    std::stringstream buffer;
    buffer << in.rdbuf();
    text_ = buffer.str();
    if( text_.empty() )
    {
      ADD_FAILURE() << "could not read " << path;
    }
  }

  /// The objects inside the top-level array called \p name.
  std::vector< std::string > section( std::string const& name ) const
  {
    std::vector< std::string > out;
    std::string const key = "\"" + name + "\"";
    size_t at = text_.find( key );
    if( at == std::string::npos ) { return out; }

    at = text_.find( '[', at );
    int depth = 0;
    size_t start = 0;
    for( size_t i = at; i < text_.size(); ++i )
    {
      char const ch = text_[ i ];
      if( ch == '{' )
      {
        if( depth == 0 ) { start = i; }
        ++depth;
      }
      else if( ch == '}' )
      {
        if( --depth == 0 ) { out.push_back( text_.substr( start, i - start + 1 ) ); }
      }
      else if( ch == ']' && depth == 0 )
      {
        break;
      }
    }
    return out;
  }

  static double number( std::string const& object, std::string const& key )
  {
    size_t at = object.find( "\"" + key + "\"" );
    EXPECT_NE( at, std::string::npos ) << key;
    at = object.find( ':', at ) + 1;
    return std::strtod( object.c_str() + at, nullptr );
  }

  static std::vector< double > numbers( std::string const& object,
                                        std::string const& key )
  {
    std::vector< double > out;
    size_t at = object.find( "\"" + key + "\"" );
    EXPECT_NE( at, std::string::npos ) << key;
    at = object.find( '[', at ) + 1;
    size_t const end = object.find( ']', at );

    char const* p = object.c_str() + at;
    char const* stop = object.c_str() + end;
    while( p < stop )
    {
      char* next = nullptr;
      double const v = std::strtod( p, &next );
      if( next == p ) { break; }
      out.push_back( v );
      p = next;
      while( p < stop && ( *p == ',' || *p == ' ' || *p == '\n' ) ) { ++p; }
    }
    return out;
  }

private:
  std::string text_;
};

/// The recording's directory, compiled in by CMake and overridable by the
/// environment so that a copy of the tree can be checked against another.
std::string golden_path()
{
  char const* dir = std::getenv( "VIAME_GOLDEN_MATH_DIR" );
  if( !dir ) { dir = VIAME_GOLDEN_MATH_DIR; }
  return std::string( dir ) + "/eigen.json";
}

/// A row-major list into a column-major dynamic matrix.
dynamic_matrix< double > to_matrix( std::vector< double > const& flat,
                                    unsigned rows, unsigned cols )
{
  dynamic_matrix< double > m( rows, cols );
  for( unsigned r = 0; r < rows; ++r )
  {
    for( unsigned c = 0; c < cols; ++c ) { m( r, c ) = flat[ r * cols + c ]; }
  }
  return m;
}

template < unsigned N >
matrix_< N, N, double > to_fixed( std::vector< double > const& flat )
{
  matrix_< N, N, double > m;
  for( unsigned r = 0; r < N; ++r )
  {
    for( unsigned c = 0; c < N; ++c ) { m( r, c ) = flat[ r * N + c ]; }
  }
  return m;
}

/// Same direction or the opposite one: a null vector and an eigenvector are
/// each defined up to sign, and the two implementations need not agree on it.
double up_to_sign( std::vector< double > const& a,
                   std::vector< double > const& b )
{
  double same = 0.0, flipped = 0.0;
  for( size_t i = 0; i < a.size(); ++i )
  {
    same = std::max( same, std::abs( a[ i ] - b[ i ] ) );
    flipped = std::max( flipped, std::abs( a[ i ] + b[ i ] ) );
  }
  return std::min( same, flipped );
}

} // namespace

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST ( math, determinant_and_inverse_match_eigen )
{
  golden g( golden_path() );
  auto const cases = g.section( "square" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    unsigned const n = static_cast< unsigned >( golden::number( c, "n" ) );
    auto const a = golden::numbers( c, "a" );
    auto const want_inv = golden::numbers( c, "inverse" );
    double const want_det = golden::number( c, "det" );

    if( n == 2 )
    {
      auto const m = to_fixed< 2 >( a );
      EXPECT_NEAR( m.determinant(), want_det, 1e-11 );
      auto const inv = m.inverse();
      for( unsigned r = 0; r < 2; ++r )
      {
        for( unsigned col = 0; col < 2; ++col )
        {
          EXPECT_NEAR( inv( r, col ), want_inv[ r * 2 + col ], 1e-11 );
        }
      }
      auto const id = m * inv;
      for( unsigned r = 0; r < 2; ++r )
      {
        for( unsigned col = 0; col < 2; ++col )
        {
          EXPECT_NEAR( id( r, col ), r == col ? 1.0 : 0.0, 1e-12 );
        }
      }
    }
    else if( n == 3 )
    {
      auto const m = to_fixed< 3 >( a );
      EXPECT_NEAR( m.determinant(), want_det, 1e-11 );
      auto const inv = m.inverse();
      for( unsigned r = 0; r < 3; ++r )
      {
        for( unsigned col = 0; col < 3; ++col )
        {
          EXPECT_NEAR( inv( r, col ), want_inv[ r * 3 + col ], 1e-11 );
        }
      }
      auto const id = m * inv;
      for( unsigned r = 0; r < 3; ++r )
      {
        for( unsigned col = 0; col < 3; ++col )
        {
          EXPECT_NEAR( id( r, col ), r == col ? 1.0 : 0.0, 1e-12 );
        }
      }
    }
    else
    {
      auto const m = to_fixed< 4 >( a );
      EXPECT_NEAR( m.determinant(), want_det, 1e-10 );
      auto const inv = m.inverse();
      for( unsigned r = 0; r < 4; ++r )
      {
        for( unsigned col = 0; col < 4; ++col )
        {
          EXPECT_NEAR( inv( r, col ), want_inv[ r * 4 + col ], 1e-11 );
        }
      }
      auto const id = m * inv;
      for( unsigned r = 0; r < 4; ++r )
      {
        for( unsigned col = 0; col < 4; ++col )
        {
          EXPECT_NEAR( id( r, col ), r == col ? 1.0 : 0.0, 1e-12 );
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( math, singular_values_and_null_vector_match_eigen )
{
  golden g( golden_path() );
  auto const cases = g.section( "svd" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    unsigned const rows = static_cast< unsigned >( golden::number( c, "rows" ) );
    unsigned const cols = static_cast< unsigned >( golden::number( c, "cols" ) );
    auto const a = to_matrix( golden::numbers( c, "a" ), rows, cols );
    auto const want_s = golden::numbers( c, "singular_values" );
    auto const want_nv = golden::numbers( c, "null_vector" );

    jacobi_svd< double > svd( a );

    for( unsigned i = 0; i < cols; ++i )
    {
      EXPECT_NEAR( svd.singularValues()[ i ], want_s[ i ], 1e-11 );
    }

    EXPECT_LT( up_to_sign( svd.null_vector(), want_nv ), 1e-9 );

    // U diag(s) V^T reconstructs A, which the recording cannot tell us
    for( unsigned r = 0; r < rows; ++r )
    {
      for( unsigned col = 0; col < cols; ++col )
      {
        double sum = 0.0;
        for( unsigned k = 0; k < cols; ++k )
        {
          sum += svd.matrixU()( r, k ) * svd.singularValues()[ k ] *
                 svd.matrixV()( col, k );
        }
        EXPECT_NEAR( sum, a( r, col ), 1e-12 );
      }
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( math, symmetric_eigenvalues_match_eigen )
{
  golden g( golden_path() );
  auto const cases = g.section( "symmetric_eigen" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    auto const m = to_fixed< 3 >( golden::numbers( c, "a" ) );
    auto const want = golden::numbers( c, "eigenvalues" );

    jacobi_eigen_symmetric< 3, double > js( m );

    for( unsigned i = 0; i < 3; ++i )
    {
      EXPECT_NEAR( js.eigenvalues()[ i ], want[ i ], 1e-12 );
    }

    // A v == lambda v for each pair
    for( unsigned j = 0; j < 3; ++j )
    {
      auto const v = js.eigenvectors().col( j );
      auto const av = m * v;
      for( unsigned i = 0; i < 3; ++i )
      {
        EXPECT_NEAR( av[ i ], js.eigenvalues()[ j ] * v[ i ], 1e-12 );
      }
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( math, cholesky_matches_eigen )
{
  golden g( golden_path() );
  auto const cases = g.section( "cholesky" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    auto const a = to_fixed< 3 >( golden::numbers( c, "a" ) );
    auto const want = golden::numbers( c, "l" );

    matrix_< 3, 3, double > l;
    ASSERT_TRUE( cholesky( a, l ) );

    for( unsigned r = 0; r < 3; ++r )
    {
      for( unsigned col = 0; col < 3; ++col )
      {
        EXPECT_NEAR( l( r, col ), want[ r * 3 + col ], 1e-12 );
      }
    }

    auto const reconstructed = l * l.transpose();
    for( unsigned r = 0; r < 3; ++r )
    {
      for( unsigned col = 0; col < 3; ++col )
      {
        EXPECT_NEAR( reconstructed( r, col ), a( r, col ), 1e-12 );
      }
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( math, least_squares_matches_eigen )
{
  golden g( golden_path() );
  auto const cases = g.section( "least_squares" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    unsigned const rows = static_cast< unsigned >( golden::number( c, "rows" ) );
    unsigned const cols = static_cast< unsigned >( golden::number( c, "cols" ) );
    auto const a = to_matrix( golden::numbers( c, "a" ), rows, cols );
    auto const b = golden::numbers( c, "b" );
    auto const want = golden::numbers( c, "x" );

    auto const x = solve_least_squares( a, b );
    for( unsigned i = 0; i < cols; ++i )
    {
      EXPECT_NEAR( x[ i ], want[ i ], 1e-10 );
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( math, rotations_match_eigen )
{
  golden g( golden_path() );
  auto const cases = g.section( "rotation" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    auto const axis = golden::numbers( c, "axis" );
    double const angle = golden::number( c, "angle" );
    auto const want_q = golden::numbers( c, "quaternion_xyzw" );
    auto const want_m = golden::numbers( c, "matrix" );
    auto const v = golden::numbers( c, "vector" );
    auto const want_r = golden::numbers( c, "rotated" );
    auto const axis2 = golden::numbers( c, "axis2" );
    double const angle2 = golden::number( c, "angle2" );
    auto const want_c = golden::numbers( c, "composed_xyzw" );
    double const want_d = golden::number( c, "angular_distance" );

    auto const q = quaternion_d::from_axis_angle(
      angle, vector_3d( axis[ 0 ], axis[ 1 ], axis[ 2 ] ) );
    auto const q2 = quaternion_d::from_axis_angle(
      angle2, vector_3d( axis2[ 0 ], axis2[ 1 ], axis2[ 2 ] ) );

    std::vector< double > const got_q{ q.x(), q.y(), q.z(), q.w() };
    EXPECT_LT( up_to_sign( got_q, want_q ), 1e-14 );

    auto const m = q.toRotationMatrix();
    for( unsigned r = 0; r < 3; ++r )
    {
      for( unsigned col = 0; col < 3; ++col )
      {
        EXPECT_NEAR( m( r, col ), want_m[ r * 3 + col ], 1e-14 );
      }
    }

    auto const rotated = q * vector_3d( v[ 0 ], v[ 1 ], v[ 2 ] );
    for( unsigned i = 0; i < 3; ++i )
    {
      EXPECT_NEAR( rotated[ i ], want_r[ i ], 1e-13 );
    }

    auto const composed = q * q2;
    std::vector< double > const got_c{ composed.x(), composed.y(),
                                       composed.z(), composed.w() };
    EXPECT_LT( up_to_sign( got_c, want_c ), 1e-14 );

    EXPECT_NEAR( q.angularDistance( q2 ), want_d, 1e-12 );

    // A rotation matrix is orthogonal with determinant one, which the
    // recording does not say
    EXPECT_NEAR( m.determinant(), 1.0, 1e-13 );
    auto const should_be_identity = m * m.transpose();
    for( unsigned r = 0; r < 3; ++r )
    {
      for( unsigned col = 0; col < 3; ++col )
      {
        EXPECT_NEAR( should_be_identity( r, col ), r == col ? 1.0 : 0.0, 1e-14 );
      }
    }
  }
}

// ----------------------------------------------------------------------------
/// The things that need no recording because they are true by definition.
TEST ( math, identities )
{
  vector_3d const a( 1.0, 2.0, 3.0 );
  vector_3d const b( -4.0, 5.0, 6.0 );

  EXPECT_DOUBLE_EQ( a.dot( b ), -4.0 + 10.0 + 18.0 );
  EXPECT_DOUBLE_EQ( a.cross( b ).dot( a ), 0.0 );
  EXPECT_DOUBLE_EQ( a.cross( b ).dot( b ), 0.0 );
  EXPECT_NEAR( a.normalized().norm(), 1.0, 1e-15 );
  EXPECT_DOUBLE_EQ( ( a + b ).norm(), ( b + a ).norm() );

  EXPECT_DOUBLE_EQ( a.homogeneous()[ 3 ], 1.0 );
  auto const back = a.homogeneous().hnormalized();
  for( unsigned i = 0; i < 3; ++i ) { EXPECT_DOUBLE_EQ( back[ i ], a[ i ] ); }

  auto const identity = matrix_3x3d::Identity();
  EXPECT_DOUBLE_EQ( identity.determinant(), 1.0 );
  EXPECT_DOUBLE_EQ( identity.trace(), 3.0 );
  auto const ia = identity * a;
  for( unsigned i = 0; i < 3; ++i ) { EXPECT_DOUBLE_EQ( ia[ i ], a[ i ] ); }

  // Column-major storage, which numpy and OpenCV both depend on
  matrix_2x3d m;
  m( 0, 0 ) = 1; m( 0, 1 ) = 2; m( 0, 2 ) = 3;
  m( 1, 0 ) = 4; m( 1, 1 ) = 5; m( 1, 2 ) = 6;
  EXPECT_DOUBLE_EQ( m.data()[ 0 ], 1.0 );
  EXPECT_DOUBLE_EQ( m.data()[ 1 ], 4.0 );
  EXPECT_DOUBLE_EQ( m.data()[ 2 ], 2.0 );

  auto const t = m.transpose();
  EXPECT_EQ( t.rows(), 3u );
  EXPECT_EQ( t.cols(), 2u );
  for( unsigned r = 0; r < 2; ++r )
  {
    for( unsigned c = 0; c < 3; ++c ) { EXPECT_DOUBLE_EQ( t( c, r ), m( r, c ) ); }
  }

  EXPECT_TRUE( quaternion_d().toRotationMatrix().isApprox(
                 matrix_3x3d::Identity() ) );
}
