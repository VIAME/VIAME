/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Python bindings for the camera geometry in `projection.h`
///
/// The five OpenCV calls VIAME's python reaches for most after `cvtColor`:
/// `projectPoints`, `undistortPoints`, `Rodrigues`, `stereoRectify` and
/// `initUndistortRectifyMap`. All five were already implemented in
/// `projection.h` for the C++ side; none could be reached from python, which
/// is why the calibration and rectification scripts still import cv2.
///
/// The bindings are **vectorised** where `projection.h` is not: it takes one
/// point at a time, because that is how the C++ callers use it, whereas every
/// python caller has an array. Looping in C++ here rather than in python
/// keeps the call sites looking like the `cv2` ones they replace.

#include <viame/utilities/python_fold.h>

#include <viame/measurement/projection.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using array_d =
  py::array_t< double, py::array::c_style | py::array::forcecast >;

// ---------------------------------------------------------------------------
// Converting numpy to the small fixed-size types

viame::matrix_3x3d
as_matrix_3x3( array_d const& array, char const* who )
{
  auto const buffer = array.request();

  if( buffer.ndim != 2 || buffer.shape[ 0 ] != 3 || buffer.shape[ 1 ] != 3 )
  {
    throw std::invalid_argument(
      std::string( who ) + " wants a three by three matrix" );
  }

  auto const* data = static_cast< double const* >( buffer.ptr );
  viame::matrix_3x3d out;

  for( unsigned row = 0; row < 3; ++row )
  {
    for( unsigned column = 0; column < 3; ++column )
    {
      out( row, column ) = data[ row * 3 + column ];
    }
  }

  return out;
}

viame::matrix_3x4d
as_matrix_3x4( array_d const& array, char const* who )
{
  auto const buffer = array.request();

  // A three by three is accepted and widened with a zero column, which is
  // what `cv2.undistortPoints` means by passing the intrinsic matrix as `P`.
  if( buffer.ndim != 2 || buffer.shape[ 0 ] != 3 ||
      ( buffer.shape[ 1 ] != 3 && buffer.shape[ 1 ] != 4 ) )
  {
    throw std::invalid_argument(
      std::string( who ) + " wants a three by three or three by four matrix" );
  }

  auto const columns = static_cast< unsigned >( buffer.shape[ 1 ] );
  auto const* data = static_cast< double const* >( buffer.ptr );
  viame::matrix_3x4d out;

  for( unsigned row = 0; row < 3; ++row )
  {
    for( unsigned column = 0; column < 4; ++column )
    {
      out( row, column ) = ( column < columns )
        ? data[ row * columns + column ]
        : 0.0;
    }
  }

  return out;
}

viame::vector_3d
as_vector_3( array_d const& array, char const* who )
{
  auto const buffer = array.request();

  if( buffer.size != 3 )
  {
    throw std::invalid_argument(
      std::string( who ) + " wants three numbers" );
  }

  auto const* data = static_cast< double const* >( buffer.ptr );
  return viame::vector_3d( data[ 0 ], data[ 1 ], data[ 2 ] );
}

/// The N by 2 or N by 3 a vectorised call is given, as a flat reader.
struct point_array
{
  double const* data;
  size_t count;
  size_t stride;
};

point_array
as_points( array_d const& array, size_t width, char const* who )
{
  auto const buffer = array.request();

  // (N, width), and (N, 1, width) because that is the shape `cv2` hands back
  // and the call sites carry it around.
  size_t last = 0;
  size_t count = 0;

  if( buffer.ndim == 2 )
  {
    last = static_cast< size_t >( buffer.shape[ 1 ] );
    count = static_cast< size_t >( buffer.shape[ 0 ] );
  }
  else if( buffer.ndim == 3 && buffer.shape[ 1 ] == 1 )
  {
    last = static_cast< size_t >( buffer.shape[ 2 ] );
    count = static_cast< size_t >( buffer.shape[ 0 ] );
  }
  else
  {
    throw std::invalid_argument(
      std::string( who ) + " wants an N by " + std::to_string( width ) +
      " array of points" );
  }

  if( last != width )
  {
    throw std::invalid_argument(
      std::string( who ) + " wants points of width " +
      std::to_string( width ) + ", got " + std::to_string( last ) );
  }

  return { static_cast< double const* >( buffer.ptr ), count, width };
}

py::array_t< double >
empty_points( size_t count, size_t width )
{
  return py::array_t< double >( std::vector< Py_ssize_t >{
    static_cast< Py_ssize_t >( count ),
    static_cast< Py_ssize_t >( width ) } );
}

py::array_t< double >
as_array_3x3( viame::matrix_3x3d const& matrix )
{
  py::array_t< double > out( std::vector< Py_ssize_t >{ 3, 3 } );
  auto* destination = out.mutable_data();

  for( unsigned row = 0; row < 3; ++row )
  {
    for( unsigned column = 0; column < 3; ++column )
    {
      *destination++ = matrix( row, column );
    }
  }

  return out;
}

template < unsigned Rows, unsigned Columns >
py::array_t< double >
as_array( viame::matrix_< Rows, Columns, double > const& matrix )
{
  py::array_t< double > out( std::vector< Py_ssize_t >{ Rows, Columns } );
  auto* destination = out.mutable_data();

  for( unsigned row = 0; row < Rows; ++row )
  {
    for( unsigned column = 0; column < Columns; ++column )
    {
      *destination++ = matrix( row, column );
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// The bindings

py::array_t< double >
project_points( array_d const& points, array_d const& intrinsics,
                viame::measurement::distortion_t const& coefficients,
                py::object const& rotation, py::object const& translation )
{
  auto const input = as_points( points, 3, "project_points" );
  auto const matrix = as_matrix_3x3( intrinsics, "project_points" );

  auto out = empty_points( input.count, 2 );
  auto* destination = out.mutable_data();

  bool const has_rotation = !rotation.is_none();
  bool const has_translation = !translation.is_none();

  viame::matrix_3x3d turn;
  viame::vector_3d shift;

  if( has_rotation )
  {
    auto const given = rotation.cast< array_d >();

    // An axis-angle vector or a matrix, as `cv2.projectPoints` takes either
    turn = ( given.size() == 3 )
      ? viame::measurement::rodrigues( as_vector_3( given, "project_points" ) )
      : as_matrix_3x3( given, "project_points" );
  }

  if( has_translation )
  {
    shift = as_vector_3( translation.cast< array_d >(), "project_points" );
  }

  for( size_t n = 0; n < input.count; ++n )
  {
    viame::vector_3d const point( input.data[ n * 3 ],
                                  input.data[ n * 3 + 1 ],
                                  input.data[ n * 3 + 2 ] );

    auto const mapped = has_translation
      ? viame::measurement::project_point( point, turn, shift, matrix,
                                           coefficients )
      : ( has_rotation
            ? viame::measurement::project_point( point, turn, matrix,
                                                 coefficients )
            : viame::measurement::project_point( point, matrix,
                                                 coefficients ) );

    *destination++ = mapped[ 0 ];
    *destination++ = mapped[ 1 ];
  }

  return out;
}

py::array_t< double >
undistort_points( array_d const& points, array_d const& intrinsics,
                  viame::measurement::distortion_t const& coefficients,
                  py::object const& rotation, py::object const& projection )
{
  auto const input = as_points( points, 2, "undistort_points" );
  auto const matrix = as_matrix_3x3( intrinsics, "undistort_points" );

  auto turn = viame::matrix_3x3d::Identity();

  if( !rotation.is_none() )
  {
    turn = as_matrix_3x3( rotation.cast< array_d >(), "undistort_points" );
  }

  viame::matrix_3x4d onto;

  if( projection.is_none() )
  {
    // OpenCV's null `P`: the result comes back in normalised coordinates
    onto.setZero();
    onto( 0, 0 ) = 1.0;
    onto( 1, 1 ) = 1.0;
    onto( 2, 2 ) = 1.0;
  }
  else
  {
    onto = as_matrix_3x4( projection.cast< array_d >(), "undistort_points" );
  }

  auto out = empty_points( input.count, 2 );
  auto* destination = out.mutable_data();

  for( size_t n = 0; n < input.count; ++n )
  {
    viame::vector_2d const point( input.data[ n * 2 ],
                                  input.data[ n * 2 + 1 ] );

    auto const mapped = viame::measurement::undistort_point(
      point, matrix, coefficients, turn, onto );

    *destination++ = mapped[ 0 ];
    *destination++ = mapped[ 1 ];
  }

  return out;
}

py::array_t< double >
rodrigues( array_d const& value )
{
  // Both directions, as `cv2.Rodrigues` is: three numbers in gives a matrix,
  // a matrix in gives three numbers.
  if( value.size() == 3 )
  {
    return as_array_3x3(
      viame::measurement::rodrigues( as_vector_3( value, "rodrigues" ) ) );
  }

  auto const vector = viame::measurement::inverse_rodrigues(
    as_matrix_3x3( value, "rodrigues" ) );

  py::array_t< double > out( std::vector< Py_ssize_t >{ 3 } );
  auto* destination = out.mutable_data();
  destination[ 0 ] = vector[ 0 ];
  destination[ 1 ] = vector[ 1 ];
  destination[ 2 ] = vector[ 2 ];

  return out;
}

py::dict
stereo_rectify( array_d const& left_intrinsics,
                viame::measurement::distortion_t const& left_distortion,
                array_d const& right_intrinsics,
                viame::measurement::distortion_t const& right_distortion,
                size_t width, size_t height,
                array_d const& rotation, array_d const& translation,
                double alpha )
{
  auto const result = viame::measurement::stereo_rectify(
    as_matrix_3x3( left_intrinsics, "stereo_rectify" ), left_distortion,
    as_matrix_3x3( right_intrinsics, "stereo_rectify" ), right_distortion,
    width, height,
    as_matrix_3x3( rotation, "stereo_rectify" ),
    as_vector_3( translation, "stereo_rectify" ), alpha );

  py::dict out;
  out[ "left_rotation" ] = as_array_3x3( result.left_rotation );
  out[ "right_rotation" ] = as_array_3x3( result.right_rotation );
  out[ "left_projection" ] = as_array< 3, 4 >( result.left_projection );
  out[ "right_projection" ] = as_array< 3, 4 >( result.right_projection );
  out[ "disparity_to_depth" ] = as_array< 4, 4 >( result.disparity_to_depth );

  return out;
}

py::tuple
rectification_maps( array_d const& intrinsics,
                    viame::measurement::distortion_t const& coefficients,
                    array_d const& rotation, array_d const& projection,
                    size_t width, size_t height )
{
  viame::image_of< float > map_x;
  viame::image_of< float > map_y;

  viame::measurement::rectification_maps(
    as_matrix_3x3( intrinsics, "rectification_maps" ), coefficients,
    as_matrix_3x3( rotation, "rectification_maps" ),
    as_matrix_3x4( projection, "rectification_maps" ),
    width, height, map_x, map_y );

  auto copy_out = [ & ]( viame::image_of< float > const& map )
  {
    py::array_t< float > out( std::vector< Py_ssize_t >{
      static_cast< Py_ssize_t >( map.height() ),
      static_cast< Py_ssize_t >( map.width() ) } );

    auto* destination = out.mutable_data();

    for( size_t y = 0; y < map.height(); ++y )
    {
      for( size_t x = 0; x < map.width(); ++x )
      {
        *destination++ = map( x, y, 0 );
      }
    }

    return out;
  };

  return py::make_tuple( copy_out( map_x ), copy_out( map_y ) );
}

} // namespace

VIAME_PYTHON_MODULE( _projection, m )
{
  m.doc() = "VIAME's camera geometry, so python needs no OpenCV for it";

  m.def( "project_points", &project_points, py::arg( "points" ),
         py::arg( "intrinsics" ),
         py::arg( "coefficients" ) = viame::measurement::distortion_t{},
         py::arg( "rotation" ) = py::none(),
         py::arg( "translation" ) = py::none(),
         "cv2.projectPoints. Takes an N by 3 of camera-frame points and "
         "gives an N by 2. `rotation` is an axis-angle triple or a three by "
         "three, as cv2's rvec may be either." );

  m.def( "undistort_points", &undistort_points, py::arg( "points" ),
         py::arg( "intrinsics" ),
         py::arg( "coefficients" ) = viame::measurement::distortion_t{},
         py::arg( "rotation" ) = py::none(),
         py::arg( "projection" ) = py::none(),
         "cv2.undistortPoints. With no coefficients this is exact; with any "
         "it is OpenCV's five-pass fixed point iteration, which is what "
         "cv2.undistortPoints does when given no term criteria. A `None` "
         "projection returns normalised coordinates, as cv2's null P does." );

  m.def( "rodrigues", &rodrigues, py::arg( "value" ),
         "cv2.Rodrigues, both directions: three numbers in gives a rotation "
         "matrix, a matrix in gives three numbers." );

  m.def( "stereo_rectify", &stereo_rectify, py::arg( "left_intrinsics" ),
         py::arg( "left_distortion" ), py::arg( "right_intrinsics" ),
         py::arg( "right_distortion" ), py::arg( "width" ),
         py::arg( "height" ), py::arg( "rotation" ), py::arg( "translation" ),
         py::arg( "alpha" ) = 0.0,
         "cv2.stereoRectify with CALIB_ZERO_DISPARITY, which is the only way "
         "VIAME asks for it. Returns a dict of the two rotations, the two "
         "projections and the disparity to depth matrix." );

  m.def( "rectification_maps", &rectification_maps, py::arg( "intrinsics" ),
         py::arg( "coefficients" ), py::arg( "rotation" ),
         py::arg( "projection" ), py::arg( "width" ), py::arg( "height" ),
         "cv2.initUndistortRectifyMap into a pair of float maps, ready for "
         "image_kernels.remap." );
}
