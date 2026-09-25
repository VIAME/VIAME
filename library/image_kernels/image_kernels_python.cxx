/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Python bindings for the in-house image kernels.
 *
 * These exist so that VIAME's python does not need OpenCV. The kernels were
 * written during the lite port to replace OpenCV's image operations in C++
 * and are already held to the golden recordings; binding them is what lets
 * the python half stop reaching for cv2, with the same numbers rather than
 * Pillow's, whose resampling differs by a half pixel.
 *
 * Arrays cross as HxW or HxWxC uint8, the convention `rectify_image` in
 * `library/measurement` already established.
 */

#include <viame/utilities/python_fold.h>

#include <viame/image_kernels/color.h>
#include <viame/image_kernels/contours.h>
#include <viame/image_kernels/draw.h>
#include <viame/image_kernels/filter.h>
#include <viame/image_kernels/histogram.h>
#include <viame/image_kernels/morphology.h>
#include <viame/image_kernels/resample.h>
#include <viame/image_kernels/warp.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

/// The array types the kernels take. `forcecast` is deliberately **not**
/// used: a float32 depth map handed to a uint8 binding would be silently
/// truncated, and a silently wrong depth map is worse than a TypeError.
/// Where a caller may reasonably have either, both are bound.
template < typename T >
using array_of = py::array_t< T, py::array::c_style >;

/// A view over numpy memory. The caller keeps the array alive for the call.
template < typename T >
viame::image_of< T >
as_image( array_of< T > const& array )
{
  auto const buffer = array.request();

  if( buffer.ndim < 2 || buffer.ndim > 3 )
  {
    throw std::invalid_argument( "image must be HxW or HxWxC uint8" );
  }

  auto const depth =
    static_cast< size_t >( buffer.ndim == 3 ? buffer.shape[2] : 1 );

  return viame::image_of< T >(
    static_cast< T const* >( buffer.ptr ),
    static_cast< size_t >( buffer.shape[1] ),
    static_cast< size_t >( buffer.shape[0] ),
    depth,
    static_cast< ptrdiff_t >( depth ),
    static_cast< ptrdiff_t >( buffer.shape[1] ) * static_cast< ptrdiff_t >( depth ),
    1 );
}

/// Copied by index rather than memcpy'd: an image_of carries its own strides
/// and a kernel's result is not required to be packed.
template < typename T >
py::array
as_array( viame::image_of< T > const& image, bool keep_third_axis )
{
  std::vector< Py_ssize_t > shape{
    static_cast< Py_ssize_t >( image.height() ),
    static_cast< Py_ssize_t >( image.width() ) };

  if( keep_third_axis )
  {
    shape.push_back( static_cast< Py_ssize_t >( image.depth() ) );
  }

  py::array_t< T > out( shape );
  T* destination = out.mutable_data();

  for( size_t y = 0; y < image.height(); ++y )
  {
    for( size_t x = 0; x < image.width(); ++x )
    {
      for( size_t d = 0; d < image.depth(); ++d )
      {
        *destination++ = image( x, y, d );
      }
    }
  }

  return std::move( out );
}

template < typename T >
py::array
resize( array_of< T > const& array, size_t width, size_t height )
{
  return as_array(
    viame::image_kernels::resize_bilinear( as_image( array ), width, height ),
    array.ndim() == 3 );
}

template < typename T >
py::array
resize_area( array_of< T > const& array, size_t width, size_t height )
{
  return as_array(
    viame::image_kernels::resize_area( as_image( array ), width, height ),
    array.ndim() == 3 );
}

template < typename T >
py::array
crop( array_of< T > const& array, size_t left, size_t top, size_t width,
      size_t height )
{
  return as_array(
    viame::image_kernels::crop( as_image( array ), left, top, width, height ),
    array.ndim() == 3 );
}

py::array
to_gray( array_of< uint8_t > const& array )
{
  if( array.ndim() != 3 )
  {
    throw std::invalid_argument( "to_gray wants an HxWx3 image" );
  }
  auto const source = as_image( array );
  return as_array( viame::image_kernels::rgb_to_gray( source ), false );
}

py::array
to_rgb( array_of< uint8_t > const& array )
{
  auto const source = as_image( array );
  return as_array( viame::image_kernels::gray_to_rgb( source ), true );
}

py::array
swap_channels( array_of< uint8_t > const& array )
{
  if( array.ndim() != 3 )
  {
    throw std::invalid_argument( "swap_channels wants an HxWx3 image" );
  }
  auto const source = as_image( array );
  return as_array( viame::image_kernels::swap_rb( source ), true );
}

/// The three-channel conversions all have the same shape: HxWx3 in, HxWx3
/// out. `three_channel` is the check they share, named so the error says
/// which conversion asked.
array_of< uint8_t > const&
three_channel(
  array_of< uint8_t > const& array,
  char const* who )
{
  if( array.ndim() != 3 || array.shape( 2 ) != 3 )
  {
    throw std::invalid_argument( std::string( who ) + " wants an HxWx3 image" );
  }
  return array;
}

py::array
to_hsv( array_of< uint8_t > const& array )
{
  auto const source = as_image( three_channel( array, "to_hsv" ) );
  return as_array( viame::image_kernels::rgb_to_hsv( source ), true );
}

py::array
from_hsv( array_of< uint8_t > const& array )
{
  auto const source = as_image( three_channel( array, "from_hsv" ) );
  return as_array( viame::image_kernels::hsv_to_rgb( source ), true );
}

py::array
to_hls( array_of< uint8_t > const& array )
{
  auto const source = as_image( three_channel( array, "to_hls" ) );
  return as_array( viame::image_kernels::rgb_to_hls( source ), true );
}

py::array
from_hls( array_of< uint8_t > const& array )
{
  auto const source = as_image( three_channel( array, "from_hls" ) );
  return as_array( viame::image_kernels::hls_to_rgb( source ), true );
}

py::array
to_lab( array_of< uint8_t > const& array )
{
  auto const source = as_image( three_channel( array, "to_lab" ) );
  return as_array( viame::image_kernels::rgb_to_lab( source ), true );
}

py::array
from_lab( array_of< uint8_t > const& array )
{
  auto const source = as_image( three_channel( array, "from_lab" ) );
  return as_array( viame::image_kernels::lab_to_rgb( source ), true );
}

py::array
demosaic( array_of< uint8_t > const& array,
          std::string const& pattern )
{
  using viame::image_kernels::bayer_pattern;

  if( !( array.ndim() == 2 || ( array.ndim() == 3 && array.shape( 2 ) == 1 ) ) )
  {
    throw std::invalid_argument( "demosaic wants a single plane mosaic" );
  }

  bayer_pattern which;
  if( pattern == "BG" )      { which = bayer_pattern::BG; }
  else if( pattern == "GB" ) { which = bayer_pattern::GB; }
  else if( pattern == "RG" ) { which = bayer_pattern::RG; }
  else if( pattern == "GR" ) { which = bayer_pattern::GR; }
  else
  {
    throw std::invalid_argument(
      "demosaic pattern must be one of BG, GB, RG, GR; got '" + pattern + "'" );
  }

  auto const source = as_image( array );
  return as_array( viame::image_kernels::demosaic( source, which ), true );
}

// ---------------------------------------------------------------------------
// Drawing
//
// These write into the caller's array rather than returning a new one, which
// is what `cv2.fillPoly`, `cv2.rectangle` and `cv2.putText` do and what the
// call sites expect: an overlay is built up by a run of them.

/// A writable view over numpy memory, for the drawing kernels.
template < typename T >
viame::image_of< T >
as_mutable_image( array_of< T >& array )
{
  auto buffer = array.request( true );

  if( buffer.ndim < 2 || buffer.ndim > 3 )
  {
    throw std::invalid_argument( "image must be HxW or HxWxC uint8" );
  }

  auto const depth =
    static_cast< size_t >( buffer.ndim == 3 ? buffer.shape[ 2 ] : 1 );

  return viame::image_of< T >(
    static_cast< T* >( buffer.ptr ),
    static_cast< size_t >( buffer.shape[ 1 ] ),
    static_cast< size_t >( buffer.shape[ 0 ] ),
    depth,
    static_cast< ptrdiff_t >( depth ),
    static_cast< ptrdiff_t >( buffer.shape[ 1 ] ) *
      static_cast< ptrdiff_t >( depth ),
    1 );
}

viame::image_kernels::colour
as_colour( py::object const& value )
{
  if( py::isinstance< py::float_ >( value ) ||
      py::isinstance< py::int_ >( value ) )
  {
    return { value.cast< double >() };
  }

  return value.cast< std::vector< double > >();
}

std::vector< viame::image_kernels::point >
as_points( py::array_t< double, py::array::c_style | py::array::forcecast >
             const& array )
{
  auto const buffer = array.request();

  if( buffer.ndim != 2 || buffer.shape[ 1 ] != 2 )
  {
    throw std::invalid_argument( "points must be an N by 2 array of x, y" );
  }

  auto const* data = static_cast< double const* >( buffer.ptr );
  std::vector< viame::image_kernels::point > out;
  out.reserve( static_cast< size_t >( buffer.shape[ 0 ] ) );

  for( Py_ssize_t n = 0; n < buffer.shape[ 0 ]; ++n )
  {
    viame::image_kernels::point p;
    p.i = static_cast< long >( std::lround( data[ n * 2 ] ) );
    p.j = static_cast< long >( std::lround( data[ n * 2 + 1 ] ) );
    out.push_back( p );
  }

  return out;
}

template < typename T >
void
fill_polygon( array_of< T >& array,
              py::array_t< double, py::array::c_style | py::array::forcecast >
                const& points,
              py::object const& colour )
{
  auto image = as_mutable_image( array );
  viame::image_kernels::fill_polygon( image, as_points( points ),
                                      as_colour( colour ) );
}

template < typename T >
void
draw_rect( array_of< T >& array,
           long left, long top, long right, long bottom,
           py::object const& colour, long thickness )
{
  auto image = as_mutable_image( array );

  viame::image_kernels::rect bounds;
  bounds.left = left;
  bounds.top = top;
  bounds.right = right;
  bounds.bottom = bottom;

  viame::image_kernels::draw_rect( image, bounds, as_colour( colour ),
                                   thickness );
}

template < typename T >
void
draw_text( array_of< T >& array,
           std::string const& text, long x, long y,
           py::object const& colour, long scale )
{
  auto image = as_mutable_image( array );
  viame::image_kernels::draw_text( image, text, x, y, as_colour( colour ),
                                   scale );
}

template < typename T >
void
draw_line( array_of< T >& array,
           long x0, long y0, long x1, long y1, py::object const& colour,
           long thickness )
{
  auto image = as_mutable_image( array );
  viame::image_kernels::draw_line( image, x0, y0, x1, y1,
                                   as_colour( colour ), thickness );
}

/// `cv2.polylines`: the segments of a chain, drawn one after another.
template < typename T >
void
draw_polyline( array_of< T >& array,
               py::array_t< double, py::array::c_style | py::array::forcecast >
                 const& points,
               py::object const& colour, bool closed, long thickness )
{
  auto const chain = as_points( points );

  if( chain.size() < 2 )
  {
    return;
  }

  auto image = as_mutable_image( array );
  auto const paint = as_colour( colour );

  for( size_t n = 0; n + 1 < chain.size(); ++n )
  {
    viame::image_kernels::draw_line( image, chain[ n ].i, chain[ n ].j,
                                     chain[ n + 1 ].i, chain[ n + 1 ].j,
                                     paint, thickness );
  }

  if( closed )
  {
    viame::image_kernels::draw_line( image, chain.back().i, chain.back().j,
                                     chain.front().i, chain.front().j,
                                     paint, thickness );
  }
}

template < typename T >
void
draw_circle( array_of< T >& array,
             long x, long y, long radius, py::object const& colour,
             long thickness )
{
  auto image = as_mutable_image( array );
  viame::image_kernels::draw_circle( image, x, y, radius,
                                     as_colour( colour ), thickness );
}

py::tuple
text_size( std::string const& text, long scale )
{
  auto const box = viame::image_kernels::text_size( text, scale );
  return py::make_tuple( box.width(), box.height() );
}

// ---------------------------------------------------------------------------
// Filtering, histograms and morphology

viame::image_kernels::border_mode
as_border( std::string const& name )
{
  using viame::image_kernels::border_mode;

  if( name == "constant" )    { return border_mode::CONSTANT; }
  if( name == "replicate" )   { return border_mode::REPLICATE; }
  if( name == "reflect" )     { return border_mode::REFLECT; }
  if( name == "reflect_101" ) { return border_mode::REFLECT_101; }

  throw std::invalid_argument(
    "border must be one of constant, replicate, reflect, reflect_101; got '" +
    name + "'" );
}

py::array
gaussian_blur( array_of< uint8_t > const& array,
               size_t size, double sigma, std::string const& border )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::gaussian_blur( source, size, sigma,
                                         as_border( border ) ),
    array.ndim() == 3 );
}

py::array
box_blur( array_of< uint8_t > const& array,
          size_t size, std::string const& border )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::box_blur( source, size, as_border( border ) ),
    array.ndim() == 3 );
}

py::array
add_weighted( array_of< uint8_t > const& first,
              double alpha,
              array_of< uint8_t > const& second,
              double beta, double gamma )
{
  auto const a = as_image( first );
  auto const b = as_image( second );

  if( a.width() != b.width() || a.height() != b.height() ||
      a.depth() != b.depth() )
  {
    throw std::invalid_argument( "add_weighted wants two images of one size" );
  }

  return as_array(
    viame::image_kernels::add_weighted( a, alpha, b, beta, gamma ),
    first.ndim() == 3 );
}

py::array
normalize( array_of< uint8_t > const& array,
           double low, double high )
{
  auto const source = as_image( array );
  return as_array( viame::image_kernels::normalize_min_max( source, low, high ),
                   array.ndim() == 3 );
}

py::array
equalize( array_of< uint8_t > const& array )
{
  auto const source = as_image( array );
  return as_array( viame::image_kernels::equalize( source ),
                   array.ndim() == 3 );
}

py::array
clahe( array_of< uint8_t > const& array,
       double clip_limit, size_t tiles_x, size_t tiles_y )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::clahe( source, clip_limit, tiles_x, tiles_y ),
    array.ndim() == 3 );
}

viame::image_kernels::structuring_element
as_element( std::string const& shape, int width, int height )
{
  if( shape == "rect" )
  {
    return viame::image_kernels::rect_element( width, height );
  }
  if( shape == "cross" )
  {
    return viame::image_kernels::cross_element( width, height );
  }
  if( shape == "disk" )
  {
    return viame::image_kernels::disk_element( width / 2.0 );
  }

  throw std::invalid_argument(
    "element must be one of rect, cross, disk; got '" + shape + "'" );
}

py::array
erode( array_of< uint8_t > const& array,
       std::string const& shape, int width, int height )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::grey_erode( source,
                                      as_element( shape, width, height ) ),
    array.ndim() == 3 );
}

py::array
dilate( array_of< uint8_t > const& array,
        std::string const& shape, int width, int height )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::grey_dilate( source,
                                       as_element( shape, width, height ) ),
    array.ndim() == 3 );
}

// ---------------------------------------------------------------------------
// Warping

viame::image_kernels::interpolation
as_interpolation( std::string const& name )
{
  using viame::image_kernels::interpolation;

  if( name == "nearest" )  { return interpolation::NEAREST; }
  if( name == "bilinear" ) { return interpolation::BILINEAR; }
  if( name == "bicubic" )  { return interpolation::BICUBIC; }
  if( name == "area" )     { return interpolation::AREA; }

  throw std::invalid_argument(
    "interpolation must be one of nearest, bilinear, bicubic, area; got '" +
    name + "'" );
}

/// A map of source positions, as `remap` takes them: one plane of float.
viame::image_of< float >
as_map( py::array_t< float, py::array::c_style | py::array::forcecast > const&
          array )
{
  auto const buffer = array.request();

  if( !( buffer.ndim == 2 ||
         ( buffer.ndim == 3 && buffer.shape[ 2 ] == 1 ) ) )
  {
    throw std::invalid_argument( "a remap map is one plane of float" );
  }

  return viame::image_of< float >(
    static_cast< float const* >( buffer.ptr ),
    static_cast< size_t >( buffer.shape[ 1 ] ),
    static_cast< size_t >( buffer.shape[ 0 ] ),
    1, 1,
    static_cast< ptrdiff_t >( buffer.shape[ 1 ] ), 1 );
}

template < typename T >
py::array
remap( array_of< T > const& array,
       py::array_t< float, py::array::c_style | py::array::forcecast > const& map_x,
       py::array_t< float, py::array::c_style | py::array::forcecast > const& map_y,
       std::string const& interpolation, std::string const& border,
       double constant )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::remap( source, as_map( map_x ), as_map( map_y ),
                                 as_interpolation( interpolation ),
                                 as_border( border ), constant ),
    array.ndim() == 3 );
}

viame::matrix_3x3d
as_matrix_3x3( py::array_t< double, py::array::c_style | py::array::forcecast >
                 const& array )
{
  auto const buffer = array.request();

  if( buffer.ndim != 2 || buffer.shape[ 0 ] != 3 || buffer.shape[ 1 ] != 3 )
  {
    throw std::invalid_argument( "wanted a three by three matrix" );
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

template < typename T >
py::array
warp_perspective( array_of< T > const& array,
                  py::array_t< double, py::array::c_style | py::array::forcecast > const& transform,
                  size_t width, size_t height,
                  std::string const& interpolation, std::string const& border,
                  double constant )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::warp_perspective(
      source, as_matrix_3x3( transform ), width, height,
      as_interpolation( interpolation ), as_border( border ), constant ),
    array.ndim() == 3 );
}

template < typename T >
py::array
warp_affine( array_of< T > const& array,
             py::array_t< double, py::array::c_style | py::array::forcecast > const& transform,
             size_t width, size_t height,
             std::string const& interpolation, std::string const& border,
             double constant )
{
  auto const buffer = transform.request();

  if( buffer.ndim != 2 || buffer.shape[ 0 ] != 2 || buffer.shape[ 1 ] != 3 )
  {
    throw std::invalid_argument( "warp_affine wants a two by three matrix" );
  }

  auto const* data = static_cast< double const* >( buffer.ptr );
  viame::matrix_< 2, 3, double > affine;

  for( unsigned row = 0; row < 2; ++row )
  {
    for( unsigned column = 0; column < 3; ++column )
    {
      affine( row, column ) = data[ row * 3 + column ];
    }
  }

  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::warp_affine(
      source, affine, width, height, as_interpolation( interpolation ),
      as_border( border ), constant ),
    array.ndim() == 3 );
}

/// Bind one name to the uint8, uint16 and float32 overloads.
///
/// VIAME ships 16-bit pipelines (`common_default_input_16bit.pipe`), so a
/// filter that draws on whatever frame it is handed sees uint16, and the
/// resamplers see float32 from a depth or disparity stage. Registering only
/// uint8 and relying on a cast is what `forcecast` did, and it is what this
/// avoids.
template < typename Byte, typename Short, typename Float, typename... Extra >
void
for_every_pixel_type( py::module& m, char const* name, Byte byte_version,
                      Short short_version, Float float_version,
                      Extra&&... extra )
{
  m.def( name, byte_version, extra... );
  m.def( name, short_version, extra... );
  m.def( name, float_version, std::forward< Extra >( extra )... );
}

/// Bind one name to both the uint8 and the float32 overload.
///
/// pybind tries them in the order they are registered, and without
/// `forcecast` on either an array of the other type falls through to the
/// second rather than being silently converted. uint8 goes first because
/// that is what a frame is; float32 is there for the maps a depth or
/// disparity stage carries, which `cv2.remap` took happily and a uint8-only
/// binding would have truncated without a word.
template < typename Byte, typename Float, typename... Extra >
void
for_both_pixel_types( py::module& m, char const* name, Byte byte_version,
                      Float float_version, Extra&&... extra )
{
  m.def( name, byte_version, extra... );
  m.def( name, float_version, std::forward< Extra >( extra )... );
}

// ---------------------------------------------------------------------------
// Contours and connected components

/// A contour as an N by 2 array of x, y -- the shape `cv2.findContours`
/// returns once the caller has reshaped away its middle axis, and the shape
/// `fill_polygon` above takes.
py::array_t< double >
contour_array( std::vector< viame::image_kernels::point > const& contour )
{
  py::array_t< double > out( std::vector< Py_ssize_t >{
    static_cast< Py_ssize_t >( contour.size() ), 2 } );

  auto* destination = out.mutable_data();

  for( auto const& p : contour )
  {
    *destination++ = static_cast< double >( p.i );
    *destination++ = static_cast< double >( p.j );
  }

  return out;
}

std::vector< viame::image_kernels::point >
contour_points( py::array_t< double, py::array::c_style | py::array::forcecast >
                  const& array, char const* who )
{
  auto const buffer = array.request();

  if( buffer.ndim != 2 || buffer.shape[ 1 ] != 2 )
  {
    throw std::invalid_argument(
      std::string( who ) + " wants an N by 2 array of x, y" );
  }

  auto const* data = static_cast< double const* >( buffer.ptr );
  std::vector< viame::image_kernels::point > out;
  out.reserve( static_cast< size_t >( buffer.shape[ 0 ] ) );

  for( Py_ssize_t n = 0; n < buffer.shape[ 0 ]; ++n )
  {
    viame::image_kernels::point p;
    p.i = static_cast< long >( std::lround( data[ n * 2 ] ) );
    p.j = static_cast< long >( std::lround( data[ n * 2 + 1 ] ) );
    out.push_back( p );
  }

  return out;
}

template < typename T >
py::list
find_contours( array_of< T > const& array )
{
  auto const traced =
    viame::image_kernels::find_contours( as_image( array ) );

  py::list out;

  for( auto const& contour : traced )
  {
    out.append( contour_array( contour ) );
  }

  return out;
}

double
contour_area( py::array_t< double, py::array::c_style | py::array::forcecast >
                const& contour )
{
  return viame::image_kernels::contour_area(
    contour_points( contour, "contour_area" ) );
}

py::tuple
bounding_rect( py::array_t< double, py::array::c_style | py::array::forcecast >
                 const& contour )
{
  auto const box = viame::image_kernels::bounding_rect(
    contour_points( contour, "bounding_rect" ) );

  // x, y, width, height, as `cv2.boundingRect` returns
  return py::make_tuple( box.left, box.top, box.width(), box.height() );
}

py::array_t< double >
convex_hull( py::array_t< double, py::array::c_style | py::array::forcecast >
               const& points )
{
  return contour_array( viame::image_kernels::convex_hull(
    contour_points( points, "convex_hull" ) ) );
}

py::dict
min_area_rect( py::array_t< double, py::array::c_style | py::array::forcecast >
                 const& points )
{
  auto const box = viame::image_kernels::min_area_rect(
    contour_points( points, "min_area_rect" ) );

  auto const corners = box.corners();

  py::array_t< double > corner_array( std::vector< Py_ssize_t >{ 4, 2 } );
  auto* destination = corner_array.mutable_data();

  for( auto const& corner : corners )
  {
    *destination++ = corner.first;
    *destination++ = corner.second;
  }

  py::dict out;
  out[ "centre" ] = py::make_tuple( box.centre_i, box.centre_j );
  out[ "size" ] = py::make_tuple( box.width, box.height );
  out[ "angle" ] = box.angle;
  out[ "corners" ] = corner_array;

  return out;
}

py::array_t< double >
approx_poly( py::array_t< double, py::array::c_style | py::array::forcecast >
               const& contour, double epsilon )
{
  return contour_array( viame::image_kernels::approx_poly(
    contour_points( contour, "approx_poly" ), epsilon ) );
}

template < typename T >
py::tuple
label_components( array_of< T > const& array, int connectivity )
{

  auto const how = ( connectivity == 4 )
    ? viame::image_kernels::connectivity::FOUR
    : viame::image_kernels::connectivity::EIGHT;

  size_t count = 0;
  auto const labels =
    viame::image_kernels::label_components( as_image( array ), how, count );

  py::array_t< int32_t > out( std::vector< Py_ssize_t >{
    static_cast< Py_ssize_t >( labels.height() ),
    static_cast< Py_ssize_t >( labels.width() ) } );

  auto* destination = out.mutable_data();

  for( size_t y = 0; y < labels.height(); ++y )
  {
    for( size_t x = 0; x < labels.width(); ++x )
    {
      *destination++ = labels( x, y, 0 );
    }
  }

  // count + 1 to match `cv2.connectedComponents`, which counts the background
  return py::make_tuple( static_cast< int >( count ) + 1, out );
}

template < typename T >
py::array
filter_2d( array_of< T > const& array,
           py::array_t< double, py::array::c_style | py::array::forcecast >
             const& weights,
           std::string const& border, double constant )
{
  auto const buffer = weights.request();

  if( buffer.ndim != 2 )
  {
    throw std::invalid_argument( "filter_2d wants a two dimensional kernel" );
  }

  viame::image_kernels::kernel k;
  k.width = static_cast< size_t >( buffer.shape[ 1 ] );
  k.height = static_cast< size_t >( buffer.shape[ 0 ] );

  auto const* data = static_cast< double const* >( buffer.ptr );
  k.weights.assign( data, data + k.width * k.height );

  return as_array(
    viame::image_kernels::filter_2d( as_image( array ), k,
                                     as_border( border ), constant ),
    array.ndim() == 3 );
}

} // namespace

VIAME_PYTHON_MODULE( _image_kernels, m )
{
  m.doc() = "VIAME's own image kernels, so python needs no OpenCV";

  for_every_pixel_type( m, "resize", &resize< uint8_t >,
         &resize< uint16_t >, &resize< float >,
         py::arg( "image" ), py::arg( "width" ),
         py::arg( "height" ),
         "Bilinear resize. The same kernel the C++ pipelines use, so a "
         "resized frame matches whether it was resized here or there." );

  for_every_pixel_type( m, "resize_area", &resize_area< uint8_t >,
         &resize_area< uint16_t >,
         &resize_area< float >, py::arg( "image" ), py::arg( "width" ),
         py::arg( "height" ),
         "Resize by averaging each destination pixel's source footprint, "
         "which is the right filter for shrinking. Agrees with OpenCV's "
         "INTER_AREA to within one grey level." );

  for_every_pixel_type( m, "crop", &crop< uint8_t >,
         &crop< uint16_t >, &crop< float >,
         py::arg( "image" ), py::arg( "left" ), py::arg( "top" ),
         py::arg( "width" ), py::arg( "height" ),
         "Crop to a rectangle, clamped to the image." );

  m.def( "to_gray", &to_gray, py::arg( "image" ),
         "RGB to single channel, by the same luma weights as the C++ side." );

  m.def( "to_rgb", &to_rgb, py::arg( "image" ),
         "Single channel to three identical ones." );

  m.def( "swap_channels", &swap_channels, py::arg( "image" ),
         "RGB to BGR, or back." );

  m.def( "to_hsv", &to_hsv, py::arg( "image" ),
         "RGB to HSV. Hue is 0..179 and saturation and value 0..255, which "
         "is OpenCV's 8-bit scaling, not a textbook's 0..360." );

  m.def( "from_hsv", &from_hsv, py::arg( "image" ),
         "HSV back to RGB, on the same 0..179 hue scale as to_hsv." );

  m.def( "to_hls", &to_hls, py::arg( "image" ),
         "RGB to HLS, hue on the same 0..179 scale as to_hsv." );

  m.def( "from_hls", &from_hls, py::arg( "image" ),
         "HLS back to RGB." );

  m.def( "to_lab", &to_lab, py::arg( "image" ),
         "RGB to CIE L*a*b*, 8-bit: L scaled to 0..255 and a and b offset "
         "by 128, again OpenCV's scaling." );

  m.def( "from_lab", &from_lab, py::arg( "image" ),
         "L*a*b* back to RGB, on the same 8-bit scaling as to_lab." );

  m.def( "demosaic", &demosaic, py::arg( "image" ), py::arg( "pattern" ),
         "Bayer mosaic to RGB. The pattern names the **mosaic** -- \"BG\" is "
         "blue at (0, 0) -- which is what a camera datasheet means and the "
         "reverse of how OpenCV spells its constants, so this is "
         "cv2.COLOR_BayerRG2RGB, not BayerBG2RGB." );

  for_both_pixel_types( m, "fill_polygon", &fill_polygon< uint8_t >,
         &fill_polygon< uint16_t >, py::arg( "image" ),
         py::arg( "points" ), py::arg( "colour" ),
         "Fill a polygon in place, outline included -- which is what "
         "cv2.fillPoly does, and is not what \"fill\" suggests. Points are "
         "an N by 2 array of x, y." );

  for_both_pixel_types( m, "draw_rect", &draw_rect< uint8_t >,
         &draw_rect< uint16_t >, py::arg( "image" ), py::arg( "left" ),
         py::arg( "top" ), py::arg( "right" ), py::arg( "bottom" ),
         py::arg( "colour" ), py::arg( "thickness" ) = 1,
         "cv2.rectangle, with one difference that matters: `right` and "
         "`bottom` are **exclusive**, where cv2.rectangle's second corner "
         "is inclusive -- so a ported call passes x2 + 1 and y2 + 1. Given "
         "that, this is pixel for pixel what cv2.rectangle draws at "
         "thickness 1 and at -1, which fills. Thicker outlines differ: the "
         "kernel squares off each step where OpenCV mitres the corner." );

  for_both_pixel_types( m, "draw_text", &draw_text< uint8_t >,
         &draw_text< uint16_t >, py::arg( "image" ), py::arg( "text" ),
         py::arg( "x" ), py::arg( "y" ), py::arg( "colour" ),
         py::arg( "scale" ) = 1,
         "Draw text with its **top left** at (x, y), where cv2.putText "
         "places it by the baseline. The glyphs are a 5 by 7 bitmap font, "
         "so this does not look like OpenCV's text; it is legible, which is "
         "what a debug overlay needs." );

  for_both_pixel_types( m, "draw_line", &draw_line< uint8_t >,
         &draw_line< uint16_t >, py::arg( "image" ), py::arg( "x0" ),
         py::arg( "y0" ), py::arg( "x1" ), py::arg( "y1" ),
         py::arg( "colour" ), py::arg( "thickness" ) = 1,
         "cv2.line. Bresenham, as cv2.LINE_8 is, but the tie breaking "
         "differs: about one pixel in nine of a long diagonal lands on the "
         "other side of the step. A thickness above one draws a filled "
         "square at each step, which is OpenCV's approximation for a thin "
         "line and not its mitred join." );

  for_both_pixel_types( m, "draw_polyline", &draw_polyline< uint8_t >,
         &draw_polyline< uint16_t >, py::arg( "image" ), py::arg( "points" ),
         py::arg( "colour" ), py::arg( "closed" ) = false,
         py::arg( "thickness" ) = 1,
         "cv2.polylines: the segments of an N by 2 chain of x, y." );

  for_both_pixel_types( m, "draw_circle", &draw_circle< uint8_t >,
         &draw_circle< uint16_t >, py::arg( "image" ), py::arg( "x" ),
         py::arg( "y" ), py::arg( "radius" ), py::arg( "colour" ),
         py::arg( "thickness" ) = 1,
         "cv2.circle, pixel for pixel at thickness 1. A thickness below "
         "zero fills." );

  m.def( "text_size", &text_size, py::arg( "text" ), py::arg( "scale" ) = 1,
         "The (width, height) of text, as cv2.getTextSize reports it." );

  m.def( "gaussian_blur", &gaussian_blur, py::arg( "image" ),
         py::arg( "size" ), py::arg( "sigma" ) = 0.0,
         py::arg( "border" ) = "reflect_101",
         "cv2.GaussianBlur. `size` is the odd kernel width and height, and "
         "sigma is derived from it when left at zero." );

  m.def( "box_blur", &box_blur, py::arg( "image" ), py::arg( "size" ),
         py::arg( "border" ) = "reflect_101", "cv2.blur." );

  m.def( "add_weighted", &add_weighted, py::arg( "first" ),
         py::arg( "alpha" ), py::arg( "second" ), py::arg( "beta" ),
         py::arg( "gamma" ) = 0.0,
         "first * alpha + second * beta + gamma, saturated. "
         "cv2.addWeighted." );

  m.def( "normalize", &normalize, py::arg( "image" ), py::arg( "low" ) = 0.0,
         py::arg( "high" ) = 255.0,
         "Rescale the image's range onto [low, high]. cv2.normalize with "
         "NORM_MINMAX." );

  m.def( "equalize", &equalize, py::arg( "image" ),
         "cv2.equalizeHist." );

  m.def( "clahe", &clahe, py::arg( "image" ), py::arg( "clip_limit" ) = 40.0,
         py::arg( "tiles_x" ) = 8, py::arg( "tiles_y" ) = 8,
         "Contrast limited adaptive histogram equalisation, which is what "
         "cv2.createCLAHE().apply() does." );

  m.def( "erode", &erode, py::arg( "image" ), py::arg( "shape" ) = "rect",
         py::arg( "width" ) = 3, py::arg( "height" ) = 3,
         "Grey erosion. cv2.erode with cv2.getStructuringElement; the shape "
         "is one of rect, cross, disk." );

  m.def( "dilate", &dilate, py::arg( "image" ), py::arg( "shape" ) = "rect",
         py::arg( "width" ) = 3, py::arg( "height" ) = 3,
         "Grey dilation. cv2.dilate." );

  for_every_pixel_type( m, "remap", &remap< uint8_t >,
         &remap< uint16_t >, &remap< float >,
         py::arg( "image" ), py::arg( "map_x" ),
         py::arg( "map_y" ), py::arg( "interpolation" ) = "bilinear",
         py::arg( "border" ) = "constant", py::arg( "constant" ) = 0.0,
         "cv2.remap. The maps give the source position of each output "
         "pixel, one plane of float each, and the output takes their size -- "
         "which is what a rectification does: the maps are made once and "
         "every frame is sampled through them." );

  for_every_pixel_type( m, "warp_perspective", &warp_perspective< uint8_t >,
         &warp_perspective< uint16_t >,
         &warp_perspective< float >, py::arg( "image" ),
         py::arg( "transform" ), py::arg( "width" ) = 0,
         py::arg( "height" ) = 0, py::arg( "interpolation" ) = "bilinear",
         py::arg( "border" ) = "constant", py::arg( "constant" ) = 0.0,
         "cv2.warpPerspective. The transform maps source to destination; "
         "a zero width or height keeps the source's." );

  for_every_pixel_type( m, "warp_affine", &warp_affine< uint8_t >,
         &warp_affine< uint16_t >,
         &warp_affine< float >, py::arg( "image" ),
         py::arg( "transform" ), py::arg( "width" ) = 0,
         py::arg( "height" ) = 0, py::arg( "interpolation" ) = "bilinear",
         py::arg( "border" ) = "constant", py::arg( "constant" ) = 0.0,
         "cv2.warpAffine, by a two by three matrix." );

  for_both_pixel_types( m, "find_contours", &find_contours< uint8_t >,
         &find_contours< uint16_t >, py::arg( "mask" ),
         "cv2.findContours with RETR_EXTERNAL, which is the only mode VIAME "
         "asks for. Suzuki and Abe's border following, as OpenCV's is: the "
         "traced point sets, areas and bounding boxes are identical. One "
         "difference -- each contour here is **closed**, repeating its first "
         "point at the end, where cv2 leaves it open, so a contour is one "
         "point longer than cv2's. Returns a list of N by 2 arrays of x, y, "
         "which is cv2's shape without its middle axis." );

  m.def( "contour_area", &contour_area, py::arg( "contour" ),
         "cv2.contourArea, unsigned." );

  m.def( "bounding_rect", &bounding_rect, py::arg( "contour" ),
         "cv2.boundingRect: (x, y, width, height)." );

  m.def( "convex_hull", &convex_hull, py::arg( "points" ),
         "cv2.convexHull. Andrew's monotone chain; collinear points are "
         "dropped, as OpenCV's are." );

  m.def( "min_area_rect", &min_area_rect, py::arg( "points" ),
         "cv2.minAreaRect, as a dict of centre, size, angle and the four "
         "`corners` -- which is cv2.boxPoints, so the two calls are one "
         "here. Adjacent entries are adjacent corners; the run may start at "
         "a different corner than OpenCV's or go the other way round, "
         "because OpenCV normalises its angle into [0, 90) and this keeps "
         "the edge it found." );

  m.def( "approx_poly", &approx_poly, py::arg( "contour" ),
         py::arg( "epsilon" ), "cv2.approxPolyDP, closed." );

  for_both_pixel_types( m, "label_components", &label_components< uint8_t >,
         &label_components< uint16_t >, py::arg( "mask" ),
         py::arg( "connectivity" ) = 8,
         "cv2.connectedComponents: (count, labels), where count includes "
         "the background as label 0." );

  for_every_pixel_type( m, "filter_2d", &filter_2d< uint8_t >,
         &filter_2d< uint16_t >, &filter_2d< float >, py::arg( "image" ),
         py::arg( "kernel" ), py::arg( "border" ) = "reflect_101",
         py::arg( "constant" ) = 0.0,
         "cv2.filter2D with a two dimensional kernel. Correlation, not "
         "convolution, as OpenCV's is." );
}
