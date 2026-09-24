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
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

/// A view over numpy memory. The caller keeps the array alive for the call.
viame::image_of< uint8_t >
as_image( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
{
  auto const buffer = array.request();

  if( buffer.ndim < 2 || buffer.ndim > 3 )
  {
    throw std::invalid_argument( "image must be HxW or HxWxC uint8" );
  }

  auto const depth =
    static_cast< size_t >( buffer.ndim == 3 ? buffer.shape[2] : 1 );

  return viame::image_of< uint8_t >(
    static_cast< uint8_t const* >( buffer.ptr ),
    static_cast< size_t >( buffer.shape[1] ),
    static_cast< size_t >( buffer.shape[0] ),
    depth,
    static_cast< ptrdiff_t >( depth ),
    static_cast< ptrdiff_t >( buffer.shape[1] ) * static_cast< ptrdiff_t >( depth ),
    1 );
}

/// Copied by index rather than memcpy'd: an image_of carries its own strides
/// and a kernel's result is not required to be packed.
py::array
as_array( viame::image_of< uint8_t > const& image, bool keep_third_axis )
{
  std::vector< Py_ssize_t > shape{
    static_cast< Py_ssize_t >( image.height() ),
    static_cast< Py_ssize_t >( image.width() ) };

  if( keep_third_axis )
  {
    shape.push_back( static_cast< Py_ssize_t >( image.depth() ) );
  }

  py::array_t< uint8_t > out( shape );
  uint8_t* destination = out.mutable_data();

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

py::array
resize( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
        size_t width, size_t height )
{
  auto const source = as_image( array );
  return as_array( viame::image_kernels::resize_bilinear( source, width, height ),
                   array.ndim() == 3 );
}

py::array
resize_area( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
             size_t width, size_t height )
{
  auto const source = as_image( array );
  return as_array( viame::image_kernels::resize_area( source, width, height ),
                   array.ndim() == 3 );
}

py::array
crop( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
      size_t left, size_t top, size_t width, size_t height )
{
  auto const source = as_image( array );
  return as_array( viame::image_kernels::crop( source, left, top, width, height ),
                   array.ndim() == 3 );
}

py::array
to_gray( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
{
  if( array.ndim() != 3 )
  {
    throw std::invalid_argument( "to_gray wants an HxWx3 image" );
  }
  auto const source = as_image( array );
  return as_array( viame::image_kernels::rgb_to_gray( source ), false );
}

py::array
to_rgb( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
{
  auto const source = as_image( array );
  return as_array( viame::image_kernels::gray_to_rgb( source ), true );
}

py::array
swap_channels( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
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
py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const&
three_channel(
  py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
  char const* who )
{
  if( array.ndim() != 3 || array.shape( 2 ) != 3 )
  {
    throw std::invalid_argument( std::string( who ) + " wants an HxWx3 image" );
  }
  return array;
}

py::array
to_hsv( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
{
  auto const source = as_image( three_channel( array, "to_hsv" ) );
  return as_array( viame::image_kernels::rgb_to_hsv( source ), true );
}

py::array
from_hsv( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
{
  auto const source = as_image( three_channel( array, "from_hsv" ) );
  return as_array( viame::image_kernels::hsv_to_rgb( source ), true );
}

py::array
to_hls( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
{
  auto const source = as_image( three_channel( array, "to_hls" ) );
  return as_array( viame::image_kernels::rgb_to_hls( source ), true );
}

py::array
from_hls( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
{
  auto const source = as_image( three_channel( array, "from_hls" ) );
  return as_array( viame::image_kernels::hls_to_rgb( source ), true );
}

py::array
to_lab( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
{
  auto const source = as_image( three_channel( array, "to_lab" ) );
  return as_array( viame::image_kernels::rgb_to_lab( source ), true );
}

py::array
from_lab( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
{
  auto const source = as_image( three_channel( array, "from_lab" ) );
  return as_array( viame::image_kernels::lab_to_rgb( source ), true );
}

py::array
demosaic( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
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
viame::image_of< uint8_t >
as_mutable_image( py::array_t< uint8_t, py::array::c_style >& array )
{
  auto buffer = array.request( true );

  if( buffer.ndim < 2 || buffer.ndim > 3 )
  {
    throw std::invalid_argument( "image must be HxW or HxWxC uint8" );
  }

  auto const depth =
    static_cast< size_t >( buffer.ndim == 3 ? buffer.shape[ 2 ] : 1 );

  return viame::image_of< uint8_t >(
    static_cast< uint8_t* >( buffer.ptr ),
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

void
fill_polygon( py::array_t< uint8_t, py::array::c_style >& array,
              py::array_t< double, py::array::c_style | py::array::forcecast >
                const& points,
              py::object const& colour )
{
  auto image = as_mutable_image( array );
  viame::image_kernels::fill_polygon( image, as_points( points ),
                                      as_colour( colour ) );
}

void
draw_rect( py::array_t< uint8_t, py::array::c_style >& array,
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

void
draw_text( py::array_t< uint8_t, py::array::c_style >& array,
           std::string const& text, long x, long y,
           py::object const& colour, long scale )
{
  auto image = as_mutable_image( array );
  viame::image_kernels::draw_text( image, text, x, y, as_colour( colour ),
                                   scale );
}

void
draw_line( py::array_t< uint8_t, py::array::c_style >& array,
           long x0, long y0, long x1, long y1, py::object const& colour )
{
  auto image = as_mutable_image( array );
  viame::image_kernels::draw_line( image, x0, y0, x1, y1,
                                   as_colour( colour ) );
}

void
draw_circle( py::array_t< uint8_t, py::array::c_style >& array,
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
gaussian_blur( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
               size_t size, double sigma, std::string const& border )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::gaussian_blur( source, size, sigma,
                                         as_border( border ) ),
    array.ndim() == 3 );
}

py::array
box_blur( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
          size_t size, std::string const& border )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::box_blur( source, size, as_border( border ) ),
    array.ndim() == 3 );
}

py::array
add_weighted( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& first,
              double alpha,
              py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& second,
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
normalize( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
           double low, double high )
{
  auto const source = as_image( array );
  return as_array( viame::image_kernels::normalize_min_max( source, low, high ),
                   array.ndim() == 3 );
}

py::array
equalize( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array )
{
  auto const source = as_image( array );
  return as_array( viame::image_kernels::equalize( source ),
                   array.ndim() == 3 );
}

py::array
clahe( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
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
erode( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
       std::string const& shape, int width, int height )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::grey_erode( source,
                                      as_element( shape, width, height ) ),
    array.ndim() == 3 );
}

py::array
dilate( py::array_t< uint8_t, py::array::c_style | py::array::forcecast > const& array,
        std::string const& shape, int width, int height )
{
  auto const source = as_image( array );
  return as_array(
    viame::image_kernels::grey_dilate( source,
                                       as_element( shape, width, height ) ),
    array.ndim() == 3 );
}

} // namespace

VIAME_PYTHON_MODULE( _image_kernels, m )
{
  m.doc() = "VIAME's own image kernels, so python needs no OpenCV";

  m.def( "resize", &resize, py::arg( "image" ), py::arg( "width" ),
         py::arg( "height" ),
         "Bilinear resize. The same kernel the C++ pipelines use, so a "
         "resized frame matches whether it was resized here or there." );

  m.def( "resize_area", &resize_area, py::arg( "image" ), py::arg( "width" ),
         py::arg( "height" ),
         "Resize by averaging each destination pixel's source footprint, "
         "which is the right filter for shrinking. Agrees with OpenCV's "
         "INTER_AREA to within one grey level." );

  m.def( "crop", &crop, py::arg( "image" ), py::arg( "left" ), py::arg( "top" ),
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

  m.def( "fill_polygon", &fill_polygon, py::arg( "image" ),
         py::arg( "points" ), py::arg( "colour" ),
         "Fill a polygon in place, outline included -- which is what "
         "cv2.fillPoly does, and is not what \"fill\" suggests. Points are "
         "an N by 2 array of x, y." );

  m.def( "draw_rect", &draw_rect, py::arg( "image" ), py::arg( "left" ),
         py::arg( "top" ), py::arg( "right" ), py::arg( "bottom" ),
         py::arg( "colour" ), py::arg( "thickness" ) = 1,
         "cv2.rectangle, with one difference that matters: `right` and "
         "`bottom` are **exclusive**, where cv2.rectangle's second corner "
         "is inclusive -- so a ported call passes x2 + 1 and y2 + 1. Given "
         "that, this is pixel for pixel what cv2.rectangle draws at "
         "thickness 1 and at -1, which fills. Thicker outlines differ: the "
         "kernel squares off each step where OpenCV mitres the corner." );

  m.def( "draw_text", &draw_text, py::arg( "image" ), py::arg( "text" ),
         py::arg( "x" ), py::arg( "y" ), py::arg( "colour" ),
         py::arg( "scale" ) = 1,
         "Draw text with its **top left** at (x, y), where cv2.putText "
         "places it by the baseline. The glyphs are a 5 by 7 bitmap font, "
         "so this does not look like OpenCV's text; it is legible, which is "
         "what a debug overlay needs." );

  m.def( "draw_line", &draw_line, py::arg( "image" ), py::arg( "x0" ),
         py::arg( "y0" ), py::arg( "x1" ), py::arg( "y1" ),
         py::arg( "colour" ),
         "cv2.line, one pixel wide. Bresenham, as cv2.LINE_8 is, but the "
         "tie breaking differs: about one pixel in nine of a long diagonal "
         "lands on the other side of the step." );

  m.def( "draw_circle", &draw_circle, py::arg( "image" ), py::arg( "x" ),
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
}
