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

#include <viame/image_kernels/background.h>
#include <viame/image_kernels/color.h>
#include <viame/image_kernels/edges.h>
#include <viame/image_kernels/hough.h>
#include <viame/image_kernels/smoother.h>
#include <viame/image_kernels/stereo.h>
#include <viame/image_kernels/contours.h>
#include <viame/image_kernels/corners.h>
#include <viame/image_kernels/denoise.h>
#include <viame/image_kernels/distance.h>
#include <viame/image_kernels/draw.h>
#include <viame/image_kernels/filter.h>
#include <viame/image_kernels/histogram.h>
#include <viame/image_kernels/match.h>
#include <viame/image_kernels/morphology.h>
#include <viame/image_kernels/optical_flow.h>
#include <viame/image_kernels/resample.h>
#include <viame/image_kernels/warp.h>
#include <viame/image_kernels/grabcut.h>
#include <viame/image_kernels/watershed.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

// Convert Python arguments before releasing the GIL, and reacquire it before
// constructing Python results. Forward references so output arguments survive.
template < typename Function, typename... Args >
decltype(auto)
call_kernel( Function&& function, Args&&... args )
{
  py::gil_scoped_release release;
  return std::forward< Function >( function )( std::forward< Args >( args )... );
}

#define VIAME_KERNEL_CALL( name, ... ) \
  call_kernel( []( auto&&... args ) -> decltype(auto) { \
    return viame::image_kernels::name( \
      std::forward< decltype(args) >( args )... ); }, __VA_ARGS__ )

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

/// Copy the kernel result into NumPy, respecting its plane and row strides.
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
  if( image.width() == 0 || image.height() == 0 || image.depth() == 0 )
  {
    return out;
  }

  {
    py::gil_scoped_release release;
    auto const width = image.width(), depth = image.depth();
    for( size_t y = 0; y < image.height(); ++y )
    {
      auto const* row = image.first_pixel() + y * image.h_step();
      if( image.w_step() == static_cast< ptrdiff_t >( depth ) &&
          ( depth == 1 || image.d_step() == 1 ) )
      {
        std::memcpy( destination, row, width * depth * sizeof( T ) );
        destination += width * depth;
      }
      else
      {
        // Copy each plane along contiguous output pixels, exposing strides
        // once per row rather than through image accessors per sample.
        for( size_t d = 0; d < depth; ++d )
        {
          auto const* src = row + d * image.d_step();
          for( size_t x = 0; x < width; ++x )
          { destination[x * depth + d] = src[x * image.w_step()]; }
        }
        destination += width * depth;
      }
    }
  }

  return std::move( out );
}

viame::image_kernels::interpolation
as_interpolation( std::string const& name );

template < typename T >
py::array
resize( array_of< T > const& array, size_t width, size_t height,
        std::string const& interpolation )
{
  return as_array(
    VIAME_KERNEL_CALL( resize, as_image( array ), width, height,
                                  as_interpolation( interpolation ) ),
    array.ndim() == 3 );
}

template <typename T>
py::array resize_letterbox(array_of<T> const& image, int width, int height)
{
  return as_array(VIAME_KERNEL_CALL(resize_letterbox, as_image(image), width, height),
                  image.ndim() == 3);
}

template < typename T >
py::array
resize_area( array_of< T > const& array, size_t width, size_t height )
{
  return as_array(
    VIAME_KERNEL_CALL( resize_area, as_image( array ), width, height ),
    array.ndim() == 3 );
}

template < typename T >
py::array
crop( array_of< T > const& array, size_t left, size_t top, size_t width,
      size_t height )
{
  return as_array(
    VIAME_KERNEL_CALL( crop, as_image( array ), left, top, width, height ),
    array.ndim() == 3 );
}

template < typename T >
py::array
to_gray( array_of< T > const& array )
{
  if( array.ndim() != 3 )
  {
    throw std::invalid_argument( "to_gray wants an HxWx3 image" );
  }
  auto const source = as_image( array );
  return as_array( VIAME_KERNEL_CALL( rgb_to_gray, source ), false );
}

template < typename T >
py::array
to_rgb( array_of< T > const& array )
{
  auto const source = as_image( array );
  return as_array( VIAME_KERNEL_CALL( gray_to_rgb, source ), true );
}

template < typename T >
py::array
swap_channels( array_of< T > const& array )
{
  if( array.ndim() != 3 )
  {
    throw std::invalid_argument( "swap_channels wants an HxWx3 image" );
  }
  auto const source = as_image( array );
  return as_array( VIAME_KERNEL_CALL( swap_rb, source ), true );
}

/// The three-channel conversions all have the same shape: HxWx3 in, HxWx3
/// out. `three_channel` is the check they share, named so the error says
/// which conversion asked.
template < typename T >
array_of< T > const&
three_channel(
  array_of< T > const& array,
  char const* who )
{
  if( array.ndim() != 3 || array.shape( 2 ) != 3 )
  {
    throw std::invalid_argument( std::string( who ) + " wants an HxWx3 image" );
  }
  return array;
}

// The hue and cylinder conversions come in two scalings, and which one a
// caller gets is decided by the **type it passes**, exactly as
// `cv::cvtColor` decides it. An 8-bit image has hue halved into 0..179 and
// saturation and value over 0..255, because a byte cannot hold degrees; a
// float image keeps hue in 0..360 and the other two in 0..1.
//
// Both were always in `color.h`; only the uint8 half was bound, which left
// every float caller -- the netharn augmenters among them -- on cv2, because
// handing their 0..360 hue to the 8-bit form silently halves it.
template < typename T >
py::array
to_hsv( array_of< T > const& array )
{
  auto const source = as_image( three_channel( array, "to_hsv" ) );
  return as_array( VIAME_KERNEL_CALL( rgb_to_hsv, source ), true );
}

template < typename T >
py::array
from_hsv( array_of< T > const& array )
{
  auto const source = as_image( three_channel( array, "from_hsv" ) );
  return as_array( VIAME_KERNEL_CALL( hsv_to_rgb, source ), true );
}

template < typename T >
py::array
to_hls( array_of< T > const& array )
{
  auto const source = as_image( three_channel( array, "to_hls" ) );
  return as_array( VIAME_KERNEL_CALL( rgb_to_hls, source ), true );
}

template < typename T >
py::array
from_hls( array_of< T > const& array )
{
  auto const source = as_image( three_channel( array, "from_hls" ) );
  return as_array( VIAME_KERNEL_CALL( hls_to_rgb, source ), true );
}

template < typename T >
py::array
to_lab( array_of< T > const& array, bool linear )
{
  auto const source = as_image( three_channel< T >( array, "to_lab" ) );
  return as_array( VIAME_KERNEL_CALL( rgb_to_lab, source, linear ), true );
}

template < typename T >
py::array
from_lab( array_of< T > const& array, bool linear )
{
  auto const source = as_image( three_channel< T >( array, "from_lab" ) );
  return as_array( VIAME_KERNEL_CALL( lab_to_rgb, source, linear ), true );
}

py::array
denoise( array_of< uint8_t > const& array,
         double strength, int patch, int window )
{
  auto const source = as_image( array );
  return as_array(
    VIAME_KERNEL_CALL( denoise_non_local_means, source, strength, patch,
                       window ),
    array.ndim() == 3 );
}

py::array
denoise_colour( array_of< uint8_t > const& array,
                double strength, double colour_strength,
                int patch, int window )
{
  auto const source =
    as_image( three_channel< uint8_t >( array, "denoise_colour" ) );
  return as_array(
    VIAME_KERNEL_CALL( denoise_non_local_means_colour, source, strength,
                       colour_strength, patch, window ),
    true );
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
  return as_array( VIAME_KERNEL_CALL( demosaic, source, which ), true );
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
  VIAME_KERNEL_CALL( fill_polygon, image, as_points( points ),
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

  VIAME_KERNEL_CALL( draw_rect, image, bounds, as_colour( colour ),
                                   thickness );
}

template < typename T >
void
draw_text( array_of< T >& array,
           std::string const& text, long x, long y,
           py::object const& colour, long scale )
{
  auto image = as_mutable_image( array );
  VIAME_KERNEL_CALL( draw_text, image, text, x, y, as_colour( colour ),
                                   scale );
}

template < typename T >
void
draw_line( array_of< T >& array,
           long x0, long y0, long x1, long y1, py::object const& colour,
           long thickness )
{
  auto image = as_mutable_image( array );
  VIAME_KERNEL_CALL( draw_line, image, x0, y0, x1, y1,
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
    VIAME_KERNEL_CALL( draw_line, image, chain[ n ].i, chain[ n ].j,
                                     chain[ n + 1 ].i, chain[ n + 1 ].j,
                                     paint, thickness );
  }

  if( closed )
  {
    VIAME_KERNEL_CALL( draw_line, image, chain.back().i, chain.back().j,
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
  VIAME_KERNEL_CALL( draw_circle, image, x, y, radius,
                                     as_colour( colour ), thickness );
}

template < typename T >
void
fill_ellipse( array_of< T >& array,
              long x, long y, long radius_x, long radius_y,
              py::object const& colour, double angle )
{
  auto image = as_mutable_image( array );
  VIAME_KERNEL_CALL( fill_ellipse, image, x, y, radius_x, radius_y,
                                      as_colour( colour ), angle );
}

py::tuple
text_size( std::string const& text, long scale )
{
  auto const box = VIAME_KERNEL_CALL( text_size, text, scale );
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
  if( name == "wrap" )        { return border_mode::WRAP; }

  throw std::invalid_argument(
    "border must be one of constant, replicate, reflect, reflect_101, wrap; "
    "got '" + name + "'" );
}

struct gaussian_buffer
{
  std::mutex mutex;
  viame::image_kernels::gaussian_workspace scratch;
};
struct stereo_buffer
{
  std::mutex mutex;
  viame::image_kernels::stereo_workspace scratch;
};

template <typename T>
py::array gaussian_blur ( array_of<T> const &array, size_t size, double sigma,
                          std::string const &border, gaussian_buffer *workspace )
{
  auto const source = as_image( array );
  auto const mode = as_border ( border );
  auto const result = call_kernel (
      [&]
      {
        std::unique_lock<std::mutex> lock;
        if ( workspace )
        {
          lock = std::unique_lock<std::mutex> ( workspace->mutex );
        }
        return viame::image_kernels::gaussian_blur (
            source, size, sigma, mode, workspace ? &workspace->scratch : nullptr );
      } );
  return as_array ( result, array.ndim () == 3 );
}

template < typename T >
py::array
gaussian_blur_float_taps( array_of< T > const& array, size_t size,
                          double sigma, std::string const& border )
{
  auto const source = as_image( array );
  return as_array(
    VIAME_KERNEL_CALL( gaussian_blur_float_taps, source, size, sigma,
                       as_border( border ) ),
    array.ndim() == 3 );
}

template < typename T >
py::array
box_blur( array_of< T > const& array,
          size_t size, std::string const& border )
{
  auto const source = as_image( array );
  return as_array(
    VIAME_KERNEL_CALL( box_blur, source, size, as_border( border ) ),
    array.ndim() == 3 );
}

template < typename T >
py::array
add_weighted( array_of< T > const& first,
              double alpha,
              array_of< T > const& second,
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
    VIAME_KERNEL_CALL( add_weighted, a, alpha, b, beta, gamma ),
    first.ndim() == 3 );
}

template < typename T >
py::array
normalize( array_of< T > const& array,
           double low, double high )
{
  auto const source = as_image( array );
  return as_array( VIAME_KERNEL_CALL( normalize_min_max, source, low, high ),
                   array.ndim() == 3 );
}

py::array
equalize( array_of< uint8_t > const& array )
{
  auto const source = as_image( array );
  return as_array( VIAME_KERNEL_CALL( equalize, source ),
                   array.ndim() == 3 );
}

py::array_t< float >
smooth_globally( array_of< uint8_t > const& guide, array_of< float > const& image,
                 double lambda, double sigma, double attenuation,
                 int iterations )
{
  auto const one = as_image( guide );
  auto const two = as_image( image );

  return as_array(
    VIAME_KERNEL_CALL( smooth_globally, one, two, lambda, sigma, attenuation,
                       iterations ),
    image.ndim() == 3 ).cast< py::array_t< float > >();
}

py::array_t< float >
filter_disparity_wls( array_of< uint8_t > const& guide,
                      array_of< int16_t > const& left,
                      array_of< int16_t > const& right, double lambda,
                      double sigma, int left_offset, int right_offset,
                      int min_disparity, int discontinuity_radius,
                      int lrc_threshold, double roll_off )
{
  auto const one = as_image( guide );
  auto const two = as_image( left );
  auto const three = as_image( right );

  viame::image_kernels::wls_params params;
  params.lambda = lambda;
  params.sigma = sigma;
  params.left_offset = left_offset;
  params.right_offset = right_offset;
  params.min_disparity = min_disparity;
  params.discontinuity_radius = discontinuity_radius;
  params.lrc_threshold = lrc_threshold;
  params.roll_off = static_cast< float >( roll_off );

  return as_array(
    VIAME_KERNEL_CALL( filter_disparity_wls, one, two, three, params ),
    false ).cast< py::array_t< float > >();
}

py::array_t< int16_t >
stereo_bm( array_of< uint8_t > const& left, array_of< uint8_t > const& right,
           int num_disparities, int block_size, int min_disparity,
           std::string const& pre_filter, int pre_filter_size,
           int pre_filter_cap, int texture_threshold, int uniqueness_ratio,
           int speckle_window_size, int speckle_range, int disp12_max_diff )
{
  auto const one = as_image( left );
  auto const two = as_image( right );

  viame::image_kernels::bm_params params;
  params.num_disparities = num_disparities;
  params.block_size = block_size;
  params.min_disparity = min_disparity;
  params.pre_filter_size = pre_filter_size;
  params.pre_filter_cap = pre_filter_cap;
  params.texture_threshold = texture_threshold;
  params.uniqueness_ratio = uniqueness_ratio;
  params.speckle_window_size = speckle_window_size;
  params.speckle_range = speckle_range;
  params.disp12_max_diff = disp12_max_diff;

  if( pre_filter == "xsobel" )
  {
    params.pre_filter = viame::image_kernels::bm_prefilter::XSOBEL;
  }
  else if( pre_filter == "normalized_response" )
  {
    params.pre_filter =
      viame::image_kernels::bm_prefilter::NORMALIZED_RESPONSE;
  }
  else
  {
    throw std::invalid_argument(
      "stereo_bm: pre_filter must be 'xsobel' or 'normalized_response'; got '" +
      pre_filter + "'" );
  }

  return as_array(
    VIAME_KERNEL_CALL( stereo_bm, one, two, params ),
    false ).cast< py::array_t< int16_t > >();
}

py::array_t<int16_t> stereo_sgbm ( array_of<uint8_t> const &left,
                                   array_of<uint8_t> const &right, int min_disparity,
                                   int num_disparities, int block_size, int p1, int p2,
                                   int disp12_max_diff, int pre_filter_cap,
                                   int uniqueness_ratio, int speckle_window_size,
                                   int speckle_range, std::string const &mode,
                                   stereo_buffer *workspace )
{
  auto const one = as_image( left );
  auto const two = as_image( right );

  viame::image_kernels::sgbm_params params;
  params.min_disparity = min_disparity;
  params.num_disparities = num_disparities;
  params.block_size = block_size;
  params.p1 = p1;
  params.p2 = p2;
  params.disp12_max_diff = disp12_max_diff;
  params.pre_filter_cap = pre_filter_cap;
  params.uniqueness_ratio = uniqueness_ratio;
  params.speckle_window_size = speckle_window_size;
  params.speckle_range = speckle_range;
  if( mode == "sgbm" )
  {
    params.mode = viame::image_kernels::sgbm_mode::SGBM;
  }
  else if( mode == "hh" )
  {
    params.mode = viame::image_kernels::sgbm_mode::HH;
  }
  else if( mode == "sgbm_3way" )
  {
    params.mode = viame::image_kernels::sgbm_mode::SGBM_3WAY;
  }
  else
  {
    throw std::invalid_argument(
      "stereo_sgbm: mode must be one of sgbm, hh, sgbm_3way; got '" + mode +
      "'" );
  }

  auto const found = call_kernel (
      [&]
      {
        std::unique_lock<std::mutex> lock;
        if ( workspace )
        {
          lock = std::unique_lock<std::mutex> ( workspace->mutex );
        }
        return viame::image_kernels::stereo_sgbm (
            one, two, params, workspace ? &workspace->scratch : nullptr );
      } );

  py::array_t< int16_t > out( std::vector< Py_ssize_t >{
    static_cast< Py_ssize_t >( found.height() ),
    static_cast< Py_ssize_t >( found.width() ) } );

  auto* destination = out.mutable_data();

  for( size_t j = 0; j < found.height(); ++j )
  {
    for( size_t i = 0; i < found.width(); ++i )
    {
      *destination++ = found( i, j, 0 );
    }
  }

  return out;
}

py::array_t< float >
hough_circles( array_of< uint8_t > const& array, double dp, double min_dist,
               double canny_threshold, double acc_threshold, int min_radius,
               int max_radius, int max_circles )
{
  auto const source = as_image( array );
  auto const found = VIAME_KERNEL_CALL( hough_circles, source, dp, min_dist,
                                        canny_threshold, acc_threshold,
                                        min_radius, max_radius, max_circles );

  py::array_t< float > out( std::vector< Py_ssize_t >{
    static_cast< Py_ssize_t >( found.size() ), 3 } );

  auto* destination = out.mutable_data();

  for( auto const& one : found )
  {
    *destination++ = one.x;
    *destination++ = one.y;
    *destination++ = one.radius;
  }

  return out;
}

py::array
canny( array_of< uint8_t > const& array, double low, double high,
       size_t aperture, bool l2_gradient )
{
  auto const source = as_image( array );
  return as_array(
    VIAME_KERNEL_CALL( canny, source, low, high, aperture, l2_gradient ),
    array.ndim() == 3 );
}

template < typename T >
py::array
median_blur( array_of< T > const& array, size_t size )
{
  auto const source = as_image( array );
  return as_array( VIAME_KERNEL_CALL( median_blur, source, size ),
                   array.ndim() == 3 );
}

template < typename T >
py::array
clahe( array_of< T > const& array,
       double clip_limit, size_t tiles_x, size_t tiles_y )
{
  auto const source = as_image( array );
  return as_array(
    VIAME_KERNEL_CALL( clahe, source, clip_limit, tiles_x, tiles_y ),
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
  if( shape == "ellipse" )
  {
    return viame::image_kernels::ellipse_element( width, height );
  }

  throw std::invalid_argument(
    "element must be one of rect, cross, disk, ellipse; got '" + shape +
    "'" );
}

template < typename T >
py::array
erode( array_of< T > const& array,
       std::string const& shape, int width, int height )
{
  auto const source = as_image( array );
  return as_array(
    VIAME_KERNEL_CALL( grey_erode, source,
                                      as_element( shape, width, height ) ),
    array.ndim() == 3 );
}

template < typename T >
py::array
dilate( array_of< T > const& array,
        std::string const& shape, int width, int height )
{
  auto const source = as_image( array );
  return as_array(
    VIAME_KERNEL_CALL( grey_dilate, source,
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
  if( name == "bilinear_exact" ) { return interpolation::BILINEAR_EXACT; }

  throw std::invalid_argument(
    "interpolation must be one of nearest, bilinear, bilinear_exact, "
    "bicubic, area; got '" +
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
    VIAME_KERNEL_CALL( remap, source, as_map( map_x ), as_map( map_y ),
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
    VIAME_KERNEL_CALL( warp_perspective,
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
    VIAME_KERNEL_CALL( warp_affine,
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
/// Use only for operations whose supported dtypes exclude uint16. NumPy
/// safe casts still allow uint16 to reach a float overload without forcecast;
/// operations preserving uint16 must use for_every_pixel_type instead.
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
    VIAME_KERNEL_CALL( find_contours, as_image( array ) );

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
  return VIAME_KERNEL_CALL( contour_area,
    contour_points( contour, "contour_area" ) );
}

py::tuple
bounding_rect( py::array_t< double, py::array::c_style | py::array::forcecast >
                 const& contour )
{
  auto const box = VIAME_KERNEL_CALL( bounding_rect,
    contour_points( contour, "bounding_rect" ) );

  // x, y, width, height, as `cv2.boundingRect` returns
  return py::make_tuple( box.left, box.top, box.width(), box.height() );
}

/// Every border of every component, outer and hole, which is
/// `cv2.findContours` under `RETR_LIST` or `RETR_CCOMP`.
///
/// `find_contours` above takes the `RETR_EXTERNAL` shortcut and is what most
/// callers want. This is for the ones that need the holes too -- the blob
/// detector above all, which finds a dark shape as the **hole** in the
/// lighter region around it and would see nothing at all without them.
template < typename T >
py::array
distance_transform( array_of< T > const& array )
{
  return as_array(
    VIAME_KERNEL_CALL( distance_transform, as_image( array ) ), false );
}

/// `cv::BackgroundSubtractorMOG2` in python: stateful, so a class rather than
/// a function. The mask comes back as an H by W uint8, 255 for foreground.
class mog2_wrapper
{
public:
  mog2_wrapper( int history, double var_threshold, int mixtures,
                double background_ratio, double var_threshold_gen,
                double var_init, double var_min, double var_max,
                double complexity_reduction )
  {
    viame::image_kernels::mog2_params params;
    params.history = history;
    params.var_threshold = var_threshold;
    params.mixtures = mixtures;
    params.background_ratio = background_ratio;
    params.var_threshold_gen = var_threshold_gen;
    params.var_init = var_init;
    params.var_min = var_min;
    params.var_max = var_max;
    params.complexity_reduction = complexity_reduction;

    m_model.reset( new viame::image_kernels::mog2_background( params ) );
  }

  py::array
  apply( array_of< uint8_t > const& array, double learning_rate )
  {
    auto const source = as_image( array );
    viame::image_of< uint8_t > result;
    {
      py::gil_scoped_release release;
      std::lock_guard< std::mutex > lock( m_mutex );
      result = m_model->apply( source, learning_rate );
    }
    return as_array( result, false );
  }

  void clear()
  {
    py::gil_scoped_release release;
    std::lock_guard< std::mutex > lock( m_mutex );
    m_model->reset();
  }

  size_t frames() const
  {
    py::gil_scoped_release release;
    std::lock_guard< std::mutex > lock( m_mutex );
    return m_model->frames();
  }

private:
  mutable std::mutex m_mutex;
  std::shared_ptr< viame::image_kernels::mog2_background > m_model;
};

template < typename T >
py::array_t< float >
good_features( array_of< T > const& array, int max_corners,
               double quality_level, double min_distance, int block_size,
               int aperture )
{
  auto const found = VIAME_KERNEL_CALL( good_features_to_track,
    as_image( array ), max_corners, quality_level, min_distance, block_size,
    aperture );

  py::array_t< float > out( std::vector< Py_ssize_t >{
    static_cast< Py_ssize_t >( found.size() ), 2 } );

  auto* destination = out.mutable_data();

  for( auto const& corner : found )
  {
    *destination++ = corner.first;
    *destination++ = corner.second;
  }

  return out;
}

py::array_t< float >
fast_corners( array_of< uint8_t > const& array, int threshold, bool suppress )
{
  auto const found = VIAME_KERNEL_CALL( fast_corners,
    as_image( array ), threshold, suppress );

  py::array_t< float > out( std::vector< Py_ssize_t >{
    static_cast< Py_ssize_t >( found.size() ), 3 } );

  auto* destination = out.mutable_data();

  for( auto const& corner : found )
  {
    *destination++ = corner.x;
    *destination++ = corner.y;
    *destination++ = corner.response;
  }

  return out;
}

template < typename T >
py::array
min_eigen_value( array_of< T > const& array, int block_size, int aperture )
{
  return as_array( VIAME_KERNEL_CALL( min_eigen_value,
    as_image( array ), block_size, aperture ), false );
}

py::tuple
lucas_kanade( array_of< uint8_t > const& first,
              array_of< uint8_t > const& second,
              py::array_t< float, py::array::c_style > const& points,
              int win_width, int win_height, int levels, int iterations,
              double epsilon, double min_eigen, int threads )
{
  auto const buffer = points.request();

  if( !( buffer.ndim == 2 && buffer.shape[ 1 ] == 2 ) )
  {
    throw std::invalid_argument(
      "lucas_kanade wants an N by 2 array of x, y" );
  }

  auto const count = static_cast< size_t >( buffer.shape[ 0 ] );
  auto const* source = static_cast< float const* >( buffer.ptr );

  std::vector< std::pair< float, float > > wanted;
  wanted.reserve( count );

  for( size_t i = 0; i < count; ++i )
  {
    wanted.emplace_back( source[ i * 2 ], source[ i * 2 + 1 ] );
  }

  viame::image_kernels::lucas_kanade_params params;
  params.win_width = win_width;
  params.win_height = win_height;
  params.levels = levels;
  params.iterations = iterations;
  params.epsilon = epsilon;
  params.min_eigen = min_eigen;
  params.threads = threads;

  std::vector< uint8_t > status;

  auto const moved = VIAME_KERNEL_CALL( lucas_kanade_flow,
    as_image( first ), as_image( second ), wanted, status, params );

  py::array_t< float > places( std::vector< Py_ssize_t >{
    static_cast< Py_ssize_t >( moved.size() ), 2 } );
  py::array_t< uint8_t > kept( std::vector< Py_ssize_t >{
    static_cast< Py_ssize_t >( status.size() ) } );

  auto* place = places.mutable_data();
  auto* keep = kept.mutable_data();

  for( size_t i = 0; i < moved.size(); ++i )
  {
    *place++ = moved[ i ].first;
    *place++ = moved[ i ].second;
    keep[ i ] = status[ i ];
  }

  return py::make_tuple( places, kept );
}

template < typename T >
py::array
optical_flow( array_of< T > const& first, array_of< T > const& second,
              double pyr_scale, int levels, int winsize, int iterations,
              int poly_n, double poly_sigma )
{
  viame::image_kernels::farneback_params params;
  params.pyr_scale = pyr_scale;
  params.levels = levels;
  params.winsize = winsize;
  params.iterations = iterations;
  params.poly_n = poly_n;
  params.poly_sigma = poly_sigma;

  return as_array(
    VIAME_KERNEL_CALL( farneback_optical_flow,
      as_image( first ), as_image( second ), params ), true );
}

template < typename T >
py::list
find_borders( array_of< T > const& array )
{
  auto const traced =
    VIAME_KERNEL_CALL( find_borders, as_image( array ) );

  py::list out;

  for( auto const& border : traced )
  {
    out.append( py::make_tuple( contour_array( border.points ),
                                border.is_hole ) );
  }

  return out;
}

double
arc_length( py::array_t< double, py::array::c_style | py::array::forcecast >
              const& contour, bool closed )
{
  return VIAME_KERNEL_CALL( arc_length,
    contour_points( contour, "arc_length" ), closed );
}

py::dict
moments( py::array_t< double, py::array::c_style | py::array::forcecast >
           const& contour )
{
  auto const found = VIAME_KERNEL_CALL( moments,
    contour_points( contour, "moments" ) );

  py::dict out;
  out[ "m00" ] = found.m00;
  out[ "m10" ] = found.m10;
  out[ "m01" ] = found.m01;
  out[ "m20" ] = found.m20;
  out[ "m11" ] = found.m11;
  out[ "m02" ] = found.m02;

  return out;
}

/// The sub-pixel points `intersect_convex` deals in, which `contour_points`
/// cannot give: it rounds to the pixel grid, and the vertices of an overlap
/// are where two edges cross.
std::vector< std::pair< double, double > >
polygon_points( py::array_t< double, py::array::c_style | py::array::forcecast >
                  const& array, char const* who )
{
  auto const buffer = array.request();

  if( buffer.ndim != 2 || buffer.shape[ 1 ] != 2 )
  {
    throw std::invalid_argument(
      std::string( who ) + " wants an N by 2 array of x, y" );
  }

  auto const* data = static_cast< double const* >( buffer.ptr );
  std::vector< std::pair< double, double > > out;
  out.reserve( static_cast< size_t >( buffer.shape[ 0 ] ) );

  for( Py_ssize_t n = 0; n < buffer.shape[ 0 ]; ++n )
  {
    out.emplace_back( data[ n * 2 ], data[ n * 2 + 1 ] );
  }

  return out;
}

py::tuple
intersect_convex(
  py::array_t< double, py::array::c_style | py::array::forcecast > const& first,
  py::array_t< double, py::array::c_style | py::array::forcecast > const& second )
{
  auto const overlap = VIAME_KERNEL_CALL( intersect_convex,
    polygon_points( first, "intersect_convex" ),
    polygon_points( second, "intersect_convex" ) );

  py::array_t< double > points(
    std::vector< Py_ssize_t >{ static_cast< Py_ssize_t >( overlap.size() ),
                               2 } );
  auto* destination = points.mutable_data();

  for( auto const& corner : overlap )
  {
    *destination++ = corner.first;
    *destination++ = corner.second;
  }

  // The area first, as `cv2.intersectConvexConvex` returns it, and the
  // polygon second.
  return py::make_tuple( VIAME_KERNEL_CALL( polygon_area, overlap ),
                         points );
}

py::array_t< double >
convex_hull( py::array_t< double, py::array::c_style | py::array::forcecast >
               const& points )
{
  return contour_array( VIAME_KERNEL_CALL( convex_hull,
    contour_points( points, "convex_hull" ) ) );
}

py::dict
min_area_rect( py::array_t< double, py::array::c_style | py::array::forcecast >
                 const& points )
{
  auto const box = VIAME_KERNEL_CALL( min_area_rect,
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
  return contour_array( VIAME_KERNEL_CALL( approx_poly,
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
    VIAME_KERNEL_CALL( label_components, as_image( array ), how, count );

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
    VIAME_KERNEL_CALL( filter_2d, as_image( array ), k,
                                     as_border( border ), constant ),
    array.ndim() == 3 );
}

template < typename T >
py::array
match_template( array_of< T > const& array, array_of< T > const& pattern )
{
  return as_array(
    VIAME_KERNEL_CALL( match_template_ncc, as_image( array ),
                                              as_image( pattern ) ),
    false );
}

/// `cv2.morphologyEx` for the two compositions VIAME asks for.
template < typename T >
py::array
morphology( array_of< T > const& array, std::string const& operation,
            std::string const& shape, int width, int height, int iterations )
{
  auto const element = as_element( shape, width, height );
  auto image = as_image( array );
  viame::image_of< T > work( image );

  bool const opening = ( operation == "open" );

  if( !opening && operation != "close" )
  {
    throw std::invalid_argument(
      "morphology operation must be open or close; got '" + operation + "'" );
  }

  // An opening is an erosion then a dilation and a closing the reverse.
  // `iterations` applies **each half** that many times -- n erosions and
  // then n dilations -- rather than repeating the pair, which is what
  // `cv2.morphologyEx` does and is not the same thing: at three iterations
  // the two differ by over a hundred grey levels.
  auto const passes = std::max( 1, iterations );

  auto const first = opening ? &viame::image_kernels::grey_erode< T >
                             : &viame::image_kernels::grey_dilate< T >;
  auto const second = opening ? &viame::image_kernels::grey_dilate< T >
                              : &viame::image_kernels::grey_erode< T >;

  {
    py::gil_scoped_release release;
    for( int pass = 0; pass < passes; ++pass ) { work = first( work, element ); }
    for( int pass = 0; pass < passes; ++pass ) { work = second( work, element ); }
  }

  return as_array( work, array.ndim() == 3 );
}

/// `cv2.copyMakeBorder`, in any of the modes `border_mode` carries.
///
/// The padding is read through `border_index`, which is the same rule the
/// filters and the warps pad with -- so an image padded here and then
/// filtered agrees with one filtered with the border rule applied inline,
/// which is the only way the two can be used together.
///
/// `value` is used by `constant` alone, as OpenCV's is.
template < typename T >
py::array
make_border( array_of< T > const& array, size_t top, size_t bottom,
             size_t left, size_t right, py::object const& value,
             std::string const& border )
{
  auto const source = as_image( array );
  auto const paint = as_colour( value );
  auto const mode = as_border( border );

  viame::image_of< T > out( source.width() + left + right,
                            source.height() + top + bottom,
                            source.depth() );

  auto const width = static_cast< long >( source.width() );
  auto const height = static_cast< long >( source.height() );

  {
    py::gil_scoped_release release;
    for( size_t y = 0; y < out.height(); ++y )
    {
      auto const from_y = viame::image_kernels::detail::border_index(
        static_cast< long >( y ) - static_cast< long >( top ), height, mode );

      for( size_t x = 0; x < out.width(); ++x )
      {
        auto const from_x = viame::image_kernels::detail::border_index(
          static_cast< long >( x ) - static_cast< long >( left ), width, mode );

        // -1 from either axis is `border_index` saying "outside, and the mode
        // has nothing to read" -- which only `CONSTANT` says.
        bool const inside = from_x >= 0 && from_y >= 0;

        for( size_t d = 0; d < out.depth(); ++d )
        {
          out( x, y, d ) = inside
            ? source( static_cast< size_t >( from_x ),
                      static_cast< size_t >( from_y ), d )
            : viame::image_kernels::saturate_pixel< T >(
                viame::image_kernels::detail::plane_value( paint, d ) );
        }
      }
    }
  }

  return as_array( out, array.ndim() == 3 );
}

template < typename T >
void
watershed( array_of< T > const& array,
           py::array_t< int32_t, py::array::c_style >& markers )
{
  auto buffer = markers.request( true );

  if( !( buffer.ndim == 2 ||
         ( buffer.ndim == 3 && buffer.shape[ 2 ] == 1 ) ) )
  {
    throw std::invalid_argument( "watershed markers are a single plane" );
  }

  viame::image_of< int32_t > seeds(
    static_cast< int32_t* >( buffer.ptr ),
    static_cast< size_t >( buffer.shape[ 1 ] ),
    static_cast< size_t >( buffer.shape[ 0 ] ),
    1, 1,
    static_cast< ptrdiff_t >( buffer.shape[ 1 ] ), 1 );

  VIAME_KERNEL_CALL( watershed, as_image( array ), seeds );
}

py::tuple
grab_cut( array_of< uint8_t > const& image,
          py::array_t< uint8_t, py::array::c_style >& mask,
          std::array< int, 4 > const& rect, int iterations,
          std::string const& mode,
          py::array_t< double, py::array::c_style >& background_model,
          py::array_t< double, py::array::c_style >& foreground_model,
          uint64_t rng_state )
{
  using viame::image_kernels::grabcut_mode;

  auto const source = as_image( image );

  grabcut_mode how;
  if( mode == "rect" )            { how = grabcut_mode::WITH_RECT; }
  else if( mode == "mask" )       { how = grabcut_mode::WITH_MASK; }
  else if( mode == "eval" )       { how = grabcut_mode::EVAL; }
  else if( mode == "eval_frozen" ) { how = grabcut_mode::EVAL_FREEZE_MODEL; }
  else
  {
    throw std::invalid_argument(
      "grab_cut: mode must be one of rect, mask, eval, eval_frozen; got '" +
      mode + "'" );
  }

  // The mask is written in place when the caller supplied one, as
  // `cv2.grabCut`'s is; `rect` mode replaces it and hands back a new array.
  viame::image_of< uint8_t > labels;
  if( how != grabcut_mode::WITH_RECT )
  {
    auto buffer = mask.request( true );
    if( !( buffer.ndim == 2 ||
           ( buffer.ndim == 3 && buffer.shape[ 2 ] == 1 ) ) )
    {
      throw std::invalid_argument( "grab_cut: the mask is a single plane" );
    }
    labels = viame::image_of< uint8_t >(
      static_cast< uint8_t* >( buffer.ptr ),
      static_cast< size_t >( buffer.shape[ 1 ] ),
      static_cast< size_t >( buffer.shape[ 0 ] ), 1, 1,
      static_cast< ptrdiff_t >( buffer.shape[ 1 ] ), 1 );
  }

  auto const as_model = []( py::array_t< double, py::array::c_style > const& a )
  {
    auto const buffer = a.request();
    auto const count = static_cast< size_t >( buffer.size );
    if( count != 0 && count != 65 )
    {
      throw std::invalid_argument(
        "grab_cut: a model array holds 65 doubles or nothing" );
    }
    auto const* data = static_cast< double const* >( buffer.ptr );
    return std::vector< double >( data, data + count );
  };

  auto background = as_model( background_model );
  auto foreground = as_model( foreground_model );

  {
    py::gil_scoped_release release;
    viame::image_kernels::grab_cut( source, labels, rect, iterations, how,
                                    background, foreground, rng_state );
  }

  py::array_t< double > out_background( 65 );
  py::array_t< double > out_foreground( 65 );
  std::copy( background.begin(), background.end(),
             out_background.mutable_data() );
  std::copy( foreground.begin(), foreground.end(),
             out_foreground.mutable_data() );

  return py::make_tuple( as_array( labels, false ), out_background,
                         out_foreground );
}

template < typename T >
py::array_t< double >
corner_subpix( array_of< T > const& array,
               py::array_t< double, py::array::c_style | py::array::forcecast >
                 const& corners,
               int half_width, int half_height, int iterations,
               double epsilon )
{
  auto const buffer = corners.request();

  if( buffer.ndim != 2 || buffer.shape[ 1 ] != 2 )
  {
    throw std::invalid_argument(
      "corner_subpix wants an N by 2 array of x, y" );
  }

  auto const* data = static_cast< double const* >( buffer.ptr );
  std::vector< std::pair< double, double > > points;
  points.reserve( static_cast< size_t >( buffer.shape[ 0 ] ) );

  for( Py_ssize_t n = 0; n < buffer.shape[ 0 ]; ++n )
  {
    points.emplace_back( data[ n * 2 ], data[ n * 2 + 1 ] );
  }

  VIAME_KERNEL_CALL( corner_subpix, as_image( array ), points, half_width,
                                       half_height, iterations, epsilon );

  py::array_t< double > out( std::vector< Py_ssize_t >{
    static_cast< Py_ssize_t >( points.size() ), 2 } );
  auto* destination = out.mutable_data();

  for( auto const& point : points )
  {
    *destination++ = point.first;
    *destination++ = point.second;
  }

  return out;
}

#undef VIAME_KERNEL_CALL

} // namespace

VIAME_PYTHON_MODULE( _image_kernels, m )
{
  m.doc() = "VIAME's own image kernels, so python needs no OpenCV";
  py::class_<gaussian_buffer> ( m, "GaussianWorkspace" )
      .def ( py::init<> (),
             "Reusable float Gaussian scratch; shared calls are serialized." );
  py::class_<stereo_buffer> ( m, "StereoWorkspace" )
      .def ( py::init<> (), "Reusable stereo cost rows; shared calls are serialized." );
  m.def ( "kernel_thread_count", &viame::image_kernels::kernel_thread_count,
          "Worker budget from VIAME_NUM_THREADS, read on first use." );

  for_every_pixel_type(m, "resize_letterbox", &resize_letterbox<uint8_t>,
    &resize_letterbox<uint16_t>, &resize_letterbox<float>,
    py::arg("image"), py::arg("width"), py::arg("height"),
    "Aspect-preserving area/Lanczos4 resize with black padding.");

  for_every_pixel_type( m, "resize", &resize< uint8_t >,
         &resize< uint16_t >, &resize< float >,
         py::arg( "image" ), py::arg( "width" ),
         py::arg( "height" ), py::arg( "interpolation" ) = "bilinear",
         "Resize using nearest, bilinear, bilinear_exact, bicubic or area "
         "interpolation." );

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

  for_every_pixel_type( m, "to_gray", &to_gray< uint8_t >,
         &to_gray< uint16_t >, &to_gray< float >, py::arg( "image" ),
         "RGB to single channel, by the same luma weights as the C++ side." );

  for_every_pixel_type( m, "to_rgb", &to_rgb< uint8_t >,
         &to_rgb< uint16_t >, &to_rgb< float >, py::arg( "image" ),
         "Single channel to three identical ones." );

  for_every_pixel_type( m, "swap_channels", &swap_channels< uint8_t >,
         &swap_channels< uint16_t >, &swap_channels< float >, py::arg( "image" ),
         "RGB to BGR, or back." );

  for_both_pixel_types( m, "to_hsv", &to_hsv< uint8_t >, &to_hsv< float >,
         py::arg( "image" ),
         "RGB to HSV, in the scaling the input type chooses -- as "
         "cv2.cvtColor does. uint8 gives hue 0..179 with saturation and "
         "value 0..255, OpenCV's 8-bit convention rather than a textbook's; "
         "float32 gives hue 0..360 with the other two over 0..1." );

  for_both_pixel_types( m, "from_hsv", &from_hsv< uint8_t >,
         &from_hsv< float >, py::arg( "image" ),
         "HSV back to RGB, on whichever scale to_hsv gives that type." );

  for_both_pixel_types( m, "to_hls", &to_hls< uint8_t >, &to_hls< float >,
         py::arg( "image" ),
         "RGB to HLS, on the same two scales as to_hsv." );

  for_both_pixel_types( m, "from_hls", &from_hls< uint8_t >,
         &from_hls< float >, py::arg( "image" ),
         "HLS back to RGB." );

  for_both_pixel_types( m, "to_lab", &to_lab< uint8_t >, &to_lab< float >,
         py::arg( "image" ),
         py::arg( "linear" ) = false,
         "RGB to CIE L*a*b*, 8-bit: L scaled to 0..255 and a and b offset "
         "by 128, again OpenCV's scaling. With linear set, the input is "
         "taken as linear light and the sRGB transfer curve is skipped, "
         "which is cv2.COLOR_LRGB2Lab rather than COLOR_RGB2Lab. A float32 "
         "image gets the real ranges instead -- L 0..100, a and b about "
         "-128..127 -- and is clamped to 0..1 on the way in, as OpenCV "
         "clamps it, so anything at or above 1 is white." );

  for_both_pixel_types( m, "from_lab", &from_lab< uint8_t >,
         &from_lab< float >, py::arg( "image" ),
         py::arg( "linear" ) = false,
         "L*a*b* back to RGB, on the same 8-bit scaling as to_lab. With "
         "linear set, the result is linear light -- cv2.COLOR_Lab2LRGB. "
         "float32 takes and returns the real ranges. The float pair is "
         "within 8e-05 of cv2 in this direction and about 0.3 of an L unit "
         "in the other, where OpenCV interpolates the sRGB curve off a "
         "1024 knot spline instead of evaluating it." );

  m.def( "denoise", &denoise, py::arg( "image" ),
         py::arg( "strength" ) = 3.0, py::arg( "patch" ) = 7,
         py::arg( "window" ) = 21,
         "Non-local means denoising, cv2.fastNlMeansDenoising with NORM_L2, "
         "for one to three planes. Exact: every weight is a fixed-point "
         "integer over a quantised distance, so there is no float to "
         "disagree about." );

  m.def( "denoise_colour", &denoise_colour, py::arg( "image" ),
         py::arg( "strength" ) = 3.0, py::arg( "colour_strength" ) = 3.0,
         py::arg( "patch" ) = 7, py::arg( "window" ) = 21,
         "cv2.fastNlMeansDenoisingColored: to L*a*b* through the **linear** "
         "transfer, L denoised at strength and the chroma pair together at "
         "colour_strength, then back. Note OpenCV converts with LBGR2Lab, "
         "not BGR2Lab, so denoising happens in linear light." );

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

  for_every_pixel_type( m, "fill_ellipse", &fill_ellipse< uint8_t >,
         &fill_ellipse< uint16_t >, &fill_ellipse< float >,
         py::arg( "image" ), py::arg( "x" ), py::arg( "y" ),
         py::arg( "radius_x" ), py::arg( "radius_y" ), py::arg( "colour" ),
         py::arg( "angle" ) = 0.0,
         "Fill an ellipse in place, which is cv2.ellipse at thickness -1. "
         "Filled only: an outlined ellipse is a polygonal approximation in "
         "OpenCV and a filled one is exact, so only the exact half is here." );

  for_both_pixel_types( m, "draw_circle", &draw_circle< uint8_t >,
         &draw_circle< uint16_t >, py::arg( "image" ), py::arg( "x" ),
         py::arg( "y" ), py::arg( "radius" ), py::arg( "colour" ),
         py::arg( "thickness" ) = 1,
         "cv2.circle, pixel for pixel at thickness 1. A thickness below "
         "zero fills." );

  m.def( "text_size", &text_size, py::arg( "text" ), py::arg( "scale" ) = 1,
         "The (width, height) of text, as cv2.getTextSize reports it." );

  for_every_pixel_type (
      m, "gaussian_blur", &gaussian_blur<uint8_t>, &gaussian_blur<uint16_t>,
      &gaussian_blur<float>, py::arg ( "image" ), py::arg ( "size" ),
      py::arg ( "sigma" ) = 0.0, py::arg ( "border" ) = "reflect_101",
      py::arg ( "workspace" ) = nullptr,
      "cv2.GaussianBlur. `size` is the odd kernel width and height, and "
      "sigma is derived from it when left at zero." );

  for_every_pixel_type( m, "gaussian_blur_float_taps",
         &gaussian_blur_float_taps< uint8_t >,
         &gaussian_blur_float_taps< uint16_t >,
         &gaussian_blur_float_taps< float >, py::arg( "image" ),
         py::arg( "size" ), py::arg( "sigma" ) = 0.0,
         py::arg( "border" ) = "reflect_101",
         "cv2.GaussianBlur as it behaves when handed a **submatrix**, which "
         "is cv2.sepFilter2D with the float kernel: the bit-exact "
         "fixed-point path is guarded on the input not being one. The two "
         "differ by a count on about a fifth of an 8-bit frame. ORB's "
         "pyramid blur is this one. Exact against cv2 except where the "
         "float accumulation lands on an exact half, which only a dyadic "
         "kernel -- `sigma=0` at size 9 or less -- produces: cv2 rounds "
         "those to even in its vector body and away from zero in its "
         "scalar remainder, so its own answer moves with the vector "
         "width. This takes the vector body." );

  for_every_pixel_type( m, "box_blur", &box_blur< uint8_t >,
         &box_blur< uint16_t >,
         &box_blur< float >, py::arg( "image" ), py::arg( "size" ),
         py::arg( "border" ) = "reflect_101", "cv2.blur." );

  for_every_pixel_type( m, "add_weighted", &add_weighted< uint8_t >,
         &add_weighted< uint16_t >,
         &add_weighted< float >, py::arg( "first" ),
         py::arg( "alpha" ), py::arg( "second" ), py::arg( "beta" ),
         py::arg( "gamma" ) = 0.0,
         "first * alpha + second * beta + gamma, saturated -- and for a "
         "float image \"saturated\" means nothing is clamped, as "
         "cv2.addWeighted on a float does not clamp either." );

  for_every_pixel_type( m, "normalize", &normalize< uint8_t >,
         &normalize< uint16_t >,
         &normalize< float >, py::arg( "image" ), py::arg( "low" ) = 0.0,
         py::arg( "high" ) = 255.0,
         "Rescale the image's range onto [low, high]. cv2.normalize with "
         "NORM_MINMAX, whose alpha and beta are the two ends in either "
         "order." );

  m.def( "smooth_globally", &smooth_globally, py::arg( "guide" ),
         py::arg( "image" ), py::arg( "lambda_" ), py::arg( "sigma" ),
         py::arg( "attenuation" ) = 0.25, py::arg( "iterations" ) = 3,
         "cv2.ximgproc.fastGlobalSmootherFilter: an edge aware smoother that "
         "alternates a horizontal and a vertical tridiagonal solve rather than "
         "solving the two dimensional system. The guide is one or three planes "
         "of bytes and gives the edges; the image is float." );

  m.def( "filter_disparity_wls", &filter_disparity_wls, py::arg( "guide" ),
         py::arg( "left" ), py::arg( "right" ), py::arg( "lambda_" ) = 8000.0,
         py::arg( "sigma" ) = 1.0, py::arg( "left_offset" ) = 0,
         py::arg( "right_offset" ) = 0, py::arg( "min_disparity" ) = 0,
         py::arg( "discontinuity_radius" ) = 5,
         py::arg( "lrc_threshold" ) = 24, py::arg( "roll_off" ) = 0.001,
         "cv2.ximgproc.DisparityWLSFilter with its confidence map on: the "
         "left and right disparity maps in sixteenths, smoothed towards the "
         "guide's edges and weighted by how much each pixel can be trusted. "
         "The result is float, still in sixteenths." );

  m.def( "stereo_bm", &stereo_bm, py::arg( "left" ), py::arg( "right" ),
         py::arg( "num_disparities" ) = 64, py::arg( "block_size" ) = 21,
         py::arg( "min_disparity" ) = 0, py::arg( "pre_filter" ) = "xsobel",
         py::arg( "pre_filter_size" ) = 9, py::arg( "pre_filter_cap" ) = 31,
         py::arg( "texture_threshold" ) = 10,
         py::arg( "uniqueness_ratio" ) = 15,
         py::arg( "speckle_window_size" ) = 0, py::arg( "speckle_range" ) = 0,
         py::arg( "disp12_max_diff" ) = -1,
         "cv2.StereoBM.compute: a signed 16 bit disparity map in sixteenths "
         "of a pixel, with (min_disparity - 1) * 16 meaning no disparity. "
         "Block matching rather than semi-global -- the sum of absolute "
         "differences over a square window, with no smoothness term, which "
         "is why it is a different algorithm and not a setting of "
         "`stereo_sgbm`. A pre_filter_cap above 31 or a block_size above 21 "
         "is refused: cv2 runs a different accumulation there, with a "
         "different sub-pixel rule at the ends of the disparity range." );

  m.def ( "stereo_sgbm", &stereo_sgbm, py::arg ( "left" ), py::arg ( "right" ),
          py::arg ( "min_disparity" ) = 0, py::arg ( "num_disparities" ) = 16,
          py::arg ( "block_size" ) = 3, py::arg ( "p1" ) = 0, py::arg ( "p2" ) = 0,
          py::arg ( "disp12_max_diff" ) = 0, py::arg ( "pre_filter_cap" ) = 0,
          py::arg ( "uniqueness_ratio" ) = 0, py::arg ( "speckle_window_size" ) = 0,
          py::arg ( "speckle_range" ) = 0, py::arg ( "mode" ) = "sgbm",
          py::arg ( "workspace" ) = nullptr,
          "cv2.StereoSGBM.compute: a signed 16 bit disparity map in sixteenths "
          "of a pixel, with (min_disparity - 1) * 16 meaning no disparity. "
          "`mode` is one of \"sgbm\", \"hh\" and \"sgbm_3way\", which are "
          "MODE_SGBM, MODE_HH and MODE_SGBM_3WAY -- three different "
          "aggregations rather than three settings of one." );

  m.def( "hough_circles", &hough_circles, py::arg( "image" ),
         py::arg( "dp" ) = 1.0, py::arg( "min_dist" ) = 1.0,
         py::arg( "canny_threshold" ) = 100.0,
         py::arg( "acc_threshold" ) = 100.0, py::arg( "min_radius" ) = 0,
         py::arg( "max_radius" ) = 0, py::arg( "max_circles" ) = -1,
         "cv2.HoughCircles with HOUGH_GRADIENT, as an N by 3 array of x, y "
         "and radius. `canny_threshold` is OpenCV's param1 and "
         "`acc_threshold` its param2." );

  m.def( "canny", &canny, py::arg( "image" ), py::arg( "low" ),
         py::arg( "high" ), py::arg( "aperture" ) = 3,
         py::arg( "l2_gradient" ) = false,
         "cv2.Canny: a single plane 8 bit edge map, 0 or 255. The gradient is "
         "a 16 bit Sobel with a replicated border and the direction is "
         "quantised in 15 bit fixed point, both of which OpenCV's answer "
         "depends on. The aperture is 3 or 5; cv2 also takes 7, where the 16 "
         "bit gradient saturates and the two part company." );

  m.def( "equalize", &equalize, py::arg( "image" ),
         "cv2.equalizeHist." );

  // uint8 and uint16, not float: `clahe` static_asserts on an integer
  // pixel and is right to. It equalises a histogram, which needs a bounded
  // range of discrete levels to build one over -- and `cv2.createCLAHE`
  // accepts 8 and 16 bit for the same reason.
  for_every_pixel_type( m, "median_blur", &median_blur< uint8_t >,
         &median_blur< uint16_t >, &median_blur< float >,
         py::arg( "image" ), py::arg( "size" ),
         "cv2.medianBlur. The size is odd and at least 3, and the border "
         "replicates, as OpenCV's does. Exact by construction: an odd window "
         "has a single middle sample." );

  for_both_pixel_types( m, "clahe", &clahe< uint8_t >, &clahe< uint16_t >,
         py::arg( "image" ), py::arg( "clip_limit" ) = 40.0,
         py::arg( "tiles_x" ) = 8, py::arg( "tiles_y" ) = 8,
         "Contrast limited adaptive histogram equalisation, which is what "
         "cv2.createCLAHE().apply() does." );

  for_every_pixel_type( m, "erode", &erode< uint8_t >,
         &erode< uint16_t >, &erode< float >,
         py::arg( "image" ), py::arg( "shape" ) = "rect",
         py::arg( "width" ) = 3, py::arg( "height" ) = 3,
         "Grey erosion. cv2.erode with cv2.getStructuringElement; the shape "
         "is one of rect, cross, disk." );

  for_every_pixel_type( m, "dilate", &dilate< uint8_t >,
         &dilate< uint16_t >, &dilate< float >,
         py::arg( "image" ), py::arg( "shape" ) = "rect",
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

  for_both_pixel_types( m, "distance_transform",
         &distance_transform< uint8_t >, &distance_transform< uint16_t >,
         py::arg( "mask" ),
         "The distance from each non-zero pixel to the nearest zero, as "
         "float32. cv2.distanceTransform with DIST_L2 and a mask of 3, which "
         "is a chamfer approximation and not the Euclidean distance its name "
         "suggests." );

  for_both_pixel_types( m, "find_borders", &find_borders< uint8_t >,
         &find_borders< uint16_t >, py::arg( "mask" ),
         "Every border of every component as (points, is_hole), which is "
         "cv2.findContours under RETR_LIST. find_contours gives the outer "
         "ones alone, which is RETR_EXTERNAL and what most callers want." );

  m.def( "arc_length", &arc_length, py::arg( "contour" ),
         py::arg( "closed" ) = true,
         "The length of a contour, which is cv2.arcLength." );

  m.def( "moments", &moments, py::arg( "contour" ),
         "A contour's spatial moments through the second order, which is "
         "cv2.moments on a contour." );

  m.def( "intersect_convex", &intersect_convex, py::arg( "first" ),
         py::arg( "second" ),
         "The overlap of two convex polygons as (area, points), which is "
         "cv2.intersectConvexConvex." );

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

  for_every_pixel_type( m, "match_template", &match_template< uint8_t >,
         &match_template< uint16_t >, &match_template< float >,
         py::arg( "image" ), py::arg( "pattern" ),
         "cv2.matchTemplate with TM_CCOEFF_NORMED, which is the only mode "
         "VIAME asks for. All planes score together, as OpenCV does, and "
         "the mean subtracted is per plane." );

  for_every_pixel_type( m, "morphology", &morphology< uint8_t >,
         &morphology< uint16_t >, &morphology< float >, py::arg( "image" ),
         py::arg( "operation" ), py::arg( "shape" ) = "rect",
         py::arg( "width" ) = 3, py::arg( "height" ) = 3,
         py::arg( "iterations" ) = 1,
         "cv2.morphologyEx for MORPH_OPEN and MORPH_CLOSE. `iterations` "
         "repeats the pair, as cv2 does." );

  for_every_pixel_type( m, "make_border", &make_border< uint8_t >,
         &make_border< uint16_t >, &make_border< float >, py::arg( "image" ),
         py::arg( "top" ), py::arg( "bottom" ), py::arg( "left" ),
         py::arg( "right" ), py::arg( "value" ) = 0,
         py::arg( "border" ) = "constant",
         "cv2.copyMakeBorder. `border` is constant, replicate, reflect, "
         "reflect_101 or wrap, and `value` is used by constant alone -- as "
         "OpenCV's is." );

  py::class_< mog2_wrapper >( m, "Mog2Background",
         "cv2.BackgroundSubtractorMOG2 with shadow detection off: a mixture "
         "of Gaussians per pixel, updated one frame at a time. `apply` "
         "returns the foreground mask, 255 where the pixel did not match the "
         "background. Stateful -- the frames have to arrive in order." )
    .def( py::init< int, double, int, double, double, double, double, double,
                    double >(),
          py::arg( "history" ) = 500, py::arg( "var_threshold" ) = 16.0,
          py::arg( "mixtures" ) = 5, py::arg( "background_ratio" ) = 0.9,
          py::arg( "var_threshold_gen" ) = 9.0, py::arg( "var_init" ) = 15.0,
          py::arg( "var_min" ) = 4.0, py::arg( "var_max" ) = 75.0,
          py::arg( "complexity_reduction" ) = 0.05 )
    .def( "apply", &mog2_wrapper::apply, py::arg( "image" ),
          py::arg( "learning_rate" ) = -1.0,
          "Update the mixture with one frame and return its foreground mask. "
          "A negative learning rate asks for OpenCV's automatic "
          "1 / min(2 * frames, history)." )
    .def( "reset", &mog2_wrapper::clear,
          "Forget every frame seen so far." )
    .def_property_readonly( "frames", &mog2_wrapper::frames,
          "How many frames have been through it." );

  m.def ( "fast_corners", &fast_corners,
          py::arg( "image" ), py::arg( "threshold" ) = 10,
          py::arg( "suppress" ) = true,
          "cv2.FAST with TYPE_9_16 on one plane of bytes. Returns an N by 3 "
          "float32 array of x, y and the suppression score, in raster order. "
          "The score is zero when `suppress` is false, as cv2's is." );

  for_both_pixel_types( m, "good_features_to_track",
         &good_features< uint8_t >, &good_features< float >,
         py::arg( "image" ), py::arg( "max_corners" ) = 1000,
         py::arg( "quality_level" ) = 0.01,
         py::arg( "min_distance" ) = 10.0, py::arg( "block_size" ) = 3,
         py::arg( "aperture" ) = 3,
         "cv2.goodFeaturesToTrack with the Shi-Tomasi measure -- the Harris "
         "one is not offered. Returns an N by 2 float32 array of x, y, "
         "strongest first, thinned so that no two are within "
         "`min_distance`. `max_corners` at or below zero means no limit." );

  for_both_pixel_types( m, "min_eigen_value", &min_eigen_value< uint8_t >,
         &min_eigen_value< float >, py::arg( "image" ),
         py::arg( "block_size" ) = 3, py::arg( "aperture" ) = 3,
         "cv2.cornerMinEigenVal: the smaller eigenvalue of the gradient "
         "covariance over a block, which is large only where the gradient "
         "points two ways at once. `aperture` has to be 3." );

  m.def( "lucas_kanade", &lucas_kanade, py::arg( "first" ),
         py::arg( "second" ), py::arg( "points" ),
         py::arg( "win_width" ) = 21, py::arg( "win_height" ) = 21,
         py::arg( "levels" ) = 3, py::arg( "iterations" ) = 30,
         py::arg( "epsilon" ) = 0.01, py::arg( "min_eigen" ) = 1e-4,
         py::arg( "threads" ) = 0,
         "cv2.calcOpticalFlowPyrLK. Returns the moved points as an N by 2 "
         "float32 array and a uint8 status, 1 where the point was followed. "
         "`epsilon` is squared and compared against the squared step, as "
         "OpenCV's is. `threads` divides the points, 0 for one per core; the "
         "answer does not depend on the division, and OpenCV parallelises the "
         "same loop." );

  for_every_pixel_type( m, "optical_flow", &optical_flow< uint8_t >,
         &optical_flow< uint16_t >, &optical_flow< float >,
         py::arg( "first" ), py::arg( "second" ),
         py::arg( "pyr_scale" ) = 0.5, py::arg( "levels" ) = 3,
         py::arg( "winsize" ) = 15, py::arg( "iterations" ) = 3,
         py::arg( "poly_n" ) = 5, py::arg( "poly_sigma" ) = 1.2,
         "cv2.calcOpticalFlowFarneback with `flags` at zero. Returns an "
         "H by W by 2 float32 array of the displacement from `first` to "
         "`second`, horizontal first, in pixels. Both images are a single "
         "plane." );

  m.def( "grab_cut", &grab_cut, py::arg( "image" ),
         py::arg( "mask" ), py::arg( "rect" ) = std::array< int, 4 >{ 0, 0, 0, 0 },
         py::arg( "iterations" ) = 1, py::arg( "mode" ) = "mask",
         py::arg( "background_model" ) =
           py::array_t< double, py::array::c_style >( 0 ),
         py::arg( "foreground_model" ) =
           py::array_t< double, py::array::c_style >( 0 ),
         py::arg( "rng_state" ) = 0xffffffffu,
         "cv2.grabCut. Returns (mask, background_model, foreground_model); "
         "the mask is written in place as well, except in `rect` mode which "
         "replaces it. `mode` is one of \"rect\", \"mask\", \"eval\" and "
         "\"eval_frozen\", which are GC_INIT_WITH_RECT, GC_INIT_WITH_MASK, "
         "GC_EVAL and GC_EVAL_FREEZE_MODEL. Labels are 0 background, 1 "
         "foreground, 2 probably background, 3 probably foreground, as cv2's "
         "are. **cv2's own answer depends on `theRNG()`**, a global mutable "
         "generator its k-means seeding draws from, so this starts from the "
         "state a fresh process has -- which is what cv2 gives before "
         "anything else has used it. `rng_state` is that state, so passing the "
         "same number `cv2.setRNGSeed` was given reproduces cv2 at another "
         "point." );

  for_both_pixel_types( m, "watershed", &watershed< uint8_t >,
         &watershed< uint16_t >, py::arg( "image" ), py::arg( "markers" ),
         "cv2.watershed: Meyer's flooding over a hierarchical queue, "
         "modifying `markers` in place. A positive marker seeds a region, "
         "zero is ground to claim, and where two regions meet the pixel "
         "becomes -1. The image's one pixel border is -1 by definition, as "
         "OpenCV's is." );

  for_every_pixel_type( m, "corner_subpix", &corner_subpix< uint8_t >,
         &corner_subpix< uint16_t >, &corner_subpix< float >,
         py::arg( "image" ), py::arg( "corners" ),
         py::arg( "half_width" ) = 5, py::arg( "half_height" ) = 5,
         py::arg( "iterations" ) = 40, py::arg( "epsilon" ) = 0.001,
         "cv2.cornerSubPix. At a corner the image gradient is orthogonal to "
         "the vector from the corner to the pixel carrying it, which is a "
         "two by two system in the corner position; this solves it and "
         "re-centres until it settles. Returns the refined N by 2." );
}
