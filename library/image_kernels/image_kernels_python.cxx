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
#include <viame/image_kernels/resample.h>
#include <viame/image_kernels/warp.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
}
