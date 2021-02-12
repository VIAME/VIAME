// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "high_pass_filter.h"

#include <arrows/vxl/image_container.h>
#include <vital/util/enum_converter.h>

#include <vil/vil_convert.h>
#include <vil/vil_image_view.h>
#include <vil/vil_math.h>
#include <vil/vil_plane.h>
#include <vil/vil_transpose.h>

namespace kwiver {

namespace arrows {

namespace vxl {

namespace { // anonymous

enum filter_mode
{
  MODE_box,
  MODE_bidir,
};

ENUM_CONVERTER( mode_converter, filter_mode,
                { "box", MODE_box }, { "bidir", MODE_bidir } )

// ----------------------------------------------------------------------------
template < typename T > struct accumulator;

template <>
struct accumulator< char >
{ using type = int; };

template <>
struct accumulator< signed char >
{ using type = signed int; };

template <>
struct accumulator< unsigned char >
{ using type = unsigned int; };

template <>
struct accumulator< signed short >
{ using type = signed int; };

template <>
struct accumulator< unsigned short >
{ using type = unsigned int; };

template <>
struct accumulator< signed int >
{ using type = signed long long; };

template <>
struct accumulator< unsigned int >
{ using type = unsigned long long; };

template <>
struct accumulator< float >
{ using type = double; };

template <>
struct accumulator< double >
{ using type = double; };

} // namespace <anonymous>

// ----------------------------------------------------------------------------
/// Private implementation class
class high_pass_filter::priv
{
public:
  priv( high_pass_filter* parent )
    : p{ parent }
  {
  }

  ~priv()
  {
  }

  // Internal parameters/settings
  filter_mode mode = MODE_box;
  unsigned kernel_width = 7;
  unsigned kernel_height = 7;
  bool treat_as_interlaced = false;
  bool output_net_only = false;
  high_pass_filter* p;

  // Perform box filtering
  template < typename PixType > vil_image_view< PixType >
  box_high_pass_filter( vil_image_view< PixType > const& grey_img );
  // Given an input grayscale image, and a smoothed version of this greyscale
  // image, calculate the bidirectional filter response in the horizontal
  // direction.
  template < typename PixType > void
  horizontal_box_bidirectional_pass(
    vil_image_view< PixType > const& grey,
    vil_image_view< PixType > const& smoothed,
    vil_image_view< PixType >& output, unsigned kernel_width );
  // Given an input grayscale image, and a smoothed version of this greyscale
  // image, calculate the bidirectional filter response in the vertical
  // direction.
  template < typename PixType > void
  vertical_box_bidirectional_pass(
    vil_image_view< PixType > const& grey,
    vil_image_view< PixType > const& smoothed,
    vil_image_view< PixType >& output, unsigned kernel_height );
  // Perform bidirectional filtering
  template < typename PixType > vil_image_view< PixType >
  bidirection_box_filter( vil_image_view< PixType > const& grey_img );
  // Function which returns an image view of the even rows of an image
  template < typename PixType > inline vil_image_view< PixType >
  even_rows( vil_image_view< PixType > const& im );
  // Function which returns an image view of the odd rows of an image
  template < typename PixType > inline vil_image_view< PixType >
  odd_rows( vil_image_view< PixType > const& im );
  // Fast 1D (horizontal) box filter smoothing
  template < typename PixType > void
  box_average_horizontal( vil_image_view< PixType > const& src,
                          vil_image_view< PixType >& dst,
                          unsigned kernel_width );
  // Fast 1D (vertical) box filter smoothing
  template < typename PixType > void
  box_average_vertical( vil_image_view< PixType > const& src,
                        vil_image_view< PixType >& dst,
                        unsigned kernel_height );
  // Apply a high pass filter to one frame and return the output
  template < typename PixType > kwiver::vital::image_container_sptr
  filter( vil_image_view< PixType >& input );
};

// ----------------------------------------------------------------------------
template < typename PixType >
vil_image_view< PixType >
high_pass_filter::priv
::box_high_pass_filter( vil_image_view< PixType > const& grey_img )
{
  vil_image_view< PixType > output{ grey_img.ni(), grey_img.nj(), 3 };

  vil_image_view< PixType > filter_x = vil_plane( output, 0 );
  vil_image_view< PixType > filter_y = vil_plane( output, 1 );
  vil_image_view< PixType > filter_xy = vil_plane( output, 2 );

  box_average_horizontal( grey_img, filter_x, kernel_width );
  box_average_vertical( grey_img, filter_y, kernel_height );

  // Apply horizontal smoothing to the vertically smoothed image to get a 2D
  // box filter
  box_average_horizontal( filter_y, filter_xy, kernel_width );

  // Report the difference between the pixel value and all of the smoothed
  // responses
  vil_math_image_abs_difference( grey_img, filter_x, filter_x );
  vil_math_image_abs_difference( grey_img, filter_y, filter_y );
  vil_math_image_abs_difference( grey_img, filter_xy, filter_xy );

  return output;
}

// ----------------------------------------------------------------------------
template < typename PixType >
void
high_pass_filter::priv
::horizontal_box_bidirectional_pass( vil_image_view< PixType > const& grey,
                                     vil_image_view< PixType > const& smoothed,
                                     vil_image_view< PixType >& output,
                                     unsigned kernel_width )
{
  unsigned const ni = grey.ni();
  unsigned const nj = grey.nj();

  auto const offset = ( kernel_width / 2 ) + 1;

  if( ni < 2 * offset + 1 )
  {
    return;
  }

  output.fill( 0 );

  PixType diff1 = 0;
  PixType diff2 = 0;

  for( unsigned j = 0; j < nj; ++j )
  {
    for( unsigned i = offset; i < ni - offset; ++i )
    {
      PixType const& val = grey( i, j );
      PixType const& avg1 = smoothed( i - offset, j );
      PixType const& avg2 = smoothed( i + offset, j );

      diff1 = ( val > avg1 ? val - avg1 : avg1 - val );
      diff2 = ( val > avg2 ? val - avg2 : avg2 - val );

      output( i, j ) = std::min( diff1, diff2 );
    }
  }
}

// ----------------------------------------------------------------------------
template < typename PixType >
void
high_pass_filter::priv
::vertical_box_bidirectional_pass( vil_image_view< PixType > const& grey,
                                   vil_image_view< PixType > const& smoothed,
                                   vil_image_view< PixType >& output,
                                   unsigned kernel_height )
{
  vil_image_view< PixType > grey_t = vil_transpose( grey );
  vil_image_view< PixType > smoothed_t = vil_transpose( smoothed );
  vil_image_view< PixType > output_t = vil_transpose( output );

  horizontal_box_bidirectional_pass( grey_t, smoothed_t,
                                     output_t, kernel_height );
}

// ----------------------------------------------------------------------------
template < typename PixType >
vil_image_view< PixType >
high_pass_filter::priv
::bidirection_box_filter( vil_image_view< PixType > const& grey_img )
{
  vil_image_view< PixType > output{ grey_img.ni(), grey_img.nj(), 3 };

  // Create views of each plane
  vil_image_view< PixType > filter_x = vil_plane( output, 0 );
  vil_image_view< PixType > filter_y = vil_plane( output, 1 );
  vil_image_view< PixType > filter_xy = vil_plane( output, 2 );

  // Report the difference between the pixel value, and all of the smoothed
  // responses, using the xy channel as a temporary buffer to avoid additional
  // memory allocation.
  box_average_vertical( grey_img, filter_xy, kernel_height );
  horizontal_box_bidirectional_pass( grey_img, filter_xy, filter_x,
                                     kernel_width );
  box_average_horizontal( grey_img, filter_xy, kernel_width );
  vertical_box_bidirectional_pass( grey_img, filter_xy, filter_y,
                                   kernel_height );
  vil_math_image_max( filter_x, filter_y, filter_xy );

  return output;
}

// ----------------------------------------------------------------------------
template < typename PixType >
inline vil_image_view< PixType >
high_pass_filter::priv
::even_rows( vil_image_view< PixType > const& im )
{
  return vil_image_view< PixType >{
    im.memory_chunk(), im.top_left_ptr(),
    im.ni(), ( im.nj() + 1 ) / 2, im.nplanes(),
    im.istep(), im.jstep() * 2, im.planestep() };
}

// ----------------------------------------------------------------------------
template < typename PixType >
inline vil_image_view< PixType >
high_pass_filter::priv
::odd_rows( vil_image_view< PixType > const& im )
{
  return vil_image_view< PixType >{
    im.memory_chunk(), im.top_left_ptr() + im.jstep(),
    im.ni(), ( im.nj() ) / 2, im.nplanes(),
    im.istep(), im.jstep() * 2, im.planestep() };
}

// ----------------------------------------------------------------------------
template < typename PixType >
void
high_pass_filter::priv
::box_average_horizontal( vil_image_view< PixType > const& src,
                          vil_image_view< PixType >& dst,
                          unsigned kernel_width )
{
  if( src.ni() <= 0 )
  {
    LOG_ERROR( p->logger(), "Image width must be non-zero" );
  }
  if( kernel_width % 2 == 0 )
  {
    LOG_ERROR( p->logger(), "Kernel width must be odd" );
  }

  if( kernel_width >= src.ni() )
  {
    // Force width to be smaller than image width and odd
    kernel_width = ( src.ni() == 1 ? 1 : ( src.ni() - 2 ) | 0x01 );
  }

  auto const ni = src.ni();
  auto const nj = src.nj();
  auto const np = src.nplanes();

  dst.set_size( ni, nj, np );

  auto const half_width = kernel_width / 2;

  auto const istepS = src.istep();
  auto const jstepS = src.jstep();
  auto const pstepS = src.planestep();
  auto const istepD = dst.istep();
  auto const jstepD = dst.jstep();
  auto const pstepD = dst.planestep();

  PixType const* planeS = src.top_left_ptr();
  PixType* planeD = dst.top_left_ptr();

  for( unsigned p = 0; p < np; ++p, planeS += pstepS, planeD += pstepD )
  {
    PixType const* rowS = planeS;
    PixType*       rowD = planeD;

    for( unsigned j = 0; j < nj; ++j, rowS += jstepS, rowD += jstepD )
    {
      PixType const* pixelS1 = rowS;
      PixType const* pixelS2 = rowS;
      PixType*       pixelD = rowD;

      // fast box filter smoothing by adding one pixel to the sum and
      // subtracting another pixel at each step
      using accumulator_t = typename accumulator< PixType >::type;

      auto const kw = static_cast< accumulator_t >( kernel_width );
      accumulator_t sum = 0;
      unsigned i = 0;

      // initialize the sum for half the kernel width
      for(; i <= half_width; ++i, pixelS2 += istepS )
      {
        sum += *pixelS2;
      }

      // starting boundary case: the kernel width is expanding
      for(; i < kernel_width; ++i, pixelS2 += istepS, pixelD += istepD )
      {
        *pixelD =
          static_cast< PixType >( sum / static_cast< accumulator_t >( i ) );
        sum += *pixelS2;
      }

      // general case: add the leading edge and remove the trailing edge.
      for(; i < ni;
          ++i, pixelS1 += istepS, pixelS2 += istepS, pixelD += istepD )
      {
        *pixelD = static_cast< PixType >( sum / kw );
        sum -= *pixelS1;
        sum += *pixelS2;
      }

      // ending boundary case: the kernel is shrinking
      for( i = kernel_width; i > half_width;
           --i, pixelS1 += istepS, pixelD += istepD )
      {
        *pixelD =
          static_cast< PixType >( sum / static_cast< accumulator_t >( i ) );
        sum -= *pixelS1;
      }
    }
  }
}

// ----------------------------------------------------------------------------
template < typename PixType >
void
high_pass_filter::priv
::box_average_vertical( vil_image_view< PixType > const& src,
                        vil_image_view< PixType >& dst,
                        unsigned kernel_height )
{
  if( treat_as_interlaced )
  {
    // if interlaced, split the image into odd and even views transpose all
    // input and ouput images so that the horizontal smoothing function
    // produces vertical smoothing
    auto const im_even_t = vil_transpose( even_rows( src ) );
    auto const im_odd_t = vil_transpose( odd_rows( src ) );

    auto smooth_even_t = vil_transpose( even_rows( dst ) );
    auto smooth_odd_t = vil_transpose( odd_rows( dst ) );

    // Use a half size odd kernel since images are half height, rounded up to
    // the next odd number
    auto const half_kernel_height = 2 * ( kernel_height >> 2 ) + 1;

    // Smooth transposed inputs with the horizontal smoothing
    box_average_horizontal( im_even_t, smooth_even_t, half_kernel_height );
    box_average_horizontal( im_odd_t, smooth_odd_t, half_kernel_height );
  }
  else
  {
    // if not interlaced, transpose inputs and outputs and apply horizontal
    // smoothing.
    auto const grey_img_t = vil_transpose( src );
    auto smooth_t = vil_transpose( dst );
    box_average_horizontal( grey_img_t, smooth_t, kernel_height );
  }
}

// ----------------------------------------------------------------------------
template < typename PixType >
kwiver::vital::image_container_sptr
high_pass_filter::priv
::filter( vil_image_view< PixType >& input )
{
  vil_image_view< PixType > output;
  switch( mode )
  {
    case MODE_box:
      output = box_high_pass_filter( input );
      break;
    case MODE_bidir:
      output = bidirection_box_filter( input );
      break;
    default:
      return nullptr;
  }

  // Only report the last plane, which contains the summation of directional
  // filtering
  if( output_net_only && output.nplanes() > 1 )
  {
    output = vil_plane( output, output.nplanes() - 1 );
  }

  return std::make_shared< vxl::image_container >( output );
}

// ----------------------------------------------------------------------------
high_pass_filter
::high_pass_filter()
  : d( new priv( this ) )
{
  attach_logger( "arrows.vxl.high_pass_filter" );
}

// ----------------------------------------------------------------------------
high_pass_filter
::~high_pass_filter()
{
}

// ----------------------------------------------------------------------------
vital::config_block_sptr
high_pass_filter
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "mode", mode_converter().to_string(
                       d->mode ),
                     "Operating mode of this filter, possible values: " +
                     mode_converter().element_name_string() );
  config->set_value( "kernel_width", d->kernel_width,
                     "Pixel width of smoothing kernel" );
  config->set_value( "kernel_height", d->kernel_height,
                     "Pixel height of smoothing kernel" );
  config->set_value( "treat_as_interlaced", d->treat_as_interlaced,
                     "Process alternating rows independently" );
  config->set_value( "output_net_only", d->output_net_only,
                     "If set to false, the output image will contain multiple "
                     "planes, each representing the modal filter applied at "
                     "different orientations, as opposed to a single plane "
                     "image representing the sum of filters applied in all "
                     "directions." );

  return config;
}

// ----------------------------------------------------------------------------
void
high_pass_filter
::set_configuration( vital::config_block_sptr in_config )
{
  // Starting with our generated vital::config_block to ensure that assumed
  // values are present. An alternative is to check for key presence before
  // performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  // Settings for filtering
  d->mode = config->get_enum_value< mode_converter >( "mode" );
  d->kernel_width = config->get_value< unsigned >( "kernel_width" );
  d->kernel_height = config->get_value< unsigned >( "kernel_height" );
  d->treat_as_interlaced = config->get_value< bool >( "treat_as_interlaced" );
  d->output_net_only = config->get_value< bool >( "output_net_only" );
}

// ----------------------------------------------------------------------------
bool
high_pass_filter
::check_configuration( vital::config_block_sptr config ) const
{
  unsigned width = config->get_value< unsigned >( "kernel_width" );

  if( width % 2 == 0 )
  {
    LOG_ERROR( logger(), "Kernel width must be odd but is "
               << width );
    return false;
  }

  unsigned height = config->get_value< unsigned >( "kernel_height" );
  if( height % 2 == 0 )
  {
    LOG_ERROR( logger(), "Kernel height must be odd but is "
               << height );
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
high_pass_filter
::filter( kwiver::vital::image_container_sptr image_data )
{
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );

  if( view->nplanes() == 3 )
  {
    view = vil_convert_to_grey_using_average( view );
  }
  else if( view->nplanes() != 1 )
  {
    LOG_ERROR( logger(), "Expected 1 or 3 channels but recieved "
               << view->nplanes() );
    return kwiver::vital::image_container_sptr();
  }

#define HANDLE_CASE( T )                                          \
  case T:                                                         \
  {                                                               \
    using pix_t = vil_pixel_format_type_of< T >::component_type;  \
    auto input = static_cast< vil_image_view< pix_t > >( *view ); \
    return d->filter( input );                                    \
  }

  switch( view->pixel_format() )
  {
    HANDLE_CASE( VIL_PIXEL_FORMAT_BYTE )
    HANDLE_CASE( VIL_PIXEL_FORMAT_SBYTE )
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_16 )
    HANDLE_CASE( VIL_PIXEL_FORMAT_INT_16 )
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_32 )
    HANDLE_CASE( VIL_PIXEL_FORMAT_INT_32 )
    HANDLE_CASE( VIL_PIXEL_FORMAT_FLOAT )
    HANDLE_CASE( VIL_PIXEL_FORMAT_DOUBLE )
#undef HANDLE_CASE

    default:
      LOG_ERROR( logger(), "Invalid input format " << view->pixel_format()
                                                   << " type received" );
      return nullptr;
  }

  return nullptr;
}

} // namespace vxl

} // namespace arrows

} // namespace kwiver
