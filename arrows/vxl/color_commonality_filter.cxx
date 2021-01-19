// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "color_commonality_filter.h"

#include <arrows/vxl/image_container.h>

#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_intersection.h>
#include <vil/algo/vil_gauss_filter.h>
#include <vil/vil_crop.h>
#include <vil/vil_fill.h>
#include <vil/vil_image_view.h>

#include <limits>
#include <type_traits>

namespace kwiver {

namespace arrows {

namespace vxl {

namespace {

/// Settings for the below functions
struct color_commonality_filter_settings
{
  // Resolution per channel when building a histogram detailing commonality
  unsigned int resolution_per_channel;

  // Scale the output image (which will by default by in the range [0,1])
  // by this amount. Set as 0 to scale to the input type-specific maximum.
  unsigned int output_scale_factor;

  // Instead of computing the per-pixel color commonality out of all the
  // pixels in the entire image, should we instead compute it in grids
  // windowed across it?
  bool grid_image;
  unsigned grid_resolution_height;
  unsigned grid_resolution_width;

  // [Advanced] A pointer to a temporary buffer for the histogram, which
  // can (a) prevent having to reallocate it over and over again, and
  // (b) allow the user to use it as a by-product of the main operation.
  // If set to NULL, will use an internal histogram buffer.
  std::vector< unsigned >* histogram;

  // Default constructor
  color_commonality_filter_settings()
    : resolution_per_channel( 8 ),
      output_scale_factor( 0 ),
      grid_image( false ),
      grid_resolution_height( 5 ),
      grid_resolution_width( 6 ),
      histogram( NULL )
  {}
};

// Simple helper functions
inline bool
is_power_of_two( const unsigned num )
{
  return ( ( num > 0 ) && ( ( num & ( num - 1 ) ) == 0 ) );
}

inline unsigned int
integer_log2( unsigned int value )
{
  unsigned int l = 0;
  while( ( value >> l ) > 1 ) { ++l; }
  return l;
}

// Intersect a region with some image boundaries
template < typename PixType >
void
check_region_boundaries( vgl_box_2d< int >& bbox,
                         const vil_image_view< PixType >& img )
{
  vgl_box_2d< int > boundaries( 0, img.ni(), 0, img.nj() );
  bbox = vgl_intersection( boundaries, bbox );
}

// Point an image view to a rectangular region in the image
template < typename PixType, typename BoxType >
void
point_view_to_region( const vil_image_view< PixType >& src,
                      const vgl_box_2d< BoxType >& region,
                      vil_image_view< PixType >& dst )
{
  // Early exit case, no crop required
  if( region.min_x() == 0 && region.min_y() == 0 &&
      region.max_x() == static_cast< BoxType >( src.ni() ) &&
      region.max_y() == static_cast< BoxType >( src.nj() ) )
  {
    dst = src;
    return;
  }

  // Validate boundaries
  vgl_box_2d< BoxType > to_crop = region;
  check_region_boundaries( to_crop, src );

  // Make sure width and height are non-zero
  if( to_crop.width() == 0 || to_crop.height() == 0 )
  {
    return;
  }

  // Perform crop
  dst = vil_crop( src,
                  to_crop.min_x(),
                  to_crop.width(),
                  to_crop.min_y(),
                  to_crop.height() );
}

// Alternative call for the above
template < typename PixType, typename BoxType >
vil_image_view< PixType >
point_view_to_region( const vil_image_view< PixType >& src,
                      const vgl_box_2d< BoxType >& region )
{
  vil_image_view< PixType > output;
  point_view_to_region( src, region, output );
  return output;
}

// An optimized (unsafe if used incorrectly) function which populates
// a n^p dimensional histogram from integer image 'input' given the
// resolution of each channel of the histogram, and a bitshift value
// which maps each value to its channel step for the histogram
template < class InputType >
void
populate_image_histogram( const vil_image_view< InputType >& input,
                          unsigned* hist_top_left,
                          const unsigned bitshift,
                          const std::ptrdiff_t* hist_steps )
{
  // Get image properties
  const unsigned ni = input.ni();
  const unsigned nj = input.nj();
  const unsigned np = input.nplanes();
  const std::ptrdiff_t istep = input.istep();
  const std::ptrdiff_t jstep = input.jstep();
  const std::ptrdiff_t pstep = input.planestep();

  // Filter image
  const InputType* row = input.top_left_ptr();
  for( unsigned j = 0; j < nj; ++j, row += jstep )
  {
    const InputType* pixel = row;
    for( unsigned i = 0; i < ni; ++i, pixel += istep )
    {
      unsigned step = 0;
      const InputType* plane = pixel;
      for( unsigned p = 0; p < np; ++p, plane += pstep )
      {
        step += hist_steps[ p ] * ( *plane >> bitshift );
      }
      ++( *( hist_top_left + step ) );
    }
  }
}

// Integer-typed filtering main loop
template < class InputType, class OutputType >
typename std::enable_if< std::is_integral< OutputType >::value >::type

filter_color_image( const vil_image_view< InputType >& input,
                    vil_image_view< OutputType >& output,
                    std::vector< unsigned >& histogram,
                    const color_commonality_filter_settings& options )
{
  assert( input.ni() == output.ni() && input.nj() == output.nj() );
  assert( is_power_of_two( options.resolution_per_channel ) );

  if( input.ni() == 0 || input.nj() == 0 )
  {
    return;
  }

  // Configure output scaling settings based on the output type and user
  // settings
  const InputType input_type_max = std::numeric_limits< InputType >::max();
  const OutputType output_type_max = std::numeric_limits< OutputType >::max();
  const unsigned histogram_threshold =
    static_cast< unsigned >( output_type_max );
  unsigned histogram_scale_factor = options.output_scale_factor;

  // Use type default options if no scale factor specified
  if( histogram_scale_factor == 0 )
  {
    histogram_scale_factor = histogram_threshold;
  }

  // Populate histogram steps for each channel of the hist
  std::vector< std::ptrdiff_t > histsteps( input.nplanes() );
  histsteps[ 0 ] = 1;

  for( unsigned p = 1; p < input.nplanes(); ++p )
  {
    histsteps[ p ] = histsteps[ p - 1 ] * options.resolution_per_channel;
  }

  // Fill in histogram of the input image
  const unsigned bitshift =
    integer_log2( ( static_cast< unsigned >( input_type_max ) + 1 ) /
                  options.resolution_per_channel );

  unsigned* hist_top_left = &histogram[ 0 ];

  populate_image_histogram( input, hist_top_left, bitshift, &histsteps[ 0 ] );

  // Normalize histogram to the output types range
  unsigned sum = 0;

  for( unsigned i = 0; i < histogram.size(); ++i )
  {
    sum += histogram[ i ];
  }

  // Fill in color commonality image from the compiled histogram
  for( unsigned i = 0; i < histogram.size(); ++i )
  {
    unsigned value = ( histogram_scale_factor * histogram[ i ] ) / sum;
    histogram[ i ] =
      ( value > histogram_threshold ? histogram_threshold : value );
  }

  const unsigned ni = input.ni();
  const unsigned nj = input.nj();
  const unsigned np = input.nplanes();
  const std::ptrdiff_t istep = input.istep();
  const std::ptrdiff_t jstep = input.jstep();
  const std::ptrdiff_t pstep = input.planestep();

  const InputType* row = input.top_left_ptr();
  for( unsigned j = 0; j < nj; ++j, row += jstep )
  {
    const InputType* pixel = row;
    for( unsigned i = 0; i < ni; ++i, pixel += istep )
    {
      const InputType* plane = pixel;
      unsigned step = 0;
      for( unsigned p = 0; p < np; ++p, plane += pstep )
      {
        step += histsteps[ p ] * ( *plane >> bitshift );
      }
      output( i, j ) = *( hist_top_left + step );
    }
  }
}

// Float-typed output filtering main loop
template < class InputType, class OutputType >
typename std::enable_if< !std::is_integral< OutputType >::value >::type

filter_color_image( const vil_image_view< InputType >& input,
                    vil_image_view< OutputType >& output,
                    std::vector< unsigned >& histogram,
                    const color_commonality_filter_settings& options )
{
  assert( input.ni() == output.ni() && input.nj() == output.nj() );
  assert( is_power_of_two( options.resolution_per_channel ) );

  if( input.ni() == 0 || input.nj() == 0 )
  {
    return;
  }

  // Configure output scaling settings based on the output type and user
  // settings
  const InputType input_type_max = std::numeric_limits< InputType >::max();

  // Populate histogram steps for each channel of the hist
  std::vector< std::ptrdiff_t > histsteps( input.nplanes() );
  histsteps[ 0 ] = 1;

  for( unsigned p = 1; p < input.nplanes(); ++p )
  {
    histsteps[ p ] = histsteps[ p - 1 ] * options.resolution_per_channel;
  }

  // Fill in histogram of the input image
  const unsigned bitshift =
    integer_log2( ( static_cast< unsigned >( input_type_max ) + 1 ) /
                  options.resolution_per_channel );

  unsigned* hist_top_left = &histogram[ 0 ];

  populate_image_histogram( input, hist_top_left, bitshift, &histsteps[ 0 ] );

  // Normalize histogram to the output types range
  unsigned sum = 0;

  for( unsigned i = 0; i < histogram.size(); ++i )
  {
    sum += histogram[ i ];
  }

  // Use type default options if no scale factor specified
  OutputType scale_factor = 1.0;
  if( options.output_scale_factor != 0 )
  {
    scale_factor = static_cast< OutputType >( options.output_scale_factor );
  }
  scale_factor = scale_factor / sum;

  // Fill in color commonality image from the compiled histogram
  const unsigned ni = input.ni();
  const unsigned nj = input.nj();
  const unsigned np = input.nplanes();
  const std::ptrdiff_t istep = input.istep();
  const std::ptrdiff_t jstep = input.jstep();
  const std::ptrdiff_t pstep = input.planestep();

  const InputType* row = input.top_left_ptr();
  for( unsigned j = 0; j < nj; ++j, row += jstep )
  {
    const InputType* pixel = row;

    for( unsigned i = 0; i < ni; ++i, pixel += istep )
    {
      const InputType* plane = pixel;
      unsigned step = 0;
      for( unsigned p = 0; p < np; ++p, plane += pstep )
      {
        step += histsteps[ p ] * ( *plane >> bitshift );
      }
      output( i, j ) = scale_factor * ( *( hist_top_left + step ) );
    }
  }
}

/// Create an output image indicating the relative commonality of each
/// input pixel's color occuring in the entire input image. A lower
/// value in the output image corresponds to that pixels value being
/// less common in the entire input image.
///
/// Functions by first building a histogram of the input image, then,
/// for each pixel, looking up the value in the histogram and scaling
/// this value by a given factor.
template < class InputType, class OutputType >
void
perform_filtering( const vil_image_view< InputType >& input,
                   vil_image_view< OutputType >& output,
                   const color_commonality_filter_settings& options )
{
  assert( std::numeric_limits< InputType >::is_integer );
  assert( is_power_of_two( options.resolution_per_channel ) );

  // Set output image size
  output.set_size( input.ni(), input.nj() );

  // If we are in grid mode, simply call this function recursively
  if( options.grid_image )
  {
    color_commonality_filter_settings recursive_options = options;
    recursive_options.grid_image = false;

    // Formulate grided regions
    unsigned ni = input.ni();
    unsigned nj = input.nj();

    for( unsigned j = 0; j < options.grid_resolution_height; ++j )
    {
      for( unsigned i = 0; i < options.grid_resolution_width; ++i )
      {
        // Top left point for region
        int ti = ( i * ni ) / options.grid_resolution_width;
        int tj = ( j * nj ) / options.grid_resolution_height;

        // Bottom right
        int bi = ( ( i + 1 ) * ni ) / options.grid_resolution_width;
        int bj = ( ( j + 1 ) * nj ) / options.grid_resolution_height;

        // Formulate rect region
        vgl_box_2d< int > region( ti, bi, tj, bj );

        vil_image_view< InputType > region_data_ptr =
          point_view_to_region( input, region );
        vil_image_view< OutputType > output_data_ptr =
          point_view_to_region( output, region );

        // Process rect region independent of one another
        perform_filtering( region_data_ptr, output_data_ptr,
                           recursive_options );
      }
    }

    // We are done, we already processed the entire image
    return;
  }

  // Reset histogram (use internal or external version)
  unsigned int hist_size = options.resolution_per_channel;

  if( input.nplanes() == 3 )
  {
    hist_size = options.resolution_per_channel *
                options.resolution_per_channel *
                options.resolution_per_channel;
  }

  bool use_external_hist = ( options.histogram != NULL );

  std::vector< unsigned >* histogram = options.histogram;

  if( !use_external_hist )
  {
    histogram = new std::vector< unsigned >( hist_size );
  }

  histogram->resize( hist_size );

  std::fill( histogram->begin(), histogram->end(), 0 );

  // Fill in a color/intensity histogram of the input
  filter_color_image( input, output, *histogram, options );

  if( !use_external_hist )
  {
    delete histogram;
  }
}

} // end anonoymous namespace

// ----------------------------------------------------------------------------
/// Private implementation class
class color_commonality_filter::priv
{
public:

  priv()
    : color_resolution( 512 )
      , color_resolution_per_chan( 8 )
      , intensity_resolution( 16 )
      , smooth_image( false )
  {
  }

  ~priv()
  {
  }

  template < typename pix_t >
  kwiver::vital::image_container_sptr
  compute_commonality( vil_image_view< pix_t >& input );

  color_commonality_filter_settings settings;

  unsigned color_resolution;
  unsigned color_resolution_per_chan;
  unsigned intensity_resolution;
  bool smooth_image;

  std::vector< unsigned > color_histogram;
  std::vector< unsigned > intensity_histogram;
};

template < typename pix_t >
kwiver::vital::image_container_sptr
color_commonality_filter::priv
::compute_commonality( vil_image_view< pix_t >& input )
{
  if( smooth_image )
  {
    vil_image_view< pix_t > smoothed;
    vil_gauss_filter_2d( input, smoothed, 0.5, 2 );
    input = smoothed;
  }
  if( input.nplanes() == 1 )
  {
    vil_image_view< pix_t > output;
    settings.resolution_per_channel = intensity_resolution;
    settings.histogram = &intensity_histogram;
    perform_filtering( input, output, settings );
    return std::make_shared< vxl::image_container >( output );
  }
  else
  {
    vil_image_view< pix_t > output;
    settings.resolution_per_channel = color_resolution_per_chan;
    settings.histogram = &color_histogram;
    perform_filtering( input, output, settings );
    return std::make_shared< vxl::image_container >( output );
  }
}

// ----------------------------------------------------------------------------
color_commonality_filter
::color_commonality_filter()
  : d( new priv() )
{
  attach_logger( "arrows.vxl.color_commonality_filter" );
}

// ----------------------------------------------------------------------------
color_commonality_filter
::~color_commonality_filter()
{
}

// ----------------------------------------------------------------------------
vital::config_block_sptr
color_commonality_filter
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "color_resolution_per_channel",
                     d->color_resolution_per_chan,
                     "Resolution of the utilized histogram (per channel) if the input "
                     "contains 3 channels." );
  config->set_value( "intensity_resolution", d->intensity_resolution,
                     "Resolution of the utilized histogram if the input "
                     "contains 1 channel." );
  config->set_value( "output_scale", d->settings.output_scale_factor,
                     "Scale the output image (typically, values start in the range [0,1]) "
                     "by this amount. Enter 0 for type-specific default." );
  config->set_value( "smooth_image",  d->smooth_image,
                     "Should we smooth the input image before filtering?" );
  config->set_value( "grid_image", d->settings.grid_image,
                     "Instead of calculating which colors are more common "
                     "in the entire image, should we do it for smaller evenly "
                     "spaced regions?" );
  config->set_value( "grid_resolution_height",
                     d->settings.grid_resolution_height,
                     "Divide the height of the image into x regions, if enabled." );
  config->set_value( "grid_resolution_width",
                     d->settings.grid_resolution_width,
                     "Divide the width of the image into x regions, if enabled." );

  return config;
}

// ----------------------------------------------------------------------------
void
color_commonality_filter
::set_configuration( vital::config_block_sptr in_config )
{
  // Starting with our generated vital::config_block to ensure that assumed
  // values
  // are present. An alternative is to check for key presence before performing
  // a
  // get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  // Settings for ccf
  d->color_resolution_per_chan =
    config->get_value< unsigned >( "color_resolution_per_channel" );
  d->intensity_resolution =
    config->get_value< unsigned >( "intensity_resolution" );
  d->smooth_image =
    config->get_value< bool >( "smooth_image" );
  d->settings.output_scale_factor =
    config->get_value< unsigned >( "output_scale" );
  d->settings.grid_image =
    config->get_value< bool >( "grid_image" );

  if( d->settings.grid_image )
  {
    d->settings.grid_resolution_width =
      config->get_value< unsigned >( "grid_resolution_width" );
    d->settings.grid_resolution_height =
      config->get_value< unsigned >( "grid_resolution_height" );
  }

  d->color_resolution = d->color_resolution_per_chan *
                        d->color_resolution_per_chan *
                        d->color_resolution_per_chan;

  d->color_histogram.resize( d->color_resolution, 0 );
  d->intensity_histogram.resize( d->intensity_resolution, 0 );
}

// ----------------------------------------------------------------------------
bool
color_commonality_filter
::check_configuration( vital::config_block_sptr config ) const
{
  if( !is_power_of_two( d->color_resolution_per_chan ) ||
      !is_power_of_two( d->intensity_resolution ) )
  {
    LOG_ERROR( logger(), "Specified resolutions must be a power of 2" );
  }
  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
color_commonality_filter
::filter( kwiver::vital::image_container_sptr image_data )
{
  // Perform Basic Validation
  if( !image_data )
  {
    return kwiver::vital::image_container_sptr();
  }

  if( image_data->depth() != 1 && image_data->depth() != 3 )
  {
    LOG_ERROR( logger(), "Invalid number of input planes!" );
    return kwiver::vital::image_container_sptr();
  }

  // Get input image
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );

  // Perform different actions based on input type
#define HANDLE_CASE( T )                                             \
  case T:                                                            \
  {                                                                  \
    typedef vil_pixel_format_type_of< T >::component_type pix_t;     \
    vil_image_view< pix_t > input = view;                            \
    return d->compute_commonality( input );                          \
  }                                                                  \

  switch( view->pixel_format() )
  {
    HANDLE_CASE( VIL_PIXEL_FORMAT_BOOL );
    HANDLE_CASE( VIL_PIXEL_FORMAT_BYTE );
    HANDLE_CASE( VIL_PIXEL_FORMAT_SBYTE );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_16 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_32 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_64 );
#undef HANDLE_CASE

    default:
      LOG_ERROR( logger(), "Invalid type received" );
      return image_data;
  }

  // Code not reached, prevent warning
  return kwiver::vital::image_container_sptr();
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
