/*ckwg +5
 * Copyright 2019 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "convert_image.h"

#include <arrows/vxl/image_container.h>

#include <vil/vil_image_view.h>
#include <vil/vil_convert.h>
#include <vil/vil_plane.h>
#include <vil/vil_math.h>

#include <limits>
#include <cstdlib>
#include <type_traits>


namespace kwiver {
namespace arrows {
namespace vxl {

namespace
{

// Convert a floating point image to an intergral type by multiplying
// it by a scaling factor in addition to thresholding it in one operation.
// Performs rounding.
template< typename InType, typename OutType >
void scale_image( const vil_image_view<InType>& src,
                  vil_image_view<OutType>& dst,
                  const double& dp_scale )
{
  // Resize, cast and copy
  unsigned ni = src.ni(), nj = src.nj(), np = src.nplanes();
  dst.set_size( ni, nj, np );

  std::ptrdiff_t sistep=src.istep(), sjstep=src.jstep(), spstep=src.planestep();
  std::ptrdiff_t distep=dst.istep(), djstep=dst.jstep(), dpstep=dst.planestep();

  const OutType max_output_value = std::numeric_limits<OutType>::max();

  const InType max_input_value = static_cast< InType >(
    static_cast< double >( max_output_value ) / dp_scale );

  const InType scale = static_cast< InType >( dp_scale );

  const InType* splane = src.top_left_ptr();
  OutType* dplane = dst.top_left_ptr();
  for( unsigned p=0;p<np;++p,splane += spstep,dplane += dpstep )
  {
    const InType* srow = splane;
    OutType* drow = dplane;
    for( unsigned j=0;j<nj;++j,srow += sjstep,drow += djstep )
    {
      const InType* spixel = srow;
      OutType* dpixel = drow;
      for( unsigned i=0;i<ni;++i,spixel+=sistep,dpixel+=distep )
      {
        if( *spixel <= max_input_value )
        {
          *dpixel = static_cast<OutType>( *spixel * scale + 0.5 );
        }
        else
        {
          *dpixel = max_output_value;
        }
      }
    }
  }
}

template< typename Type >
void combine_channels( const vil_image_view<Type>& src,
                       vil_image_view<Type>& dst )
{
  if( src.nplanes() == 3 )
  {
    vil_convert_planes_to_grey( src, dst );
  }
  else
  {
    vil_math_mean_over_planes( src, dst );
  }
}


template< typename Type >
void random_grey_conversion( const vil_image_view<Type>& src,
                             vil_image_view<Type>& dst,
                             const double random_factor )
{
  if( static_cast<double>( rand() ) / RAND_MAX < random_factor )
  {
    vil_image_view<Type> compressed;
    combine_channels( src, compressed );

    dst.set_size( src.ni(), src.nj(), src.nplanes() );

    for( unsigned p = 0; p < src.nplanes(); ++p )
    {
      vil_image_view<Type> output_plane = vil_plane( dst, p );
      vil_copy_reformat( compressed, output_plane );
    }
  }
  else
  {
    dst = src;
  }
}

// Calculate the values of our image percentiles from x sampling points
template <typename PixType>
void sample_and_sort_image( const vil_image_view< PixType >& src,
                            std::vector< PixType >& dst,
                            unsigned int sampling_points,
                            bool remove_extremes )
{
  if( src.ni() * src.nj() < sampling_points )
  {
    sampling_points = src.ni() * src.nj();
  }

  if( sampling_points == 0 )
  {
    return;
  }

  const unsigned scanning_area = src.size();
  const unsigned ni = src.ni();
  const unsigned nj = src.nj();
  const unsigned np = src.nplanes();
  const unsigned pixel_step = scanning_area / sampling_points;

  dst.resize( sampling_points * np );

  unsigned position = 0;

  for( unsigned p = 0; p < np; p++ )
  {
    for( unsigned s = 0; s < sampling_points; s++, position += pixel_step )
    {
      unsigned i = position % ni;
      unsigned j = ( position / ni ) % nj;
      dst[ np*s + p ] = src( i, j, p );
    }
  }

  std::sort( dst.begin(), dst.end() );

  if( remove_extremes &&
      dst[ dst.size()-1 ] == std::numeric_limits< PixType >::max() )
  {
    for( unsigned i = 1; i < dst.size(); ++i )
    {
      if( dst[dst.size() - i] != std::numeric_limits< PixType >::max() )
      {
        dst.erase( dst.end() - i, dst.end() );
        break;
      }
    }
  }

  if( remove_extremes &&
      dst[0] == static_cast< PixType >( 0 ) )
  {
    for( unsigned i = 1; i < dst.size(); ++i )
    {
      if( dst[i] != 0 )
      {
        dst.erase( dst.begin(), dst.begin() + i );
        break;
      }
    }
  }
}

template <typename PixType>
void get_image_percentiles( const vil_image_view< PixType >& src,
                            const std::vector< double >& percentiles,
                            std::vector< PixType >& dst,
                            unsigned sampling_points,
                            bool remove_extremes )
{
  std::vector< PixType > sorted_samples;
  sample_and_sort_image( src, sorted_samples, sampling_points, remove_extremes );
  dst.resize( percentiles.size() );

  double sampling_points_m1 = static_cast<double>(sorted_samples.size()-1);

  for( unsigned i = 0; i < percentiles.size(); i++ )
  {
    unsigned ind = static_cast<unsigned>( sampling_points_m1 * percentiles[i] + 0.5 );
    dst[i] = sorted_samples[ind];
  }
}

template<typename PixType>
void percentile_threshold_above( const vil_image_view< PixType >& src,
                                 const std::vector< double >& percentiles,
                                 vil_image_view< bool >& dst,
                                 unsigned sampling_points )
{
  // Calculate thresholds
  std::vector< PixType > thresholds;
  get_image_percentiles( src, percentiles, thresholds, sampling_points );
  dst.set_size( src.ni(), src.nj(), percentiles.size() );

  // Perform thresholding
  for( unsigned i = 0; i < thresholds.size(); i++ )
  {
    vil_image_view< bool > output_plane = vil_plane( dst, i );
    vil_threshold_above( src, output_plane, thresholds[i] );
  }
}

template <typename InputType, typename OutputType>
void percentile_scale_image( const vil_image_view< InputType >& src,
                             vil_image_view< OutputType >& dst,
                             double lower, double upper,
                             unsigned sampling_points,
                             bool ignore_extremes = true )
{
  std::vector< double > percentiles( 2, 0.0 );
  percentiles[0] = lower;
  percentiles[1] = upper;

  std::vector< InputType > per_values;

  get_image_percentiles( src, percentiles, per_values, sampling_points, ignore_extremes );

  OutputType max_val = std::numeric_limits< OutputType >::max();

  InputType lower_bound = per_values[0];
  InputType upper_bound = per_values[1];

  double scale;

  if( per_values[1] - per_values[0] > 0 )
  {
    scale = ( static_cast< double >( max_val ) + 0.5 ) / ( per_values[1] - per_values[0] );
  }
  else
  {
    scale = static_cast< double >( max_val ) / std::numeric_limits< InputType >::max();
  }

  unsigned ni = src.ni(), nj = src.nj(), np = src.nplanes();
  dst.set_size( ni, nj, np );
  std::ptrdiff_t istep = src.istep(), jstep = src.jstep(), pstep = src.planestep();

  const InputType* plane = src.top_left_ptr();
  for( unsigned p=0; p<np; ++p, plane += pstep )
  {
    const InputType* row = plane;
    for( unsigned j=0; j<nj; ++j, row += jstep )
    {
      const InputType* pixel = row;
      for( unsigned i=0; i<ni; ++i, pixel+=istep )
      {
        if( *pixel < lower_bound )
        {
          dst( i, j, p ) = 0;
        }
        else if( *pixel > upper_bound )
        {
          dst( i, j, p ) = std::numeric_limits< OutputType >::max();
        }
        else
        {
          dst( i, j, p ) = static_cast< OutputType >( ( *pixel - lower_bound ) * scale );
        }
      }
    }
  }
}

template<typename PixType>
void force_three_channels( vil_image_view< PixType >& image )
{
  if( image.nplanes() == 3 )
  {
    return;
  }
  else
  {
    const vil_image_view< PixType > input = image;

    image = vil_image_view< PixType >( input.ni(), input.nj(), 3 );

    vil_image_view< PixType > plane1 = vil_plane( image, 0 );
    vil_image_view< PixType > plane2 = vil_plane( image, 1 );
    vil_image_view< PixType > plane3 = vil_plane( image, 2 );

    vil_copy_reformat( vil_plane( input, 0 ), plane1 );
    vil_copy_reformat( vil_plane( input, ( input.nplanes() > 1 ? 1 : 0 ) ), plane2 );
    vil_copy_reformat( vil_plane( input, ( input.nplanes() > 2 ? 2 : 0 ) ), plane3 );
  }
}

} // end anonoymous namespace

// --------------------------------------------------------------------------------------
/// Private implementation class
class convert_image::priv
{
public:

  priv()
   : format( "byte" )
   , single_channel( false )
   , scale_factor( 0.0 )
   , random_greyscale( 0.0 )
   , percentile_norm( -1.0 )
   , force_three_channel( false )
  {
  }

  ~priv()
  {
  }

  std::string format;
  bool single_channel;
  double scale_factor;
  double random_greyscale;
  double percentile_norm;
  bool force_three_channel;
};

// --------------------------------------------------------------------------------------
convert_image
::convert_image()
: d( new priv() )
{
  attach_logger( "arrows.vxl.convert_image" );
}

convert_image
::~convert_image()
{
}

vital::config_block_sptr
convert_image
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "format", d->format,
    "Output type format: byte, sbyte, float, double, uint16, uint32, etc." );
  config->set_value( "single_channel", d->single_channel,
    "Convert input (presumably multi-channel) to contain a single channel, using "
    "either standard RGB to grayscale conversion weights, or averaging." );
  config->set_value( "scale_factor", d->scale_factor,
    "Optional input value scaling factor" );
  config->set_value( "random_greyscale", d->random_greyscale,
    "Convert input image to a 3-channel greyscale image randomly with this percentage "
    "between 0.0 and 1.0. This is used for machine learning augmentation." );
  config->set_value( "percentile_norm", d->percentile_norm,
    "If set, perform percentile normalization such that the output image's min and max "
    "value correspond to this value to 1.0 minus this value percntile in the input image." );
  config->set_value( "force_three_channel", d->force_three_channel,
    "If set, will force the output image to be a 3-channel instead of grayscale. If it "
    "is any other channel count, it will be truncated to 3." );

  return config;
}

void
convert_image
::set_configuration( vital::config_block_sptr in_config )
{
  // Starting with our generated vital::config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing a
  // get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  // Settings for conversion
  d->format = config->get_value< std::string >( "format" );
  d->single_channel = config->get_value< bool >( "single_channel" );
  d->scale_factor = config->get_value< double >( "scale_factor" );
  d->random_greyscale = config->get_value< double >( "random_greyscale" );
  d->percentile_norm = config->get_value< double >( "percentile_norm" );
  d->force_three_channel = config->get_value< bool >( "force_three_channel" );

  // Adjustment in case user specified 1% instead of 0.01
  if( d->percentile_norm >= 0.5 )
  {
    d->percentile_norm /= 100.;
  }
}

bool
convert_image
::check_configuration( vital::config_block_sptr config ) const
{
  return true;
}

// Perform stitch operation
kwiver::vital::image_container_sptr
convert_image
::filter( kwiver::vital::image_container_sptr image_data )
{
  // Perform Basic Validation
  if( !image_data )
  {
    return kwiver::vital::image_container_sptr();
  }

  // Get input image
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );

  // Perform different actions based on input type
#define HANDLE_OUTPUT_CASE(S,T)                                        \
  if( d->format == S )                                                 \
  {                                                                    \
    typedef vil_pixel_format_type_of<T >::component_type opix_t;       \
    vil_image_view< opix_t > output;                                   \
                                                                       \
    if( d->percentile_norm >= 0.0 )                                    \
    {                                                                  \
      percentile_scale_image( input, output,                           \
        d->percentile_norm, 1.0 - d->percentile_norm, 1e8 );           \
    }                                                                  \
    else if( d->scale_factor == 0.0 )                                  \
    {                                                                  \
      vil_convert_cast( input, output );                               \
    }                                                                  \
    else                                                               \
    {                                                                  \
      scale_image< ipix_t, opix_t >( input, output, d->scale_factor ); \
    }                                                                  \
                                                                       \
    if( d->force_three_channel )                                       \
    {                                                                  \
      force_three_channels( output );                                  \
    }                                                                  \
                                                                       \
    return std::make_shared< vxl::image_container >( output );         \
  }                                                                    \

#define HANDLE_INPUT_CASE(T)                                           \
  case T:                                                              \
    {                                                                  \
      typedef vil_pixel_format_type_of<T >::component_type ipix_t;     \
                                                                       \
      if( d->format == "disable" )                                     \
      {                                                                \
        return image_data;                                             \
      }                                                                \
                                                                       \
      vil_image_view< ipix_t > input;                                  \
                                                                       \
      if( d->random_greyscale > 0.0 )                                  \
      {                                                                \
        vil_image_view< ipix_t > tmp = view;                           \
        random_grey_conversion( tmp, input, d->random_greyscale );     \
      }                                                                \
      else if( d->single_channel )                                     \
      {                                                                \
        vil_image_view< ipix_t > tmp = view;                           \
        combine_channels( tmp, input );                                \
      }                                                                \
      else                                                             \
      {                                                                \
        input = view;                                                  \
      }                                                                \
      HANDLE_OUTPUT_CASE("bool", VIL_PIXEL_FORMAT_BOOL);               \
      HANDLE_OUTPUT_CASE("byte", VIL_PIXEL_FORMAT_BYTE);               \
      HANDLE_OUTPUT_CASE("sbyte", VIL_PIXEL_FORMAT_SBYTE);             \
      HANDLE_OUTPUT_CASE("uint16", VIL_PIXEL_FORMAT_UINT_16);          \
      HANDLE_OUTPUT_CASE("int16", VIL_PIXEL_FORMAT_INT_16);            \
      HANDLE_OUTPUT_CASE("uint32", VIL_PIXEL_FORMAT_UINT_32);          \
      HANDLE_OUTPUT_CASE("int32", VIL_PIXEL_FORMAT_INT_32);            \
      HANDLE_OUTPUT_CASE("uint64", VIL_PIXEL_FORMAT_UINT_64);          \
      HANDLE_OUTPUT_CASE("int64", VIL_PIXEL_FORMAT_INT_64);            \
      HANDLE_OUTPUT_CASE("float", VIL_PIXEL_FORMAT_FLOAT);             \
      HANDLE_OUTPUT_CASE("double", VIL_PIXEL_FORMAT_DOUBLE);           \
    }                                                                  \
    break;                                                             \

  switch( view->pixel_format() )
  {
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_BOOL);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_BYTE);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_SBYTE);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_UINT_16);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_INT_16);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_UINT_32);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_INT_32);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_UINT_64);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_INT_64);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_FLOAT);
    HANDLE_INPUT_CASE(VIL_PIXEL_FORMAT_DOUBLE);

  default:
    LOG_ERROR( logger(), "Invalid input format type received" );
    return kwiver::vital::image_container_sptr();
  }

#undef HANDLE_INPUT_CASE
#undef HANDLE_OUTPUT_CASE

  LOG_ERROR( logger(), "Invalid output format type received" );
  return kwiver::vital::image_container_sptr();
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
