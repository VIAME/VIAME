/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "normalize_image_percentile.h"

#include <vital/types/image.h>
#include <vital/types/image_container.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace viame {

// -----------------------------------------------------------------------------------------------
/// @brief Storage class for private member variables
class normalize_image_percentile::priv
{
public:
  priv()
    : m_lower_percentile( 1.0 )
    , m_upper_percentile( 100.0 )
    , m_output_format( "8-bit" )
  {}

  ~priv() {}

  double m_lower_percentile;
  double m_upper_percentile;
  std::string m_output_format;
};

// =================================================================================================

normalize_image_percentile::
normalize_image_percentile()
  : d( new priv )
{
  attach_logger( "viame.core.normalize_image_percentile" );
}


normalize_image_percentile::
~normalize_image_percentile()
{}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
normalize_image_percentile::
get_configuration() const
{
  kwiver::vital::config_block_sptr config =
    kwiver::vital::algo::image_filter::get_configuration();

  config->set_value( "lower_percentile", d->m_lower_percentile,
    "Lower percentile for minimum value calculation (0.0 to 100.0). "
    "Default is 1.0 to exclude outliers." );

  config->set_value( "upper_percentile", d->m_upper_percentile,
    "Upper percentile for maximum value calculation (0.0 to 100.0). "
    "Default is 100.0 (maximum value)." );

  config->set_value( "output_format", d->m_output_format,
    "Output format: '8-bit' for 8-bit unsigned output, 'native' for same "
    "type as input with values stretched to full range. Default is '8-bit'." );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
normalize_image_percentile::
set_configuration( kwiver::vital::config_block_sptr config_in )
{
  kwiver::vital::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->m_lower_percentile = config->get_value< double >( "lower_percentile" );
  d->m_upper_percentile = config->get_value< double >( "upper_percentile" );
  d->m_output_format = config->get_value< std::string >( "output_format" );
}


// -------------------------------------------------------------------------------------------------
bool
normalize_image_percentile::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  double lower = config->get_value< double >( "lower_percentile" );
  double upper = config->get_value< double >( "upper_percentile" );
  std::string output_format = config->get_value< std::string >( "output_format" );

  if( lower < 0.0 || lower > 100.0 )
  {
    LOG_ERROR( logger(), "lower_percentile must be between 0.0 and 100.0" );
    return false;
  }

  if( upper < 0.0 || upper > 100.0 )
  {
    LOG_ERROR( logger(), "upper_percentile must be between 0.0 and 100.0" );
    return false;
  }

  if( lower >= upper )
  {
    LOG_ERROR( logger(), "lower_percentile must be less than upper_percentile" );
    return false;
  }

  if( output_format != "8-bit" && output_format != "native" )
  {
    LOG_ERROR( logger(), "output_format must be '8-bit' or 'native'" );
    return false;
  }

  return true;
}


// -------------------------------------------------------------------------------------------------
namespace {

/// Helper function to calculate percentile value from a sorted vector
template< typename T >
double calculate_percentile( const std::vector< T >& sorted_values, double percentile )
{
  if( sorted_values.empty() )
  {
    return 0.0;
  }

  double index = ( percentile / 100.0 ) * ( sorted_values.size() - 1 );
  size_t lower_idx = static_cast< size_t >( std::floor( index ) );
  size_t upper_idx = static_cast< size_t >( std::ceil( index ) );

  if( lower_idx == upper_idx || upper_idx >= sorted_values.size() )
  {
    return static_cast< double >( sorted_values[ lower_idx ] );
  }

  // Linear interpolation between the two nearest values
  double fraction = index - lower_idx;
  return static_cast< double >( sorted_values[ lower_idx ] ) * ( 1.0 - fraction ) +
         static_cast< double >( sorted_values[ upper_idx ] ) * fraction;
}

/// Extract all pixel values from an image into a vector (for any pixel type)
template< typename T >
std::vector< T > extract_pixel_values( const kwiver::vital::image& img )
{
  std::vector< T > values;
  values.reserve( img.width() * img.height() * img.depth() );

  for( size_t p = 0; p < img.depth(); ++p )
  {
    for( size_t j = 0; j < img.height(); ++j )
    {
      for( size_t i = 0; i < img.width(); ++i )
      {
        values.push_back( img.at< T >( i, j, p ) );
      }
    }
  }

  return values;
}

/// Get input pixel value as double
template< typename T >
double get_pixel_value( const kwiver::vital::image& img, size_t i, size_t j, size_t p )
{
  return static_cast< double >( img.at< T >( i, j, p ) );
}

/// Set output pixel value from normalized [0,1] value
template< typename T >
void set_pixel_value( kwiver::vital::image& img, size_t i, size_t j, size_t p,
                      double normalized, double max_val, double min_val = 0.0 )
{
  double scaled = normalized * ( max_val - min_val ) + min_val;

  // Clip to valid range
  if( scaled < min_val )
  {
    scaled = min_val;
  }
  else if( scaled > max_val )
  {
    scaled = max_val;
  }

  img.at< T >( i, j, p ) = static_cast< T >( scaled );
}

} // anonymous namespace


// -------------------------------------------------------------------------------------------------
kwiver::vital::image_container_sptr
normalize_image_percentile::
filter( kwiver::vital::image_container_sptr image_data )
{
  if( !image_data )
  {
    LOG_WARN( logger(), "Received null image container" );
    return image_data;
  }

  kwiver::vital::image input_img = image_data->get_image();

  size_t width = input_img.width();
  size_t height = input_img.height();
  size_t depth = input_img.depth();

  if( width == 0 || height == 0 )
  {
    LOG_WARN( logger(), "Received empty image" );
    return image_data;
  }

  // Extract pixel values and calculate percentiles based on pixel type
  double p_low = 0.0;
  double p_high = 255.0;

  kwiver::vital::image_pixel_traits input_traits = input_img.pixel_traits();

  if( input_traits.type == kwiver::vital::image_pixel_traits::UNSIGNED )
  {
    if( input_traits.num_bytes == 1 )
    {
      std::vector< uint8_t > values = extract_pixel_values< uint8_t >( input_img );
      std::sort( values.begin(), values.end() );
      p_low = calculate_percentile( values, d->m_lower_percentile );
      p_high = calculate_percentile( values, d->m_upper_percentile );
    }
    else if( input_traits.num_bytes == 2 )
    {
      std::vector< uint16_t > values = extract_pixel_values< uint16_t >( input_img );
      std::sort( values.begin(), values.end() );
      p_low = calculate_percentile( values, d->m_lower_percentile );
      p_high = calculate_percentile( values, d->m_upper_percentile );
    }
    else
    {
      LOG_WARN( logger(), "Unsupported unsigned pixel size: " << input_traits.num_bytes << " bytes" );
      return image_data;
    }
  }
  else if( input_traits.type == kwiver::vital::image_pixel_traits::SIGNED )
  {
    if( input_traits.num_bytes == 2 )
    {
      std::vector< int16_t > values = extract_pixel_values< int16_t >( input_img );
      std::sort( values.begin(), values.end() );
      p_low = calculate_percentile( values, d->m_lower_percentile );
      p_high = calculate_percentile( values, d->m_upper_percentile );
    }
    else
    {
      LOG_WARN( logger(), "Unsupported signed pixel size: " << input_traits.num_bytes << " bytes" );
      return image_data;
    }
  }
  else if( input_traits.type == kwiver::vital::image_pixel_traits::FLOAT )
  {
    if( input_traits.num_bytes == 4 )
    {
      std::vector< float > values = extract_pixel_values< float >( input_img );
      std::sort( values.begin(), values.end() );
      p_low = calculate_percentile( values, d->m_lower_percentile );
      p_high = calculate_percentile( values, d->m_upper_percentile );
    }
    else if( input_traits.num_bytes == 8 )
    {
      std::vector< double > values = extract_pixel_values< double >( input_img );
      std::sort( values.begin(), values.end() );
      p_low = calculate_percentile( values, d->m_lower_percentile );
      p_high = calculate_percentile( values, d->m_upper_percentile );
    }
    else
    {
      LOG_WARN( logger(), "Unsupported float pixel size: " << input_traits.num_bytes << " bytes" );
      return image_data;
    }
  }
  else
  {
    LOG_WARN( logger(), "Unsupported pixel type" );
    return image_data;
  }

  // Avoid division by zero
  double range = p_high - p_low;
  if( range <= 0.0 )
  {
    LOG_WARN( logger(), "Image has no dynamic range (min == max)" );
    range = 1.0;
  }

  // Determine output pixel traits and range
  kwiver::vital::image_pixel_traits output_traits;
  double output_max = 255.0;
  double output_min = 0.0;

  if( d->m_output_format == "native" )
  {
    output_traits = input_traits;

    if( input_traits.type == kwiver::vital::image_pixel_traits::UNSIGNED )
    {
      if( input_traits.num_bytes == 1 )
      {
        output_max = 255.0;
      }
      else if( input_traits.num_bytes == 2 )
      {
        output_max = 65535.0;
      }
    }
    else if( input_traits.type == kwiver::vital::image_pixel_traits::SIGNED )
    {
      if( input_traits.num_bytes == 2 )
      {
        output_min = -32768.0;
        output_max = 32767.0;
      }
    }
    else if( input_traits.type == kwiver::vital::image_pixel_traits::FLOAT )
    {
      output_min = 0.0;
      output_max = 1.0;
    }
  }
  else
  {
    // 8-bit output
    output_traits = kwiver::vital::image_pixel_traits_of< uint8_t >();
    output_max = 255.0;
    output_min = 0.0;
  }

  // Create output image
  kwiver::vital::image output_img( width, height, depth, false, output_traits );

  // Process each pixel
  for( size_t p = 0; p < depth; ++p )
  {
    for( size_t j = 0; j < height; ++j )
    {
      for( size_t i = 0; i < width; ++i )
      {
        // Get input value as double
        double value = 0.0;

        if( input_traits.type == kwiver::vital::image_pixel_traits::UNSIGNED )
        {
          if( input_traits.num_bytes == 1 )
          {
            value = get_pixel_value< uint8_t >( input_img, i, j, p );
          }
          else if( input_traits.num_bytes == 2 )
          {
            value = get_pixel_value< uint16_t >( input_img, i, j, p );
          }
        }
        else if( input_traits.type == kwiver::vital::image_pixel_traits::SIGNED )
        {
          if( input_traits.num_bytes == 2 )
          {
            value = get_pixel_value< int16_t >( input_img, i, j, p );
          }
        }
        else if( input_traits.type == kwiver::vital::image_pixel_traits::FLOAT )
        {
          if( input_traits.num_bytes == 4 )
          {
            value = get_pixel_value< float >( input_img, i, j, p );
          }
          else if( input_traits.num_bytes == 8 )
          {
            value = get_pixel_value< double >( input_img, i, j, p );
          }
        }

        // Normalize to [0, 1]
        double normalized = ( value - p_low ) / range;

        // Clip to [0, 1]
        if( normalized < 0.0 )
        {
          normalized = 0.0;
        }
        else if( normalized > 1.0 )
        {
          normalized = 1.0;
        }

        // Set output value based on output format
        if( d->m_output_format == "native" )
        {
          if( output_traits.type == kwiver::vital::image_pixel_traits::UNSIGNED )
          {
            if( output_traits.num_bytes == 1 )
            {
              set_pixel_value< uint8_t >( output_img, i, j, p, normalized, output_max, output_min );
            }
            else if( output_traits.num_bytes == 2 )
            {
              set_pixel_value< uint16_t >( output_img, i, j, p, normalized, output_max, output_min );
            }
          }
          else if( output_traits.type == kwiver::vital::image_pixel_traits::SIGNED )
          {
            if( output_traits.num_bytes == 2 )
            {
              set_pixel_value< int16_t >( output_img, i, j, p, normalized, output_max, output_min );
            }
          }
          else if( output_traits.type == kwiver::vital::image_pixel_traits::FLOAT )
          {
            if( output_traits.num_bytes == 4 )
            {
              set_pixel_value< float >( output_img, i, j, p, normalized, output_max, output_min );
            }
            else if( output_traits.num_bytes == 8 )
            {
              set_pixel_value< double >( output_img, i, j, p, normalized, output_max, output_min );
            }
          }
        }
        else
        {
          // 8-bit output
          set_pixel_value< uint8_t >( output_img, i, j, p, normalized, 255.0, 0.0 );
        }
      }
    }
  }

  return std::make_shared< kwiver::vital::simple_image_container >( output_img );
}


} // end namespace viame
