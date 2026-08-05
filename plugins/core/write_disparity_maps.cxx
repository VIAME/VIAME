/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "write_disparity_maps.h"

#include <vital/algo/algorithm.txx>
#include <vital/types/image.h>
#include <vital/types/image_container.h>
#include <vital/exceptions/io.h>
#include <vital/vital_config.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace viame
{

namespace kv = kwiver::vital;

// =============================================================================
// Colormap lookup tables
// =============================================================================

namespace {

// Jet colormap (256 entries, RGB)
void apply_jet_colormap( double normalized, uint8_t& r, uint8_t& g, uint8_t& b )
{
  // Jet colormap: blue -> cyan -> yellow -> red
  double v = std::max( 0.0, std::min( 1.0, normalized ) );

  if( v < 0.125 )
  {
    r = 0;
    g = 0;
    b = static_cast<uint8_t>( 128 + v * 8 * 127 );
  }
  else if( v < 0.375 )
  {
    r = 0;
    g = static_cast<uint8_t>( ( v - 0.125 ) * 4 * 255 );
    b = 255;
  }
  else if( v < 0.625 )
  {
    r = static_cast<uint8_t>( ( v - 0.375 ) * 4 * 255 );
    g = 255;
    b = static_cast<uint8_t>( 255 - ( v - 0.375 ) * 4 * 255 );
  }
  else if( v < 0.875 )
  {
    r = 255;
    g = static_cast<uint8_t>( 255 - ( v - 0.625 ) * 4 * 255 );
    b = 0;
  }
  else
  {
    r = static_cast<uint8_t>( 255 - ( v - 0.875 ) * 8 * 127 );
    g = 0;
    b = 0;
  }
}

// Inferno colormap (approximation)
void apply_inferno_colormap( double normalized, uint8_t& r, uint8_t& g, uint8_t& b )
{
  double v = std::max( 0.0, std::min( 1.0, normalized ) );

  // Simplified inferno: black -> purple -> orange -> yellow
  if( v < 0.25 )
  {
    double t = v * 4;
    r = static_cast<uint8_t>( t * 80 );
    g = static_cast<uint8_t>( t * 18 );
    b = static_cast<uint8_t>( t * 120 );
  }
  else if( v < 0.5 )
  {
    double t = ( v - 0.25 ) * 4;
    r = static_cast<uint8_t>( 80 + t * 120 );
    g = static_cast<uint8_t>( 18 + t * 32 );
    b = static_cast<uint8_t>( 120 + t * 20 );
  }
  else if( v < 0.75 )
  {
    double t = ( v - 0.5 ) * 4;
    r = static_cast<uint8_t>( 200 + t * 52 );
    g = static_cast<uint8_t>( 50 + t * 100 );
    b = static_cast<uint8_t>( 140 - t * 100 );
  }
  else
  {
    double t = ( v - 0.75 ) * 4;
    r = static_cast<uint8_t>( 252 + t * 3 );
    g = static_cast<uint8_t>( 150 + t * 105 );
    b = static_cast<uint8_t>( 40 + t * 100 );
  }
}

// Viridis colormap (approximation)
void apply_viridis_colormap( double normalized, uint8_t& r, uint8_t& g, uint8_t& b )
{
  double v = std::max( 0.0, std::min( 1.0, normalized ) );

  // Simplified viridis: purple -> blue -> green -> yellow
  if( v < 0.25 )
  {
    double t = v * 4;
    r = static_cast<uint8_t>( 68 + t * ( 59 - 68 ) );
    g = static_cast<uint8_t>( 1 + t * ( 82 - 1 ) );
    b = static_cast<uint8_t>( 84 + t * ( 139 - 84 ) );
  }
  else if( v < 0.5 )
  {
    double t = ( v - 0.25 ) * 4;
    r = static_cast<uint8_t>( 59 + t * ( 33 - 59 ) );
    g = static_cast<uint8_t>( 82 + t * ( 145 - 82 ) );
    b = static_cast<uint8_t>( 139 + t * ( 140 - 139 ) );
  }
  else if( v < 0.75 )
  {
    double t = ( v - 0.5 ) * 4;
    r = static_cast<uint8_t>( 33 + t * ( 94 - 33 ) );
    g = static_cast<uint8_t>( 145 + t * ( 201 - 145 ) );
    b = static_cast<uint8_t>( 140 + t * ( 98 - 140 ) );
  }
  else
  {
    double t = ( v - 0.75 ) * 4;
    r = static_cast<uint8_t>( 94 + t * ( 253 - 94 ) );
    g = static_cast<uint8_t>( 201 + t * ( 231 - 201 ) );
    b = static_cast<uint8_t>( 98 + t * ( 37 - 98 ) );
  }
}

// Grayscale
void apply_grayscale( double normalized, uint8_t& r, uint8_t& g, uint8_t& b )
{
  uint8_t v = static_cast<uint8_t>(
    std::max( 0.0, std::min( 255.0, normalized * 255.0 ) ) );
  r = g = b = v;
}

// Apply a colormap by name
void apply_colormap( const std::string& colormap, double normalized,
                     uint8_t& r, uint8_t& g, uint8_t& b )
{
  if( colormap == "jet" )
  {
    apply_jet_colormap( normalized, r, g, b );
  }
  else if( colormap == "inferno" )
  {
    apply_inferno_colormap( normalized, r, g, b );
  }
  else if( colormap == "viridis" )
  {
    apply_viridis_colormap( normalized, r, g, b );
  }
  else // grayscale
  {
    apply_grayscale( normalized, r, g, b );
  }
}

} // anonymous namespace

// =============================================================================
// write_disparity_maps implementation
// =============================================================================

// -----------------------------------------------------------------------------
void
write_disparity_maps
::initialize()
{
  // Initialize invalid color from defaults
  m_invalid_color_r = 0;
  m_invalid_color_g = 0;
  m_invalid_color_b = 0;
}

// -----------------------------------------------------------------------------
void
write_disparity_maps
::set_configuration_internal( kv::config_block_sptr config )
{
  // Parse invalid_color string into RGB components
  int r = 0, g = 0, b = 0;
  if( sscanf( c_invalid_color.c_str(), "%d,%d,%d", &r, &g, &b ) == 3 )
  {
    m_invalid_color_r = static_cast<uint8_t>( std::max( 0, std::min( 255, r ) ) );
    m_invalid_color_g = static_cast<uint8_t>( std::max( 0, std::min( 255, g ) ) );
    m_invalid_color_b = static_cast<uint8_t>( std::max( 0, std::min( 255, b ) ) );
  }

  // Nested algorithm
  kv::set_nested_algo_configuration<kv::algo::image_io>(
    "image_writer", config, m_image_writer );
}

// -----------------------------------------------------------------------------
bool
write_disparity_maps
::check_configuration( kv::config_block_sptr config ) const
{
  std::string colormap = config->get_value< std::string >( "colormap", "jet" );
  if( colormap != "jet" && colormap != "inferno" &&
      colormap != "viridis" && colormap != "grayscale" )
  {
    return false;
  }

  if( !kv::check_nested_algo_configuration<kv::algo::image_io>(
        "image_writer", config ) )
  {
    return false;
  }

  return true;
}

// -----------------------------------------------------------------------------
kv::image_container_sptr
write_disparity_maps
::load_( std::string const& filename ) const
{
  throw kv::file_not_read_exception( filename,
    "write_disparity_maps does not support loading images" );
}

// -----------------------------------------------------------------------------
void
write_disparity_maps
::save_( std::string const& filename, kv::image_container_sptr data ) const
{
  if( !data )
  {
    throw kv::file_write_exception( filename, "Null image container" );
  }

  if( !m_image_writer )
  {
    throw kv::file_write_exception( filename,
      "No image_writer algorithm configured" );
  }

  const auto& img = data->get_image();
  const size_t width = img.width();
  const size_t height = img.height();

  if( width == 0 || height == 0 )
  {
    throw kv::file_write_exception( filename, "Empty image" );
  }

  // Determine disparity range
  double min_disp = c_min_disparity;
  double max_disp = c_max_disparity;

  // Lambda to get disparity value at a pixel
  const char* img_data = reinterpret_cast<const char*>( img.first_pixel() );

  auto get_disparity = [&]( size_t x, size_t y ) -> double
  {
    const char* pixel_ptr = img_data + y * img.h_step() + x * img.w_step();

    if( img.pixel_traits().type == kv::image_pixel_traits::UNSIGNED &&
        img.pixel_traits().num_bytes == 2 )
    {
      const uint16_t* ptr = reinterpret_cast<const uint16_t*>( pixel_ptr );
      return static_cast<double>( *ptr ) / c_disparity_scale;
    }
    else if( img.pixel_traits().type == kv::image_pixel_traits::FLOAT &&
             img.pixel_traits().num_bytes == 4 )
    {
      const float* ptr = reinterpret_cast<const float*>( pixel_ptr );
      return static_cast<double>( *ptr );
    }
    else if( img.pixel_traits().type == kv::image_pixel_traits::UNSIGNED &&
             img.pixel_traits().num_bytes == 1 )
    {
      const uint8_t* ptr = reinterpret_cast<const uint8_t*>( pixel_ptr );
      return static_cast<double>( *ptr );
    }
    return 0.0;
  };

  // Compute auto range if needed
  if( c_auto_range )
  {
    min_disp = std::numeric_limits<double>::max();
    max_disp = std::numeric_limits<double>::lowest();

    for( size_t y = 0; y < height; ++y )
    {
      for( size_t x = 0; x < width; ++x )
      {
        double disp = get_disparity( x, y );
        if( disp > 0 && std::isfinite( disp ) )
        {
          min_disp = std::min( min_disp, disp );
          max_disp = std::max( max_disp, disp );
        }
      }
    }

    // Handle case where no valid disparities found
    if( min_disp > max_disp )
    {
      min_disp = 0.0;
      max_disp = 1.0;
    }
  }

  double range = max_disp - min_disp;
  if( range <= 0.0 )
  {
    range = 1.0;
  }

  // Create output RGB image (interleaved)
  kv::image_pixel_traits output_traits( kv::image_pixel_traits::UNSIGNED, 1 );
  kv::image output_img( width, height, 3, true, output_traits );

  char* out_data = reinterpret_cast<char*>( output_img.first_pixel() );

  for( size_t y = 0; y < height; ++y )
  {
    for( size_t x = 0; x < width; ++x )
    {
      double disp = get_disparity( x, y );

      uint8_t r, g, b;

      if( disp <= 0 || !std::isfinite( disp ) )
      {
        // Invalid disparity
        r = m_invalid_color_r;
        g = m_invalid_color_g;
        b = m_invalid_color_b;
      }
      else
      {
        // Normalize to [0, 1]
        double normalized = ( disp - min_disp ) / range;
        apply_colormap( c_colormap, normalized, r, g, b );
      }

      // Write RGB values
      char* out_ptr = out_data + y * output_img.h_step() + x * output_img.w_step();
      out_ptr[0] = static_cast<char>( r );
      out_ptr[output_img.d_step()] = static_cast<char>( g );
      out_ptr[2 * output_img.d_step()] = static_cast<char>( b );
    }
  }

  // Create output container and write using internal writer
  auto output_container = std::make_shared< kv::simple_image_container >( output_img );
  m_image_writer->save( filename, output_container );
}

} // namespace viame
