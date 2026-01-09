/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "windowed_utils.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>
#include <exception>

namespace viame {

namespace kv = kwiver::vital;

// =============================================================================
window_settings
::window_settings()
  : mode( DISABLED )
  , scale( 1.0 )
  , chip_width( 1000 )
  , chip_height( 1000 )
  , chip_step_width( 500 )
  , chip_step_height( 500 )
  , chip_edge_filter( -1 )
  , chip_edge_max_prob( -1.0 )
  , chip_adaptive_thresh( 2000000 )
  , batch_size( 1 )
  , min_detection_dim( 1 )
  , original_to_chip_size( false )
  , black_pad( false )
{}

// -----------------------------------------------------------------------------
kv::config_block_sptr
window_settings
::config() const
{
  kv::config_block_sptr config = kv::config_block::empty_config();

  rescale_option_converter conv;
  config->set_value( "mode", conv.to_string( mode ),
    "Pre-processing resize option, can be: disabled, maintain_ar, scale, "
    "chip, chip_and_original, original_and_resized, or adaptive." );
  config->set_value( "scale", scale,
    "Image scaling factor used when mode is scale or chip." );
  config->set_value( "chip_height", chip_height,
    "When in chip mode, the chip height." );
  config->set_value( "chip_width", chip_width,
    "When in chip mode, the chip width." );
  config->set_value( "chip_step_height", chip_step_height,
    "When in chip mode, the chip step size between chips." );
  config->set_value( "chip_step_width", chip_step_width,
    "When in chip mode, the chip step size between chips." );
  config->set_value( "chip_edge_filter", chip_edge_filter,
    "If using chipping, filter out detections this pixel count near borders." );
  config->set_value( "chip_edge_max_prob", chip_edge_max_prob,
    "If using chipping, maximum type probability for edge detections" );
  config->set_value( "chip_adaptive_thresh", chip_adaptive_thresh,
    "If using adaptive selection, total pixel count at which we start to chip." );
  config->set_value( "batch_size", batch_size,
    "Optional processing batch size to send to the detector." );
  config->set_value( "min_detection_dim", min_detection_dim,
    "Minimum detection dimension in original image space." );
  config->set_value( "original_to_chip_size", original_to_chip_size,
    "Optionally enforce the input image is the specified chip size" );
  config->set_value( "black_pad", black_pad,
    "Black pad the edges of resized chips to ensure consistent dimensions" );

  return config;
}

// -----------------------------------------------------------------------------
void
window_settings
::set_config( kv::config_block_sptr config )
{
  rescale_option_converter conv;
  mode = conv.from_string( config->get_value< std::string >( "mode" ) );
  scale = config->get_value< double >( "scale" );
  chip_width = config->get_value< int >( "chip_width" );
  chip_height = config->get_value< int >( "chip_height" );
  chip_step_width = config->get_value< int >( "chip_step_width" );
  chip_step_height = config->get_value< int >( "chip_step_height" );
  chip_edge_filter = config->get_value< int >( "chip_edge_filter" );
  chip_edge_max_prob = config->get_value< double >( "chip_edge_max_prob" );
  chip_adaptive_thresh = config->get_value< int >( "chip_adaptive_thresh" );
  batch_size = config->get_value< int >( "batch_size" );
  min_detection_dim = config->get_value< int >( "min_detection_dim" );
  original_to_chip_size = config->get_value< bool >( "original_to_chip_size" );
  black_pad = config->get_value< bool >( "black_pad" );
}

// -----------------------------------------------------------------------------
kv::config_block_sptr
window_settings
::chip_config() const
{
  kv::config_block_sptr config = kv::config_block::empty_config();

  rescale_option_converter conv;
  config->set_value( "mode", conv.to_string( mode ),
    "Pre-processing resize option, can be: disabled, maintain_ar, scale, "
    "chip, chip_and_original, original_and_resized, or adaptive." );
  config->set_value( "scale", scale,
    "Image scaling factor used when mode is scale or chip." );
  config->set_value( "chip_height", chip_height,
    "When in chip mode, the chip height." );
  config->set_value( "chip_width", chip_width,
    "When in chip mode, the chip width." );
  config->set_value( "chip_step_height", chip_step_height,
    "When in chip mode, the chip step size between chips." );
  config->set_value( "chip_step_width", chip_step_width,
    "When in chip mode, the chip step size between chips." );
  config->set_value( "chip_adaptive_thresh", chip_adaptive_thresh,
    "If using adaptive selection, total pixel count at which we start to chip." );
  config->set_value( "original_to_chip_size", original_to_chip_size,
    "Optionally enforce the input image is not larger than the chip size" );
  config->set_value( "black_pad", black_pad,
    "Black pad the edges of resized chips to ensure consistent dimensions" );

  return config;
}

// -----------------------------------------------------------------------------
void
window_settings
::set_chip_config( kv::config_block_sptr config )
{
  rescale_option_converter conv;
  mode = conv.from_string( config->get_value< std::string >( "mode" ) );
  scale = config->get_value< double >( "scale" );
  chip_width = config->get_value< int >( "chip_width" );
  chip_height = config->get_value< int >( "chip_height" );
  chip_step_width = config->get_value< int >( "chip_step_width" );
  chip_step_height = config->get_value< int >( "chip_step_height" );
  chip_adaptive_thresh = config->get_value< int >( "chip_adaptive_thresh" );
  original_to_chip_size = config->get_value< bool >( "original_to_chip_size" );
  black_pad = config->get_value< bool >( "black_pad" );
}

// =============================================================================

windowed_region_prop
::windowed_region_prop( image_rect r, double s1 )
 : original_roi( r ), edge_filter( -1 ),
   right_border( false ), bottom_border( false ),
   scale1( s1 ), shiftx( 0 ), shifty( 0 ), scale2( 1.0 )
{}

windowed_region_prop
::windowed_region_prop( image_rect r, int ef, bool rb, bool bb,
  double s1, int sx, int sy, double s2 )
 : original_roi( r ), edge_filter( ef ),
   right_border( rb ), bottom_border( bb ),
   scale1( s1 ), shiftx( sx ), shifty( sy ), scale2( s2 )
{}

// =============================================================================
// Bilinear interpolation resize implementation
// =============================================================================

namespace {

// Helper to clamp value to range
template< typename T >
inline T clamp( T val, T min_val, T max_val )
{
  return std::max( min_val, std::min( val, max_val ) );
}

// Get pixel value with bounds checking
template< typename T >
inline T get_pixel_safe( const kv::image& img, int x, int y, int c )
{
  x = clamp( x, 0, static_cast< int >( img.width() ) - 1 );
  y = clamp( y, 0, static_cast< int >( img.height() ) - 1 );
  return img.at< T >( x, y, c );
}

// Bilinear interpolation for a single channel
template< typename T >
inline double bilinear_sample( const kv::image& img, double x, double y, int c )
{
  int x0 = static_cast< int >( std::floor( x ) );
  int y0 = static_cast< int >( std::floor( y ) );
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  double fx = x - x0;
  double fy = y - y0;

  double v00 = static_cast< double >( get_pixel_safe< T >( img, x0, y0, c ) );
  double v10 = static_cast< double >( get_pixel_safe< T >( img, x1, y0, c ) );
  double v01 = static_cast< double >( get_pixel_safe< T >( img, x0, y1, c ) );
  double v11 = static_cast< double >( get_pixel_safe< T >( img, x1, y1, c ) );

  double v0 = v00 * ( 1.0 - fx ) + v10 * fx;
  double v1 = v01 * ( 1.0 - fx ) + v11 * fx;

  return v0 * ( 1.0 - fy ) + v1 * fy;
}

// Resize implementation for specific pixel type
template< typename T >
kv::image resize_image_typed(
  const kv::image& src,
  size_t dst_width,
  size_t dst_height )
{
  const size_t src_width = src.width();
  const size_t src_height = src.height();
  const size_t depth = src.depth();

  kv::image_of< T > dst( dst_width, dst_height, depth );

  if( dst_width == 0 || dst_height == 0 )
  {
    return dst;
  }

  const double x_scale = static_cast< double >( src_width ) / dst_width;
  const double y_scale = static_cast< double >( src_height ) / dst_height;

  for( size_t y = 0; y < dst_height; ++y )
  {
    const double src_y = ( y + 0.5 ) * y_scale - 0.5;

    for( size_t x = 0; x < dst_width; ++x )
    {
      const double src_x = ( x + 0.5 ) * x_scale - 0.5;

      for( size_t c = 0; c < depth; ++c )
      {
        double val = bilinear_sample< T >( src, src_x, src_y, c );

        // Clamp to valid range for the type
        if( std::is_integral< T >::value )
        {
          val = std::round( val );
          val = clamp( val,
            static_cast< double >( std::numeric_limits< T >::min() ),
            static_cast< double >( std::numeric_limits< T >::max() ) );
        }

        dst( x, y, c ) = static_cast< T >( val );
      }
    }
  }

  return dst;
}

} // anonymous namespace

// -----------------------------------------------------------------------------
kv::image
resize_image_bilinear(
  const kv::image& src,
  size_t dst_width,
  size_t dst_height )
{
  if( src.width() == dst_width && src.height() == dst_height )
  {
    return src;
  }

  if( dst_width == 0 || dst_height == 0 )
  {
    return kv::image();
  }

  // Dispatch based on pixel type
  const auto& traits = src.pixel_traits();

  if( traits.type == kv::image_pixel_traits::UNSIGNED )
  {
    switch( traits.num_bytes )
    {
      case 1: return resize_image_typed< uint8_t >( src, dst_width, dst_height );
      case 2: return resize_image_typed< uint16_t >( src, dst_width, dst_height );
      case 4: return resize_image_typed< uint32_t >( src, dst_width, dst_height );
      case 8: return resize_image_typed< uint64_t >( src, dst_width, dst_height );
    }
  }
  else if( traits.type == kv::image_pixel_traits::SIGNED )
  {
    switch( traits.num_bytes )
    {
      case 1: return resize_image_typed< int8_t >( src, dst_width, dst_height );
      case 2: return resize_image_typed< int16_t >( src, dst_width, dst_height );
      case 4: return resize_image_typed< int32_t >( src, dst_width, dst_height );
      case 8: return resize_image_typed< int64_t >( src, dst_width, dst_height );
    }
  }
  else if( traits.type == kv::image_pixel_traits::FLOAT )
  {
    switch( traits.num_bytes )
    {
      case 4: return resize_image_typed< float >( src, dst_width, dst_height );
      case 8: return resize_image_typed< double >( src, dst_width, dst_height );
    }
  }

  // Fallback to uint8
  return resize_image_typed< uint8_t >( src, dst_width, dst_height );
}

// -----------------------------------------------------------------------------
kv::image
crop_image(
  const kv::image& src,
  const image_rect& roi )
{
  // Return a view into the source image (shallow copy)
  // This is efficient when the crop will be immediately resized,
  // as the resize operation creates its own deep copy
  return src.crop(
    static_cast< size_t >( roi.x ),
    static_cast< size_t >( roi.y ),
    static_cast< size_t >( roi.width ),
    static_cast< size_t >( roi.height ) );
}

// -----------------------------------------------------------------------------
kv::image
scale_image_maintaining_ar(
  const kv::image& src,
  int width,
  int height,
  bool pad,
  double& scale_out )
{
  scale_out = 1.0;

  const int src_height = static_cast< int >( src.height() );
  const int src_width = static_cast< int >( src.width() );

  if( src_height == height && src_width == width )
  {
    return src;
  }

  double original_height = static_cast< double >( src_height );
  double original_width = static_cast< double >( src_width );

  if( original_height > height )
  {
    scale_out = height / original_height;
  }
  if( original_width > width )
  {
    scale_out = std::min( scale_out, width / original_width );
  }

  size_t new_width = static_cast< size_t >( std::round( original_width * scale_out ) );
  size_t new_height = static_cast< size_t >( std::round( original_height * scale_out ) );

  kv::image resized = resize_image_bilinear( src, new_width, new_height );

  if( pad && ( resized.width() != static_cast< size_t >( width ) ||
               resized.height() != static_cast< size_t >( height ) ) )
  {
    // Create padded image
    const size_t depth = src.depth();
    const auto& traits = src.pixel_traits();

    kv::image padded( width, height, depth, false, traits );

    // Zero the memory
    std::memset( padded.first_pixel(), 0,
      padded.width() * padded.height() * padded.depth() * traits.num_bytes );

    // Copy resized image into top-left corner
    const size_t copy_width = std::min( resized.width(), static_cast< size_t >( width ) );
    const size_t copy_height = std::min( resized.height(), static_cast< size_t >( height ) );

    for( size_t y = 0; y < copy_height; ++y )
    {
      for( size_t x = 0; x < copy_width; ++x )
      {
        for( size_t c = 0; c < depth; ++c )
        {
          std::memcpy(
            reinterpret_cast< char* >( padded.first_pixel() ) +
              ( y * padded.w_step() + x * padded.d_step() + c * padded.h_step() ) * traits.num_bytes,
            reinterpret_cast< const char* >( resized.first_pixel() ) +
              ( y * resized.w_step() + x * resized.d_step() + c * resized.h_step() ) * traits.num_bytes,
            traits.num_bytes );
        }
      }
    }

    return padded;
  }

  return resized;
}

// -----------------------------------------------------------------------------
kv::image
format_image(
  const kv::image& src,
  rescale_option option,
  double scale_factor,
  int width,
  int height,
  bool pad,
  double& scale_out )
{
  scale_out = 1.0;

  if( option == MAINTAIN_AR )
  {
    return scale_image_maintaining_ar( src, width, height, pad, scale_out );
  }
  else if( option == CHIP || option == SCALE ||
           option == CHIP_AND_ORIGINAL )
  {
    if( scale_factor == 1.0 )
    {
      return src;
    }
    else
    {
      size_t new_width = static_cast< size_t >(
        std::round( src.width() * scale_factor ) );
      size_t new_height = static_cast< size_t >(
        std::round( src.height() * scale_factor ) );

      scale_out = scale_factor;
      return resize_image_bilinear( src, new_width, new_height );
    }
  }
  else
  {
    rescale_option_converter conv;
    throw std::runtime_error( "Invalid resize option: " + conv.to_string( option ) );
  }
}

// -----------------------------------------------------------------------------
kv::detected_object_set_sptr
rescale_detections(
  const kv::detected_object_set_sptr detections,
  const windowed_region_prop& region_info,
  double chip_edge_max_prob )
{
  // Apply first scale transformation
  if( region_info.scale1 != 1.0 )
  {
    detections->scale( region_info.scale1 );
  }

  // Apply shift transformation
  if( region_info.shiftx != 0 || region_info.shifty != 0 )
  {
    detections->shift( region_info.shiftx, region_info.shifty );
  }

  // Apply second scale transformation
  if( region_info.scale2 != 1.0 )
  {
    detections->scale( region_info.scale2 );
  }

  const int dist = region_info.edge_filter;

  // If no edge filtering, return detections as-is
  if( dist < 0 )
  {
    return detections;
  }

  // Filter detections near edges
  const image_rect& roi = region_info.original_roi;
  std::vector< kv::detected_object_sptr > filtered_dets;

  for( auto det : *detections )
  {
    if( !det )
    {
      continue;
    }

    // Check if detection is near an edge
    if( ( roi.x > 0 && det->bounding_box().min_x() < roi.x + dist ) ||
        ( roi.y > 0 && det->bounding_box().min_y() < roi.y + dist ) ||
        ( !region_info.right_border && det->bounding_box().max_x() > roi.x + roi.width - dist ) ||
        ( !region_info.bottom_border && det->bounding_box().max_y() > roi.y + roi.height - dist ) )
    {
      // If chip_edge_max_prob is disabled (<=0), skip edge detections
      if( chip_edge_max_prob <= 0.0 )
      {
        continue;
      }

      // Clamp detection confidence to chip_edge_max_prob
      if( det->confidence() > chip_edge_max_prob )
      {
        det->set_confidence( chip_edge_max_prob );
      }

      // Clamp detection type scores to chip_edge_max_prob
      if( det->type() )
      {
        auto dot = det->type();
        std::string top_class;
        dot->get_most_likely( top_class );
        double score = dot->score( top_class );

        if( score > chip_edge_max_prob )
        {
          double scale = chip_edge_max_prob / score;

          for( auto name : dot->class_names() )
          {
            dot->set_score( name, dot->score( name ) * scale );
          }
        }
      }
    }

    filtered_dets.push_back( det );
  }

  return kv::detected_object_set_sptr(
    new kv::detected_object_set( filtered_dets ) );
}

// -----------------------------------------------------------------------------
void
prepare_image_regions(
  const kv::image& image,
  const window_settings& settings,
  std::vector< kv::image >& regions_to_process,
  std::vector< windowed_region_prop >& region_properties )
{
  // Clear output vectors
  regions_to_process.clear();
  region_properties.clear();

  const int img_width = static_cast< int >( image.width() );
  const int img_height = static_cast< int >( image.height() );

  // Determine the processing mode
  rescale_option mode = settings.mode;

  if( mode == ADAPTIVE )
  {
    if( ( img_height * img_width ) >= settings.chip_adaptive_thresh )
    {
      mode = CHIP_AND_ORIGINAL;
    }
    else if( settings.original_to_chip_size )
    {
      mode = MAINTAIN_AR;
    }
    else
    {
      mode = DISABLED;
    }
  }

  // Resize image if enabled
  kv::image resized_image;
  double scale_factor = 1.0;

  if( mode != DISABLED )
  {
    resized_image = format_image( image,
      ( mode == ORIGINAL_AND_RESIZED ? SCALE : mode ),
      settings.scale, settings.chip_width, settings.chip_height,
      false, scale_factor );
  }
  else
  {
    resized_image = image;
  }

  image_rect original_dims( 0, 0, img_width, img_height );

  const int resized_width = static_cast< int >( resized_image.width() );
  const int resized_height = static_cast< int >( resized_image.height() );

  // Create regions based on mode
  if( mode == ORIGINAL_AND_RESIZED )
  {
    if( img_height <= settings.chip_height && img_width <= settings.chip_width )
    {
      regions_to_process.push_back( image );
      region_properties.push_back( windowed_region_prop( original_dims, 1.0 ) );
    }
    else
    {
      if( ( img_height * img_width ) >= settings.chip_adaptive_thresh )
      {
        regions_to_process.push_back( resized_image );
        region_properties.push_back( windowed_region_prop( original_dims, 1.0 / scale_factor ) );
      }

      double scaled_original_scale;
      kv::image scaled_original = scale_image_maintaining_ar( image,
        settings.chip_width, settings.chip_height, settings.black_pad,
        scaled_original_scale );

      regions_to_process.push_back( scaled_original );
      region_properties.push_back( windowed_region_prop( original_dims, 1.0 / scaled_original_scale ) );
    }
  }
  else if( mode != CHIP && mode != CHIP_AND_ORIGINAL )
  {
    regions_to_process.push_back( resized_image );
    region_properties.push_back( windowed_region_prop( original_dims, 1.0 / scale_factor ) );
  }
  else
  {
    // Chip up scaled image
    for( int li = 0;
         li < resized_width - settings.chip_width + settings.chip_step_width;
         li += settings.chip_step_width )
    {
      int ti = std::min( li + settings.chip_width, resized_width );

      for( int lj = 0;
           lj < resized_height - settings.chip_height + settings.chip_step_height;
           lj += settings.chip_step_height )
      {
        int tj = std::min( lj + settings.chip_height, resized_height );

        if( tj - lj < 0 || ti - li < 0 )
        {
          continue;
        }

        image_rect resized_roi( li, lj, ti - li, tj - lj );
        image_rect original_roi(
          static_cast< int >( li / scale_factor ),
          static_cast< int >( lj / scale_factor ),
          static_cast< int >( ( ti - li ) / scale_factor ),
          static_cast< int >( ( tj - lj ) / scale_factor ) );

        kv::image cropped_chip = crop_image( resized_image, resized_roi );

        double scaled_crop_scale;
        kv::image scaled_crop = scale_image_maintaining_ar(
          cropped_chip, settings.chip_width, settings.chip_height,
          settings.black_pad, scaled_crop_scale );

        regions_to_process.push_back( scaled_crop );

        region_properties.push_back(
          windowed_region_prop( original_roi,
            settings.chip_edge_filter,
            ( li + settings.chip_step_width ) >=
              ( resized_width - settings.chip_width + settings.chip_step_width ),
            ( lj + settings.chip_step_height ) >=
              ( resized_height - settings.chip_height + settings.chip_step_height ),
            1.0 / scaled_crop_scale,
            li, lj,
            1.0 / scale_factor ) );
      }
    }

    // Extract full sized image chip if enabled
    if( mode == CHIP_AND_ORIGINAL )
    {
      if( settings.original_to_chip_size )
      {
        double scaled_original_scale;
        kv::image scaled_original = scale_image_maintaining_ar( image,
          settings.chip_width, settings.chip_height, settings.black_pad,
          scaled_original_scale );

        regions_to_process.push_back( scaled_original );
        region_properties.push_back( windowed_region_prop( original_dims, 1.0 / scaled_original_scale ) );
      }
      else
      {
        regions_to_process.push_back( image );
        region_properties.push_back( windowed_region_prop( original_dims, 1.0 ) );
      }
    }
  }
}

// -----------------------------------------------------------------------------
void scale_detections(
  kv::detected_object_set_sptr& detections,
  const windowed_region_prop& region_info )
{
  // Apply inverse scale2 transformation (divide by scale2)
  if( region_info.scale2 != 1.0 )
  {
    detections->scale( 1.0 / region_info.scale2 );
  }

  // Apply inverse shift transformation (subtract shift)
  if( region_info.shiftx != 0 || region_info.shifty != 0 )
  {
    detections->shift( -region_info.shiftx, -region_info.shifty );
  }

  // Apply inverse scale1 transformation (divide by scale1)
  if( region_info.scale1 != 1.0 )
  {
    detections->scale( 1.0 / region_info.scale1 );
  }
}

// -----------------------------------------------------------------------------
kv::detected_object_set_sptr
scale_detections_to_region(
  const kv::detected_object_set_sptr detections,
  const windowed_region_prop& region_info )
{
  if( !detections || detections->empty() )
  {
    return std::make_shared< kv::detected_object_set >();
  }

  const kv::bounding_box_d roi_box(
    region_info.original_roi.x,
    region_info.original_roi.y,
    region_info.original_roi.x + region_info.original_roi.width,
    region_info.original_roi.y + region_info.original_roi.height );

  std::vector< kv::detected_object_sptr > region_dets;

  // Filter and transform detections that overlap with this region
  for( auto det : *detections )
  {
    if( !det )
    {
      continue;
    }

    // Check if detection overlaps with this region
    kv::bounding_box_d det_box = det->bounding_box();
    kv::bounding_box_d overlap = kv::intersection( roi_box, det_box );

    if( overlap.area() <= 0 )
    {
      continue;
    }

    // Clone the detection so we don't modify the original
    region_dets.push_back( det->clone() );
  }

  kv::detected_object_set_sptr output =
    std::make_shared< kv::detected_object_set >( region_dets );

  scale_detections( output, region_info );

  return output;
}

// -----------------------------------------------------------------------------
void
scale_detections_to_region_with_mapping(
  const kv::detected_object_set_sptr detections,
  const windowed_region_prop& region_info,
  std::vector< kv::detected_object_sptr >& original_detections,
  std::vector< kv::detected_object_sptr >& scaled_detections )
{
  original_detections.clear();
  scaled_detections.clear();

  if( !detections || detections->empty() )
  {
    return;
  }

  const kv::bounding_box_d roi_box(
    region_info.original_roi.x,
    region_info.original_roi.y,
    region_info.original_roi.x + region_info.original_roi.width,
    region_info.original_roi.y + region_info.original_roi.height );

  // Filter and transform detections that overlap with this region
  for( auto det : *detections )
  {
    if( !det )
    {
      continue;
    }

    // Check if detection overlaps with this region
    kv::bounding_box_d det_box = det->bounding_box();
    kv::bounding_box_d overlap = kv::intersection( roi_box, det_box );

    if( overlap.area() <= 0 )
    {
      continue;
    }

    // Clone the detection so we don't modify the original
    scaled_detections.push_back( det->clone() );
    original_detections.push_back( det );
  }

  auto scaled_set =
    std::make_shared< kv::detected_object_set >( scaled_detections );

  scale_detections( scaled_set, region_info );
}

// -----------------------------------------------------------------------------
void
separate_boundary_detections(
  const kv::detected_object_set_sptr detections,
  int region_width,
  int region_height,
  kv::detected_object_set_sptr& boundary_detections,
  kv::detected_object_set_sptr& interior_detections )
{
  boundary_detections = std::make_shared< kv::detected_object_set >();
  interior_detections = std::make_shared< kv::detected_object_set >();

  if( !detections )
  {
    return;
  }

  for( auto det : *detections )
  {
    if( !det )
    {
      continue;
    }

    kv::bounding_box_d bbox = det->bounding_box();

    // Check if detection touches any boundary
    bool touches_boundary =
      ( bbox.min_x() <= 0.0 ) ||
      ( bbox.min_y() <= 0.0 ) ||
      ( bbox.max_x() >= region_width - 1 ) ||
      ( bbox.max_y() >= region_height - 1 );

    if( touches_boundary )
    {
      boundary_detections->add( det );
    }
    else
    {
      interior_detections->add( det );
    }
  }
}

} // end namespace viame
