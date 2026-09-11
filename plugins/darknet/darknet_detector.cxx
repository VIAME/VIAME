/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "darknet_detector.h"

#include <viame/algorithm_framework/util/cpu_timer.h>
#include <viame/algorithm_framework/exceptions/io.h>
#include <viame/algorithm_framework/config/config_block_formatter.h>
#include <viame/core_types/detected_object_set_util.h>

// The chipping, the aspect-preserving fit and the crop, all of which this
// shared with `ocv_windowed` and none of which was ever OpenCV's arithmetic.
// `darknet_custom_resize` was a second copy of the first two and is gone.
#include "../core/windowed_utils.h"

#include <image_ops/color.h>
#include <image_ops/warp.h>

#include <kwiversys/SystemTools.hxx>

#include <string>
#include <sstream>
#include <fstream>
#include <exception>
#include <limits>
#include <vector>

#include "darknet/yolo_v2_class.hpp"

namespace viame {

namespace {

namespace kv = kwiver::vital;

// ----------------------------------------------------------------------------
/// The `image_t` darknet's own `cv::Mat` overload would have built.
///
/// `Detector::detect( cv::Mat )` is three steps and this is the same three:
/// resize to the network's input with `cv::resize`, which `image_ops::resize`
/// reproduces to the count for an 8-bit image; convert to **BGR**, which is
/// `RGB2BGR` for three channels, `GRAY2BGR` for one and `RGBA2BGR` for four,
/// so in a planar image it is the planes in reverse; and transpose into
/// planes of floats over [0, 1].
///
/// Going straight to `image_t` is also what lets the darknet fork be built
/// with `ENABLE_OPENCV=OFF`, since `detect( cv::Mat )` only exists when it is
/// on -- `detect( image_t )` is there either way.
///
/// The storage is the caller's: darknet's `detect` reads the buffer and does
/// not take it.
class darknet_image
{
public:
  darknet_image( kv::image const& source, int net_width, int net_height )
  {
    auto const width = static_cast< size_t >( net_width );
    auto const height = static_cast< size_t >( net_height );

    kv::image_of< uint8_t > bytes( source );

    auto const fitted =
      ( bytes.width() == width && bytes.height() == height )
        ? bytes
        : image_ops::resize( bytes, width, height );

    auto const depth = fitted.depth();

    m_data.resize( width * height * 3 );

    for( size_t plane = 0; plane < 3; ++plane )
    {
      // One channel is replicated; three or four are reversed, which is what
      // the conversion to BGR amounts to. A fourth channel is dropped, the
      // way `RGBA2BGR` drops it.
      auto const source_plane = ( depth == 1 ) ? 0 : ( 2 - plane );

      float* out = m_data.data() + plane * width * height;

      for( size_t row = 0; row < height; ++row )
      {
        for( size_t column = 0; column < width; ++column )
        {
          out[ row * width + column ] =
            static_cast< float >( fitted( column, row, source_plane ) ) /
            255.0f;
        }
      }
    }

    m_image.w = net_width;
    m_image.h = net_height;
    m_image.c = 3;
    m_image.data = m_data.data();
  }

  image_t& get() { return m_image; }

private:
  std::vector< float > m_data;
  image_t m_image{};
};

} // anonymous namespace

// =============================================================================

class darknet_detector::priv
{
public:
  priv( darknet_detector& )
    : m_thresh( 0.24 )
    , m_hier_thresh( 0.5 )
    , m_gpu_index( -1 )
    , m_resize_option( "disabled" )
    , m_scale( 1.0 )
    , m_chip_step( 100 )
    , m_nms_threshold( 0.4 )
    , m_gs_to_rgb( true )
    , m_chip_edge_filter( 0 )
    , m_chip_adaptive_thresh( 2000000 )
    , m_is_first( true )
    , m_names()
  {}

  ~priv() {}

  // Items from the config
  std::string m_net_config;
  std::string m_weight_file;
  std::string m_class_names;

  float m_thresh;
  float m_hier_thresh;
  int m_gpu_index;

  std::string m_resize_option;
  double m_scale;
  int m_chip_step;
  double m_nms_threshold;
  bool m_gs_to_rgb;
  int m_chip_edge_filter;
  int m_chip_adaptive_thresh;
  bool m_is_first;

  // Needed to operate the model
  std::vector< std::string > m_names;
  std::unique_ptr< Detector > m_net;

  // Helper functions
  struct region_info
  {
    explicit region_info( image_rect r, double s1 )
     : original_roi( r ), edge_filter( 0 ),
       scale1( s1 ), shiftx( 0 ), shifty( 0 ), scale2( 1.0 )
    {}

    explicit region_info( image_rect r, int ef,
      double s1, int sx, int sy, double s2 )
     : original_roi( r ), edge_filter( ef ),
       scale1( s1 ), shiftx( sx ), shifty( sy ), scale2( s2 )
    {}

    image_rect original_roi;
    int edge_filter;
    double scale1;
    int shiftx, shifty;
    double scale2;
  };

  std::vector< kwiver::vital::detected_object_set_sptr > process_images(
    const std::vector< kwiver::vital::image >& images );

  kwiver::vital::detected_object_set_sptr scale_detections(
    const kwiver::vital::detected_object_set_sptr detections,
    const region_info& roi );

  kwiver::vital::logger_handle_t m_logger;
};


// =============================================================================

void
darknet_detector
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.darknet.darknet_detector" );
  d->m_logger = logger();
}


darknet_detector
::~darknet_detector()
{}


// -----------------------------------------------------------------------------
void
darknet_detector
::set_configuration_internal( kwiver::vital::config_block_sptr config )
{
  // Copy config params from class members to priv
  d->m_net_config  = c_net_config;
  d->m_weight_file = c_weight_file;
  d->m_class_names = c_class_names;
  d->m_thresh      = c_thresh;
  d->m_hier_thresh = c_hier_thresh;
  d->m_gpu_index   = c_gpu_index;
  d->m_resize_option = c_resize_option;
  d->m_scale       = c_scale;
  d->m_chip_step   = c_chip_step;
  d->m_nms_threshold = c_nms_threshold;
  d->m_gs_to_rgb   = c_gs_to_rgb;
  d->m_chip_edge_filter = c_chip_edge_filter;
  d->m_chip_adaptive_thresh = c_chip_adaptive_thresh;

  // Open file and return 'list' of labels
  std::ifstream fin( d->m_class_names.c_str() );
  d->m_names.clear();
  if( !fin )
  {
    LOG_ERROR( logger(), "Unable to open labels file: " << d->m_class_names );
  }
  std::string line;
  while( std::getline( fin, line ) )
  {
    if( line.size() > 0 )
    {
      d->m_names.push_back( line );
    }
  }
  fin.close();

  d->m_net.reset( new Detector( d->m_net_config, d->m_weight_file, d->m_gpu_index ) );

  // This assumes that there are no other users of random number
  // generator in this application.
  srand( 2222222 );
}


// -----------------------------------------------------------------------------
bool
darknet_detector
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  std::string net_config = config->get_value< std::string >( "net_config" );
  std::string class_file = config->get_value< std::string >( "class_names" );

  bool success = true;

  if( net_config.empty() )
  {
    std::stringstream str;
    kwiver::vital::config_block_formatter fmt( config );
    fmt.print( str );
    LOG_ERROR( logger(), "Required net config file not specified. "
      "Configuration is as follows:\n" << str.str() );
    success = false;
  }
  else if( !kwiversys::SystemTools::FileExists( net_config ) )
  {
    LOG_ERROR( logger(), "net config file \"" << net_config << "\" not found." );
    success = false;
  }

  if( class_file.empty() )
  {
    std::stringstream str;
    kwiver::vital::config_block_formatter fmt( config );
    fmt.print( str );
    LOG_ERROR( logger(), "Required class name list file not specified, "
      "Configuration is as follows:\n" << str.str() );
    success = false;
  }
  else if( !kwiversys::SystemTools::FileExists( class_file ) )
  {
    LOG_ERROR( logger(), "class names file \"" << class_file << "\" not found." );
    success = false;
  }

  return success;
}


// -----------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
darknet_detector
::detect( kwiver::vital::image_container_sptr image_data ) const
{
  kwiver::vital::scoped_cpu_timer t( "Time to Detect Objects" );

  if( !image_data )
  {
    LOG_WARN( d->m_logger, "Input image is empty." );
    return std::make_shared< kwiver::vital::detected_object_set >();
  }

  const kwiver::vital::image source_image = image_data->get_image();

  const int image_width = static_cast< int >( source_image.width() );
  const int image_height = static_cast< int >( source_image.height() );

  if( image_width == 0 || image_height == 0 )
  {
    LOG_WARN( d->m_logger, "Input image is empty." );
    return std::make_shared< kwiver::vital::detected_object_set >();
  }
  else if( d->m_resize_option == "adaptive" )
  {
    if( ( image_height * image_width ) >= d->m_chip_adaptive_thresh )
    {
      d->m_resize_option = "chip_and_original";
    }
    else
    {
      d->m_resize_option = "maintain_ar";
    }
  }

  kwiver::vital::image resized_image;

  kwiver::vital::detected_object_set_sptr detections;

  // resizes image if enabled
  double scale_factor = 1.0;

  if( d->m_resize_option != "disabled" )
  {
    rescale_option_converter converter;

    resized_image = format_image( source_image,
      converter.from_string( d->m_resize_option ), d->m_scale,
      d->m_net->get_net_width(), d->m_net->get_net_height(),
      true, scale_factor );
  }
  else
  {
    resized_image = source_image;
  }

  // The conversion darknet's own `mat_to_image` would do anyway, since it
  // takes `GRAY2BGR` on a single channel whatever arrives. Kept because the
  // configuration key is still there and still means this.
  if( d->m_gs_to_rgb && resized_image.depth() == 1 )
  {
    resized_image = kwiver::vital::image( image_ops::gray_to_rgb(
      kwiver::vital::image_of< uint8_t >( resized_image ) ) );
  }

  // Run detector
  detections = std::make_shared< kwiver::vital::detected_object_set >();

  image_rect original_dims( 0, 0, image_width, image_height );

  std::vector< kwiver::vital::image > regions_to_process;
  std::vector< priv::region_info > region_properties;

  const int resized_width = static_cast< int >( resized_image.width() );
  const int resized_height = static_cast< int >( resized_image.height() );

  if( d->m_resize_option != "chip" && d->m_resize_option != "chip_and_original" )
  {
    regions_to_process.push_back( resized_image );

    region_properties.push_back(
      priv::region_info( original_dims, 1.0 / scale_factor ) );
  }
  else
  {
    // Chip up scaled image
    for( int li = 0;
         li < resized_width - d->m_net->get_net_width() + d->m_chip_step;
         li += d->m_chip_step )
    {
      int ti = std::min( li + d->m_net->get_net_width(), resized_width );

      for( int lj = 0;
           lj < resized_height - d->m_net->get_net_height() + d->m_chip_step;
           lj += d->m_chip_step )
      {
        int tj = std::min( lj + d->m_net->get_net_height(), resized_height );

        image_rect resized_roi( li, lj, ti-li, tj-lj );
        image_rect original_roi( li / scale_factor,
                                 lj / scale_factor,
                                 (ti-li) / scale_factor,
                                 (tj-lj) / scale_factor );

        kwiver::vital::image cropped_chip =
          crop_image( resized_image, resized_roi );

        double scaled_crop_scale = 1.0;

        kwiver::vital::image scaled_crop = scale_image_maintaining_ar(
          cropped_chip, d->m_net->get_net_width(), d->m_net->get_net_height(),
          true, scaled_crop_scale );

        regions_to_process.push_back( scaled_crop );

        region_properties.push_back(
          priv::region_info( original_roi,
            d->m_chip_edge_filter,
            1.0 / scaled_crop_scale,
            li, lj,
            1.0 / scale_factor ) );
      }
    }

    // Extract full sized image chip if enabled
    if( d->m_resize_option == "chip_and_original" )
    {
      double scaled_original_scale = 1.0;

      kwiver::vital::image scaled_original = scale_image_maintaining_ar(
        source_image, d->m_net->get_net_width(), d->m_net->get_net_height(),
        true, scaled_original_scale );

      if( d->m_gs_to_rgb && scaled_original.depth() == 1 )
      {
        scaled_original = kwiver::vital::image( image_ops::gray_to_rgb(
          kwiver::vital::image_of< uint8_t >( scaled_original ) ) );
      }

      regions_to_process.push_back( scaled_original );

      region_properties.push_back(
        priv::region_info( original_dims, 1.0 / scaled_original_scale ) );
    }
  }

  // Process all regions
  unsigned max_count = 1;

  for( unsigned i = 0; i < regions_to_process.size(); i+= max_count )
  {
    unsigned batch_size = std::min( max_count,
      static_cast< unsigned >( regions_to_process.size() ) - i );

    std::vector< kwiver::vital::image > imgs;

    for( unsigned j = 0; j < batch_size; j++ )
    {
      imgs.push_back( regions_to_process[ i + j ] );
    }

    std::vector< kwiver::vital::detected_object_set_sptr > out = d->process_images( imgs );

    for( unsigned j = 0; j < batch_size; j++ )
    {
      detections->add( d->scale_detections( out[ j ], region_properties[ i + j ] ) );
    }
  }

  return detections;
}


// -----------------------------------------------------------------------------
std::vector< kwiver::vital::detected_object_set_sptr >
darknet_detector::priv
::process_images( const std::vector< kwiver::vital::image >& images )
{
  std::vector< kwiver::vital::detected_object_set_sptr > output;

  for( unsigned i = 0; i < images.size(); i++ )
  {
    // `Detector::detect( cv::Mat )`, unrolled: build the `image_t` that
    // overload would have built, detect on it, and scale the boxes back from
    // the network's input to the region's own size the way `detect_resized`
    // does. The `cv::Mat` overload exists only when darknet is built with
    // OpenCV; this one is there either way.
    darknet_image prepared( images[i],
      m_net->get_net_width(), m_net->get_net_height() );

    auto darknet_output = m_net->detect( prepared.get(), m_thresh );

    const float width_ratio =
      static_cast< float >( images[i].width() ) /
      static_cast< float >( m_net->get_net_width() );
    const float height_ratio =
      static_cast< float >( images[i].height() ) /
      static_cast< float >( m_net->get_net_height() );

    for( auto& box : darknet_output )
    {
      box.x = static_cast< unsigned >( box.x * width_ratio );
      box.w = static_cast< unsigned >( box.w * width_ratio );
      box.y = static_cast< unsigned >( box.y * height_ratio );
      box.h = static_cast< unsigned >( box.h * height_ratio );
    }

    auto detected_objects = std::make_shared< kwiver::vital::detected_object_set >();

    for( const auto& det : darknet_output )
    {
      kwiver::vital::bounding_box_d bbox( det.x, det.y, det.x + det.w, det.y + det.h );
      auto dot = std::make_shared< kwiver::vital::detected_object_type >(
        m_names[ det.obj_id ], det.prob );

      detected_objects->add(
        std::make_shared< kwiver::vital::detected_object >(
          bbox, det.prob, dot ) );
    }

    output.push_back( detected_objects );
  }

  return output;
}


// -----------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
darknet_detector::priv
::scale_detections(
  const kwiver::vital::detected_object_set_sptr dets,
  const region_info& info )
{
  if( info.scale1 != 1.0 )
  {
    kwiver::vital::scale_detections( dets, info.scale1 );
  }

  if( info.shiftx != 0 || info.shifty != 0 )
  {
    kwiver::vital::shift_detections( dets, info.shiftx, info.shifty );
  }

  if( info.scale2 != 1.0 )
  {
    kwiver::vital::scale_detections( dets, info.scale2 );
  }

  const int dist = info.edge_filter;

  if( dist <= 0 )
  {
    return dets;
  }

  const image_rect& roi = info.original_roi;

  std::vector< kwiver::vital::detected_object_sptr > filtered_dets;

  for( auto det : *dets )
  {
    if( !det )
    {
      continue;
    }
    if( roi.x > 0 && det->bounding_box().min_x() < roi.x + dist )
    {
      continue;
    }
    if( roi.y > 0 && det->bounding_box().min_y() < roi.y + dist )
    {
      continue;
    }
    if( det->bounding_box().max_x() > roi.x + roi.width - dist )
    {
      continue;
    }
    if( det->bounding_box().max_y() > roi.y + roi.height - dist )
    {
      continue;
    }

    filtered_dets.push_back( det );
  }

  return kwiver::vital::detected_object_set_sptr(
    new kwiver::vital::detected_object_set( filtered_dets ) );
}


} // end namespace
